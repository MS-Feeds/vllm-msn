#!/usr/bin/env python3
"""Where a turn's content comes from -- the dataset, or a live sandbox.

## Why this exists instead of a second driving loop

`predict_scbench.py`'s two multi-turn loops (`run_baseline`,
`run_sparse_attention`) touch dataset turns in exactly two places: Phase 1 reads
`conv["turns"][turn_idx]`, and the harvest retires a session once
`turn_idx >= len(conv["turns"])`. Everything else in those loops -- the
resumable session, the ledger/position-map bookkeeping, wave accounting, FLOP
attribution, dense-fallback detection -- is independent of where the text came
from.

So the live agentic path is NOT a third loop. It is a `TurnSource` swapped in
behind those two reads. That matters for more than line count: `EXPERIMENT_PLAN
.md`'s sparse-attention section records eight findings that were each a real
debugging pass on hardware -- cumulative-`token_ids` ledger drift, the
`<|eot_id|>` stop-set bug from bypassing `InputProcessor.process_inputs()`, the
async-scheduling re-park race, three separate performance cliffs. A sibling loop
would reimplement the surfaces those findings live on and would be free to
re-acquire every one of them. Injecting a turn source inherits all of it.

## The two implementations

- `DatasetTurns` -- the default, and byte-identical to the previous inline
  behaviour: hand back `conv["turns"][turn_idx]`, retire at the end of the list.
  Used by every replay and every existing row.
- `AgenticTurns` -- turn 0 is the issue statement from the samples file; every
  later turn is the OUTPUT OF EXECUTING what the model just generated. Retires
  on submit, on the step limit, or when an action cannot be parsed too many
  times running.

## Teacher forcing is exactly the difference

Under `DatasetTurns` the arm is snapped back onto the recorded dense trajectory
every turn, which is what makes a keep-rate sweep paired and per-turn gradable.
Under `AgenticTurns` nothing snaps it back: arms diverge, and by turn 5 two
configurations are in different repository states. That is the point -- it is
the only way to see divergence compounding -- but it means per-turn accuracy is
undefined for these rows and the endpoints must be conversation-level
(wall clock, sustained tokens/s, submission rate, resolve rate). See
`grade_scbench.py`'s `swebench_agent` entry for the replay side's converse.

## The budget behaves differently here, on purpose

For replay, `prep_swebench_agent_replay.py` verifies every conversation against
the driver's pre-flight ahead of time, so `num_skipped_too_large` MUST come back
0 -- a non-zero there means the prep-time and run-time checks disagree. Live,
observation lengths are not knowable in advance: `max_observation_tokens`
bounds them (`context + max_turns * (cap + max_tokens)` is the worst case), but
a long system prompt or an unlucky instance can still exceed the engine budget,
and the driver's own pre-flight retires that conversation cleanly at the turn
it would have overrun. **A non-zero `num_skipped_too_large` is therefore
EXPECTED on live rows and must be read as an exit reason, not a bug.**
"""

from __future__ import annotations

import re
import time
from typing import Optional

from agent_sandbox import DockerSandbox, SandboxError, format_observation, image_for_instance
from grade_scbench import extract_agent_action

#: Sent back when the generation carried no single bash block. Mirrors what a
#: scaffold does rather than retiring the conversation: a model that fumbles
#: one action usually recovers, and killing the run on the first fumble would
#: make the aggressive keep-rate arms look worse for a reason unrelated to
#: selection quality.
FORMAT_REMINDER = (
    "<returncode>1</returncode>\n"
    "<output>\nYour last message did not contain exactly one bash command "
    "block. Provide exactly one command, formatted as:\n\n"
    "```bash\nyour_command_here\n```\n</output>"
)

#: How the agent says it is done. MUST agree with the system prompt in the
#: samples file's `context` -- this module cannot verify that, and a mismatch
#: shows up as every conversation exhausting its step limit with a submission
#: rate of zero.
DEFAULT_SUBMIT_RE = re.compile(
    r"(^|\s)(submit|echo\s+MINI_SWE_AGENT_FINAL_OUTPUT)(\s|$)")


class DatasetTurns:
    """Turns come from the samples file. The pre-existing behaviour."""

    #: Read by the driver to decide whether to call `observe` at all, so the
    #: replay path does no per-turn work it did not do before.
    live = False

    def turn_for(self, conv: dict, turn_idx: int) -> Optional[dict]:
        turns = conv["turns"]
        if turn_idx >= len(turns):
            return None
        return turns[turn_idx]

    def observe(self, conv: dict, turn_idx: int, pred_text: str) -> None:
        """Nothing to do -- the next turn is already in the file."""

    def is_exhausted(self, conv: dict, turn_idx: int) -> bool:
        return turn_idx >= len(conv["turns"])

    def teardown(self, conv: dict) -> None:
        """No resources held."""

    def close(self) -> None:
        """No resources held."""


class AgenticTurns:
    """Turns come from executing the model's own actions in a container.

    One sandbox per conversation, created lazily on turn 0 and torn down when
    the conversation retires. The driver calls `teardown` from its own retire
    path, so a conversation that dies on a pre-flight check still releases its
    container.
    """

    live = True

    def __init__(self, tok, *, max_turns: int, max_observation_tokens: int,
                 instances_by_id: dict, timeout: int = 120,
                 max_format_retries: int = 3,
                 submit_re: re.Pattern = DEFAULT_SUBMIT_RE,
                 workdir: Optional[str] = None,
                 on_event=None,
                 sandbox_factory=None):
        self.tok = tok
        self.max_turns = max_turns
        self.max_observation_tokens = max_observation_tokens
        #: The raw SWE-bench dataset rows, keyed by instance_id. Needed only to
        #: resolve each instance's Docker image; the conversation itself comes
        #: from the samples file.
        self.instances_by_id = instances_by_id
        self.timeout = timeout
        self.max_format_retries = max_format_retries
        self.submit_re = submit_re
        self.workdir = workdir
        #: `f(str)` for diagnostics. Defaults to `print` rather than a no-op:
        #: the messages it carries are things like "docker is not on PATH" and
        #: "no SWE-bench row for this instance", which would otherwise be
        #: visible only as an exit_reason column after the run finished. The
        #: cost is that a line can tear tqdm's progress bar, which is the
        #: better trade for a failure that affects every conversation.
        self.on_event = on_event or print
        #: `f(instance_row) -> sandbox`. The seam the tests use: the control
        #: flow worth testing (format retries, submit detection, truncation,
        #: the step limit) is all in this class, and none of it needs Docker.
        #: Defaults to a started `DockerSandbox` for the real path.
        self.sandbox_factory = sandbox_factory or self._default_sandbox

        self._sandboxes: dict[str, DockerSandbox] = {}
        self._pending: dict[str, str] = {}
        self._finished: dict[str, bool] = {}
        self._format_failures: dict[str, int] = {}
        #: Conversation-level endpoints. These, not per-turn accuracy, are what
        #: a live row reports -- see the module docstring.
        self.records: dict[str, dict] = {}

    # -- TurnSource ---------------------------------------------------------

    def turn_for(self, conv: dict, turn_idx: int) -> Optional[dict]:
        conv_id = conv["id"]
        if self._finished.get(conv_id) or turn_idx >= self.max_turns:
            return None

        if turn_idx == 0:
            self._start(conv)
            if self._finished.get(conv_id):
                # `_start` failed (no matching SWE-bench row, or the container
                # would not come up). Retire NOW rather than spending turn 0 on
                # a conversation whose first `observe` can only fail -- that
                # would charge the arm a full prefill for a harness fault.
                return None
            # Turn 0 is the task, and it is the one turn the samples file still
            # supplies -- the replay file's turn 0 already holds exactly this
            # (the issue statement the recorded agent was handed), which is why
            # one samples file drives both modes.
            return {"input": conv["turns"][0]["input"], "answer": ""}

        observation = self._pending.pop(conv_id, None)
        if observation is None:
            # `observe` retired the conversation, or was never called because
            # the turn produced no output at all. Either way there is nothing
            # to inject and the session should retire rather than repeat a turn.
            return None
        return {"input": observation, "answer": ""}

    def observe(self, conv: dict, turn_idx: int, pred_text: str) -> None:
        """Execute what the model just generated; stash the next turn's input."""
        conv_id = conv["id"]
        record = self.records.get(conv_id)
        if record is None or self._finished.get(conv_id):
            # Unreachable on the driver's own ordering (`turn_for` creates the
            # record and retires on failure before any turn runs). Guarded
            # anyway: this runs once per turn of an hours-long sweep, and a
            # KeyError here would lose the whole run over a bookkeeping slip.
            return
        record["steps"] = turn_idx + 1

        action = extract_agent_action(pred_text)
        if action is None:
            self._format_failures[conv_id] = self._format_failures.get(conv_id, 0) + 1
            record["invalid_actions"] += 1
            if self._format_failures[conv_id] > self.max_format_retries:
                self._retire(conv_id, "format_retries_exhausted")
                return
            self._pending[conv_id] = FORMAT_REMINDER
            return
        self._format_failures[conv_id] = 0

        if self.submit_re.search(action):
            self._capture_patch(conv_id)
            self._retire(conv_id, "submitted")
            return

        sandbox = self._sandboxes.get(conv_id)
        if sandbox is None:
            self._retire(conv_id, "no_sandbox")
            return
        try:
            returncode, output = sandbox.execute(action)
        except SandboxError as exc:
            self.on_event(f"[agentic] {conv_id}: sandbox failed: {exc}")
            self._retire(conv_id, "sandbox_error")
            return

        self._pending[conv_id] = self._truncate(
            format_observation(returncode, output))

    def is_exhausted(self, conv: dict, turn_idx: int) -> bool:
        conv_id = conv["id"]
        if self._finished.get(conv_id):
            return True
        if turn_idx >= self.max_turns:
            # Reached here without submitting: record it, since the exit-reason
            # mix is one of the endpoints.
            self._capture_patch(conv_id)
            self._retire(conv_id, "step_limit")
            return True
        if conv_id not in self._pending:
            # `observe` produced no next observation and did not retire -- the
            # turn generated nothing at all (the driver's `output is None`
            # branch). Named explicitly so it is distinguishable in the
            # exit-reason mix from a conversation the driver's own pre-flight
            # retired.
            self._capture_patch(conv_id)
            self._retire(conv_id, "no_observation")
            return True
        return False

    def teardown(self, conv: dict) -> None:
        conv_id = conv["id"]
        record = self.records.get(conv_id)
        if record is not None and record.get("exit_reason") is None:
            # The driver retired this conversation without going through
            # `observe`/`is_exhausted` -- i.e. one of its own pre-flight checks
            # fired. Expected on live rows; see the module docstring.
            self._capture_patch(conv_id)
            record["exit_reason"] = "driver_retired"
        if record is not None and record.get("wall_seconds") is None:
            record["wall_seconds"] = time.time() - record["t_start"]
        sandbox = self._sandboxes.pop(conv_id, None)
        if sandbox is not None:
            sandbox.stop()
        self._pending.pop(conv_id, None)

    def close(self) -> None:
        """Stop every container still running.

        The driver calls `teardown` on each conversation as it retires, so on
        a clean run this finds nothing. It exists for the run that does NOT
        finish cleanly: without it, an exception anywhere in the turn loop
        leaves one `sleep infinity` container per in-flight conversation alive
        until the machine is rebooted or someone notices. Called from
        `run_experiment`'s `finally`.
        """
        for conv_id, sandbox in list(self._sandboxes.items()):
            sandbox.stop()
            self._sandboxes.pop(conv_id, None)

    # -- internals ----------------------------------------------------------

    def _default_sandbox(self, instance: dict) -> DockerSandbox:
        sandbox = DockerSandbox(
            image=image_for_instance(instance),
            timeout=self.timeout,
            **({"workdir": self.workdir} if self.workdir else {}),
        )
        sandbox.start()
        return sandbox

    def _start(self, conv: dict) -> None:
        conv_id = conv["id"]
        instance_id = conv.get("instance_id")
        self.records.setdefault(conv_id, {
            "conversation_id": conv_id,
            "instance_id": instance_id,
            "steps": 0,
            "invalid_actions": 0,
            "exit_reason": None,
            "patch": "",
            "t_start": time.time(),
            "wall_seconds": None,
        })
        if conv_id in self._sandboxes:
            return
        instance = self.instances_by_id.get(instance_id)
        if instance is None:
            self.on_event(
                f"[agentic] {conv_id}: no SWE-bench row for instance_id="
                f"{instance_id!r} -- cannot resolve a Docker image. Check "
                f"--swebench-dataset against the samples file."
            )
            self._retire(conv_id, "no_instance")
            return
        try:
            sandbox = self.sandbox_factory(instance)
        except SandboxError as exc:
            self.on_event(f"[agentic] {conv_id}: {exc}")
            self._retire(conv_id, "sandbox_start_failed")
            return
        self._sandboxes[conv_id] = sandbox

    def _capture_patch(self, conv_id: str) -> None:
        sandbox = self._sandboxes.get(conv_id)
        record = self.records.get(conv_id)
        if sandbox is None or record is None or record.get("patch"):
            return
        try:
            record["patch"] = sandbox.get_patch()
        except SandboxError as exc:
            # A missing patch is a result (unresolved), not a run failure.
            self.on_event(f"[agentic] {conv_id}: could not read patch: {exc}")

    def _retire(self, conv_id: str, reason: str) -> None:
        self._finished[conv_id] = True
        self._pending.pop(conv_id, None)
        record = self.records.get(conv_id)
        if record is not None and record.get("exit_reason") is None:
            record["exit_reason"] = reason
            record["wall_seconds"] = time.time() - record["t_start"]

    def _truncate(self, observation: str) -> str:
        """Middle-out truncation to `max_observation_tokens`.

        Both ends carry signal -- a traceback's first frames name what was run,
        the last lines name what failed -- so a head-only cut loses more than a
        middle cut. The token cap (not a character cap) is what the budget
        arithmetic depends on, which is why it is applied here with the real
        tokenizer rather than in `agent_sandbox.py`.
        """
        ids = self.tok.encode(observation, add_special_tokens=False)
        if len(ids) <= self.max_observation_tokens:
            return observation
        keep = self.max_observation_tokens
        head = self.tok.decode(ids[: keep // 2], skip_special_tokens=True)
        tail = self.tok.decode(ids[-(keep - keep // 2):], skip_special_tokens=True)
        omitted = len(ids) - keep
        return f"{head}\n... [{omitted} tokens omitted] ...\n{tail}"

    # -- reporting ----------------------------------------------------------

    def predictions_payload(self, model_name: str) -> dict:
        """SWE-bench `preds.json`, for the Docker harness or `sb-cli`.

        Resolve rate is the WEAKEST endpoint these rows produce -- on 500
        instances it detects a ~10-point collapse, not a few points of quality
        cost -- so read it after wall clock, sustained throughput and the
        exit-reason mix, and report its limits alongside it.
        """
        return {
            rec["instance_id"]: {
                "model_name_or_path": model_name,
                "instance_id": rec["instance_id"],
                "model_patch": rec["patch"],
            }
            for rec in self.records.values()
            if rec.get("instance_id")
        }
