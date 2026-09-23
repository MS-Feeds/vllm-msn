#!/usr/bin/env python3
"""A bash sandbox over one SWE-bench instance's Docker image, for the LIVE
agentic rows (`agentic.py::AgenticTurns`).

Only the live path uses this. Replay (`datasets/prep_swebench_agent_replay.py`)
reads observations a dense run already recorded, so the whole keep-rate sweep
runs with no Docker at all; Docker is needed once to record, and again only for
the handful of live configurations.

## Scope

`execute()` and `get_patch()`, nothing else. This is not an agent, not a
scaffold, and deliberately not a reimplementation of `swebench.harness` --
it is the two operations a ReAct loop needs from a container.

## Image names are looked up, never constructed

SWE-bench's instance image naming has a non-obvious encoding (the `__` in an
instance id is not preserved verbatim, and the whole key is case-normalized).
Rather than hardcode a transformation that would silently produce
`pull access denied` on a rename, `image_for_instance` asks the installed
`swebench` package for the key through its own `make_test_spec`. If `swebench`
is not installed the error says so, rather than this module guessing.

## Failure policy

A command that fails is NORMAL -- a wrong path, a failing test, a syntax error
are all things the agent must see and react to, so a non-zero return code comes
back as data. What raises is the sandbox being unusable at all (the container
died, `docker` is missing, the image cannot be pulled): those are harness
faults, and continuing would silently feed the model empty observations that
look like successful no-op commands.

A command that exceeds `timeout` is reported to the agent as a timeout
observation rather than raised, because a hung command is itself something
agents cause and must recover from.

Requires x86_64 Linux with Docker. The SWE-bench harness baseline is 120 GB
free disk, 16 GB RAM and 8 cores; recording or running live over a whole split
pulls one image per repository-version, so disk is usually what binds first.
"""

from __future__ import annotations

import shutil
import subprocess
import uuid
from dataclasses import dataclass, field

#: Where SWE-bench images check the repository out. Constant across the
#: official images; overridable per instance for a non-standard one.
DEFAULT_WORKDIR = "/testbed"

#: What a timed-out command reports back to the model. Phrased as an
#: observation, not an error, because the agent is expected to react to it.
TIMEOUT_TEMPLATE = (
    "<returncode>124</returncode>\n"
    "<output>\nCommand timed out after {timeout} seconds and was killed.\n</output>"
)

OBSERVATION_TEMPLATE = "<returncode>{returncode}</returncode>\n<output>\n{output}\n</output>"


class SandboxError(RuntimeError):
    """The sandbox itself is unusable -- distinct from a command that ran and
    failed, which is returned as data."""


def image_for_instance(instance: dict) -> str:
    """The official SWE-bench image key for one dataset row.

    Asked of the installed `swebench` package rather than reconstructed here;
    see this module's docstring.
    """
    try:
        from swebench.harness.test_spec.test_spec import make_test_spec
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SandboxError(
            "The `swebench` package is required to resolve an instance's "
            "Docker image name. Install it with `pip install swebench`. This "
            "module deliberately does not reconstruct the image key itself -- "
            "the encoding is non-obvious and a wrong guess surfaces as an "
            "opaque `pull access denied`."
        ) from exc
    return make_test_spec(instance).instance_image_key


@dataclass
class DockerSandbox:
    """One long-lived container per instance.

    Long-lived rather than one `docker run` per command, because the agent's
    filesystem edits must persist across steps -- a fresh container per command
    would discard every edit and the loop could never build up a patch.
    """

    image: str
    workdir: str = DEFAULT_WORKDIR
    #: Per-command wall-clock cap. Agents routinely launch full test suites.
    timeout: int = 120
    #: Cap on characters returned to the caller. The TOKEN cap that the budget
    #: depends on is applied upstream in `agentic.py` (it needs the tokenizer);
    #: this is a cheap guard so a `cat` of a binary cannot move megabytes
    #: through a pipe before that happens.
    max_output_chars: int = 100_000
    container: str = field(default="", init=False)

    def __post_init__(self) -> None:
        if not self.container:
            self.container = f"specprefill-{uuid.uuid4().hex[:12]}"

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if shutil.which("docker") is None:
            raise SandboxError(
                "`docker` is not on PATH. The live agentic rows need a "
                "container per instance; the replay rows need none, so if you "
                "are running the keep-rate sweep you are on the wrong path."
            )
        proc = subprocess.run(
            ["docker", "run", "-d", "--rm", "--name", self.container,
             "-w", self.workdir, self.image, "sleep", "infinity"],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise SandboxError(
                f"could not start container from {self.image!r}: "
                f"{proc.stderr.strip()}"
            )

    def stop(self) -> None:
        """Best-effort teardown. Never raises: this runs in a `finally`, and a
        failure to remove a container must not mask the exception that got us
        there. `--rm` already covers the normal path."""
        subprocess.run(["docker", "rm", "-f", self.container],
                       capture_output=True, text=True)

    def __enter__(self) -> "DockerSandbox":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()

    # -- operations --------------------------------------------------------

    def execute(self, command: str) -> tuple[int, str]:
        """Run one shell command. Returns `(returncode, combined_output)`.

        stdout and stderr are combined deliberately: the agent sees one stream,
        the way it would in a terminal, and interleaving order is part of what
        makes a traceback readable.
        """
        try:
            proc = subprocess.run(
                ["docker", "exec", "-w", self.workdir, self.container,
                 "bash", "-lc", command],
                capture_output=True, text=True, timeout=self.timeout,
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return 124, f"Command timed out after {self.timeout} seconds."
        output = (proc.stdout or "") + (proc.stderr or "")
        if len(output) > self.max_output_chars:
            omitted = len(output) - self.max_output_chars
            output = (output[: self.max_output_chars]
                      + f"\n... [{omitted} characters omitted]")
        return proc.returncode, output

    def get_patch(self, exclude_untracked: bool = False) -> str:
        """The agent's cumulative edits as a unified diff.

        `git add -A` first so files the agent CREATED are in the diff --
        SWE-bench solutions routinely add a test or a module, and a bare
        `git diff` would omit them and score the instance unresolved for a
        reason that has nothing to do with the model.
        """
        if not exclude_untracked:
            self.execute("git add -A")
        code, output = self.execute("git diff --cached")
        if code != 0:
            raise SandboxError(f"`git diff --cached` failed in {self.container}: {output}")
        return output


def format_observation(returncode: int, output: str) -> str:
    """The observation text injected as the next turn's input.

    Matches mini-swe-agent's `<returncode>`/`<output>` framing so a live
    trajectory and a recorded one are the same shape -- which is what lets a
    Phase 1 operating point be read against a Phase 2 run at all.
    """
    return OBSERVATION_TEMPLATE.format(returncode=returncode, output=output)
