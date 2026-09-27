#!/usr/bin/env python3
"""A bash sandbox over one SWE-bench instance's container image, for the LIVE
agentic rows (`agentic.py::AgenticTurns`).

Only the live path uses this. Replay (`datasets/prep_swebench_agent_replay.py`)
reads observations a dense run already recorded, so the whole keep-rate sweep
runs with no sandbox at all.

## Two backends

- **docker** -- one long-lived container per instance (`docker run -d ... sleep
  infinity`), one `docker exec` per command.
- **udocker** -- `pip install udocker`. Pure Python, no root, no daemon, built
  for HPC nodes where Docker is unavailable. `udocker create` extracts the
  image to a directory once; each `udocker run` is a fresh process over that
  same directory.

The agent's edits must survive across steps, or it can never build up a patch.
Docker gets that from the container staying alive; udocker gets it from the
extracted rootfs persisting on disk. Confirmed on a real node before this
backend was written:

    udocker run probe bash -lc "echo hi > /tmp/persist_check"
    udocker run probe bash -lc "cat /tmp/persist_check"   # -> hi

## Scope

`execute()` and `get_patch()`, nothing else. This is not an agent, not a
scaffold, and deliberately not a reimplementation of `swebench.harness` -- it
is the two operations a ReAct loop needs from a container.

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
back as data. What raises is the sandbox being unusable at all (the image
cannot be pulled, the runtime is missing): those are harness faults, and
continuing would silently feed the model empty observations that look like
successful no-op commands.

A command that exceeds `timeout` is reported to the agent as a timeout
observation rather than raised, because a hung command is itself something
agents cause and must recover from.

Disk: a few GB per repository-version under the image store (`$UDOCKER_DIR`
for udocker, `/var/lib/docker` otherwise). Under udocker's PRoot backend,
execution is several times slower than native, since syscalls are intercepted
via ptrace -- fine for the file reads and greps that dominate agent
exploration, slow for full test suites.
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field

#: Where SWE-bench images check the repository out. Constant across the
#: official images; overridable per instance for a non-standard one.
DEFAULT_WORKDIR = "/testbed"

OBSERVATION_TEMPLATE = "<returncode>{returncode}</returncode>\n<output>\n{output}\n</output>"

#: udocker prints a banner on EVERY `run`, to stdout, interleaved with the
#: command's own output:
#:
#:      ******************************************************
#:      *                                                    *
#:      *          STARTING 05931124-3b73-3d60-...           *
#:      *                                                    *
#:      ******************************************************
#:      executing: bash
#:
#: Left in, it is injected into every observation the model sees -- tens of
#: wasted tokens per turn, a corrupted trajectory, and a replay dataset full of
#: udocker noise that has nothing to do with the task. `--quiet` suppresses it
#: on most versions; this strips it regardless, because a version that ignores
#: the flag would silently poison every recorded trajectory.
#:
#: Anchored on the asterisk RULES rather than on "STARTING", so it cannot eat a
#: line of the command's real output that merely happens to mention starting.
_UDOCKER_BANNER_RE = re.compile(
    r"^[ \t]*\*{10,}[ \t]*\n(?:.*\n)*?^[ \t]*\*{10,}[ \t]*\n(?:^[ \t]*executing:.*\n?)?",
    re.MULTILINE,
)


class SandboxError(RuntimeError):
    """The sandbox itself is unusable -- distinct from a command that ran and
    failed, which is returned as data."""


def strip_udocker_banner(text: str) -> str:
    return _UDOCKER_BANNER_RE.sub("", text or "")


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
            "container image name. Install it with `pip install swebench`. "
            "This module deliberately does not reconstruct the image key "
            "itself -- the encoding is non-obvious and a wrong guess surfaces "
            "as an opaque `pull access denied`."
        ) from exc
    return make_test_spec(instance).instance_image_key


@dataclass
class _Sandbox:
    """Shared shape. Subclasses implement `start`, `stop` and `_run`."""

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

    # -- subclass contract -------------------------------------------------

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def _run(self, command: str) -> tuple[int, str]:
        """Run one shell command in the container, returning
        `(returncode, combined_output)` with no truncation applied."""
        raise NotImplementedError

    # -- shared ------------------------------------------------------------

    def __enter__(self) -> "_Sandbox":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()

    def execute(self, command: str) -> tuple[int, str]:
        """Run one shell command. Returns `(returncode, combined_output)`.

        stdout and stderr are combined deliberately: the agent sees one stream,
        the way it would in a terminal, and interleaving order is part of what
        makes a traceback readable.
        """
        returncode, output = self._run(command)
        if len(output) > self.max_output_chars:
            omitted = len(output) - self.max_output_chars
            output = (output[: self.max_output_chars]
                      + f"\n... [{omitted} characters omitted]")
        return returncode, output

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
            raise SandboxError(
                f"`git diff --cached` failed in {self.container}: {output}")
        return output

    def _in_workdir(self, command: str) -> str:
        return f"cd {shlex.quote(self.workdir)} && {command}"


@dataclass
class DockerSandbox(_Sandbox):
    """One long-lived container per instance.

    Long-lived rather than one `docker run` per command, because the agent's
    filesystem edits must persist across steps -- a fresh container per command
    would discard every edit and the loop could never build up a patch.
    """

    binary: str = "docker"

    def start(self) -> None:
        if shutil.which(self.binary) is None:
            raise SandboxError(
                f"`{self.binary}` is not on PATH. The live agentic rows need a "
                f"container per instance; the replay rows need none, so if you "
                f"are running the keep-rate sweep you are on the wrong path. "
                f"On a node without Docker, try --sandbox-backend udocker."
            )
        proc = subprocess.run(
            [self.binary, "run", "-d", "--rm", "--name", self.container,
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
        subprocess.run([self.binary, "rm", "-f", self.container],
                       capture_output=True, text=True)

    def _run(self, command: str) -> tuple[int, str]:
        try:
            proc = subprocess.run(
                [self.binary, "exec", "-w", self.workdir, self.container,
                 "bash", "-lc", command],
                capture_output=True, text=True, timeout=self.timeout,
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return 124, f"Command timed out after {self.timeout} seconds."
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


@dataclass
class UdockerSandbox(_Sandbox):
    """udocker: no daemon, no root, one extracted rootfs per instance.

    Differences from Docker that the implementation has to absorb:

    - **No `exec`.** There is no running container to enter; each `udocker run`
      starts a fresh process over the same extracted directory. State persists
      because the directory does, not because a process stays alive.
    - **No `-w`.** The working directory is set by prefixing `cd` to the
      command rather than by a flag, so this does not depend on which udocker
      versions support `--workdir`.
    - **A banner on every run**, stripped here -- see `_UDOCKER_BANNER_RE`.
    - **`pull` is a separate step.** `create` does not fetch, so an image that
      has not been pulled fails at create with a message about a missing
      image rather than fetching it.
    """

    binary: str = "udocker"

    def start(self) -> None:
        if shutil.which(self.binary) is None:
            raise SandboxError(
                f"`{self.binary}` is not on PATH. Install it with "
                f"`pip install udocker && udocker install` -- it needs no root "
                f"and no daemon. Set UDOCKER_DIR to a large filesystem first; "
                f"SWE-bench images are a few GB each."
            )
        pull = subprocess.run(
            [self.binary, "--quiet", "pull", self.image],
            capture_output=True, text=True,
        )
        if pull.returncode != 0:
            raise SandboxError(
                f"`udocker pull {self.image}` failed: "
                f"{strip_udocker_banner(pull.stderr).strip()}"
            )
        create = subprocess.run(
            [self.binary, "--quiet", "create", f"--name={self.container}",
             self.image],
            capture_output=True, text=True,
        )
        if create.returncode != 0:
            raise SandboxError(
                f"`udocker create` from {self.image!r} failed: "
                f"{strip_udocker_banner(create.stderr).strip()}"
            )

    def stop(self) -> None:
        """Best-effort teardown. Unlike Docker's `--rm` there is nothing
        automatic here: a container left behind is an extracted rootfs of a few
        GB, so failing to remove it leaks disk rather than a process."""
        subprocess.run([self.binary, "--quiet", "rm", self.container],
                       capture_output=True, text=True)

    def _run(self, command: str) -> tuple[int, str]:
        try:
            proc = subprocess.run(
                [self.binary, "--quiet", "run", self.container,
                 "bash", "-lc", self._in_workdir(command)],
                capture_output=True, text=True, timeout=self.timeout,
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return 124, f"Command timed out after {self.timeout} seconds."
        combined = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, strip_udocker_banner(combined)


#: Name -> class, for `--sandbox-backend`.
SANDBOX_BACKENDS = {
    "docker": DockerSandbox,
    "podman": lambda **kw: DockerSandbox(binary="podman", **kw),
    "udocker": UdockerSandbox,
}


def make_sandbox(backend: str, **kwargs):
    """Construct (but do not start) a sandbox for `backend`.

    `podman` reuses `DockerSandbox` rather than getting its own class: its
    `run`/`exec`/`rm` surface is CLI-compatible for everything used here, so a
    separate implementation would be a copy that could only drift.
    """
    try:
        factory = SANDBOX_BACKENDS[backend]
    except KeyError:
        raise SandboxError(
            f"unknown sandbox backend {backend!r}; expected one of "
            f"{sorted(SANDBOX_BACKENDS)}"
        ) from None
    return factory(**kwargs)


def format_observation(returncode: int, output: str) -> str:
    """The observation text injected as the next turn's input.

    Matches mini-swe-agent's `<returncode>`/`<output>` framing so a live
    trajectory and a recorded one are the same shape -- which is what lets a
    Phase 1 operating point be read against a Phase 2 run at all.
    """
    return OBSERVATION_TEMPLATE.format(returncode=returncode, output=output)
