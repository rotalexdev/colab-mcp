# Copyright 2026 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Smoke test for a PyInstaller-built colab-mcp binary.

A successful PyInstaller build proves nothing: the bundle can be produced and
still be unable to start on the target machine. This test actually executes the
artifact and fails loudly when it does not work.

Two checks:

1. ``--help`` exits 0. This catches PyInstaller loader/import failures, missing
   shared objects and mangled native extensions, because the bundle dies during
   module import before argparse ever runs.
2. A real MCP JSON-RPC handshake over stdio returns ``serverInfo`` for
   ``initialize`` and a ``tools`` array for ``tools/list``.

Standard library only, cross-platform. Usage:

    python .github/smoke_test.py <path-to-binary>
"""

import json
import os
import platform
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time

IS_WINDOWS = os.name == "nt"

# Total budget for the MCP handshake. Generous, because a PyInstaller onefile
# binary extracts itself to a temp directory on first run.
HANDSHAKE_TIMEOUT = 60.0
# Per-read wait in the stdout collection loop, so the deadline is polled often.
POLL_INTERVAL = 1.0
# `--help` should be near-instant, but allow for first-run extraction.
HELP_TIMEOUT = 60.0
# Grace period after closing stdin, before escalating to SIGTERM/SIGKILL.
SHUTDOWN_GRACE = 5.0
# How many lines of stdout/stderr to dump in a failure diagnostic.
TAIL_LINES = 40

INITIALIZE_MSG = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "smoke", "version": "0.0.0"},
    },
}
INITIALIZED_MSG = {"jsonrpc": "2.0", "method": "notifications/initialized"}
TOOLS_LIST_MSG = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}

# Substrings that indicate the bundle itself blew up rather than answering.
TRACEBACK_MARKERS = (
    "Traceback (most recent call last)",
    "ImportError",
    "ModuleNotFoundError",
    "No module named",
    "cannot open shared object file",
    "Image not found",
    "failed to execute",
    "FreeLibrary",
)


def tail(text, limit=TAIL_LINES):
    """Return the last `limit` non-empty-stripped lines of `text`."""
    if not text:
        return "(empty)"
    lines = text.splitlines()
    kept = lines[-limit:]
    prefix = "... (truncated)\n" if len(lines) > limit else ""
    return prefix + "\n".join(kept)


def child_env():
    """Environment for the child, forcing deterministic UTF-8 stdio."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def spawn(binary, args, **kwargs):
    """Start `binary` in its own process group / console group.

    PyInstaller's onefile bootloader runs the real interpreter as a *child*
    process, so killing only the parent would orphan the child. Starting the
    process in a fresh group lets us signal the whole tree on termination.
    """
    if IS_WINDOWS:
        kwargs["creationflags"] = getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(
        [binary, *args],
        env=child_env(),
        encoding="utf-8",
        errors="replace",
        **kwargs,
    )


def terminate_tree(proc):
    """Terminate `proc` and any PyInstaller child it spawned.

    PyInstaller's onefile bootloader runs the real interpreter as a child
    process, so signalling only the parent leaks the child -- and a leaked
    child keeps the stdout pipe open, which would hang the reader thread.

    We therefore signal the whole process group. We do not stop once the
    parent is gone: a child that ignores SIGTERM must still get SIGKILL.
    """
    if proc.poll() is not None:
        # Already exited. Do not signal a group for a reaped pid: the group id
        # may have been recycled and we would hit an unrelated process.
        return

    if IS_WINDOWS:
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        return

    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        pgid = None

    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            if pgid is not None:
                os.killpg(pgid, sig)
            else:
                if proc.poll() is not None:
                    return
                proc.send_signal(sig)
        except OSError:
            # ESRCH: the group is empty, so there is nothing left to kill.
            return

        try:
            proc.wait(timeout=10 if sig == signal.SIGTERM else 5)
        except subprocess.TimeoutExpired:
            continue


def diagnostic_block(title, binary, exit_code, stdout_tail, stderr_tail):
    """Build the clearly labelled failure block written to the CI log."""
    if exit_code is None:
        exit_code = "still running (never exited)"
    rule = "=" * 72
    return (
        "\n"
        f"{rule}\n"
        "SMOKE TEST FAILURE\n"
        f"{rule}\n"
        f"reason         : {title}\n"
        f"binary         : {binary}\n"
        f"exit code      : {exit_code}\n"
        "--- stdout (tail) ---\n"
        f"{stdout_tail}\n"
        "--- stderr (tail) ---\n"
        f"{stderr_tail}\n"
        f"{rule}\n"
    )


def report_failure(title, binary, exit_code, stdout_tail, stderr_tail):
    print(diagnostic_block(title, binary, exit_code, stdout_tail, stderr_tail))
    print(f"::error title=Binary smoke test failed::{title}")


# --------------------------------------------------------------------------
# Check 1: does the binary start at all?
# --------------------------------------------------------------------------


def check_help(binary):
    """Run `--help` and require exit code 0.

    Returns (ok, summary, exit_code, stdout_text, stderr_text).
    """
    try:
        proc = subprocess.run(
            [binary, "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=HELP_TIMEOUT,
            env=child_env(),
        )
    except subprocess.TimeoutExpired:
        return (
            False,
            f"`--help` did not exit within {HELP_TIMEOUT:.0f}s",
            None,
            "",
            "",
        )
    except OSError as exc:
        return False, f"could not execute the binary: {exc}", None, "", ""

    stdout_text = proc.stdout or ""
    stderr_text = proc.stderr or ""

    if proc.returncode != 0:
        return (
            False,
            f"`--help` exited {proc.returncode} (expected 0); "
            "the bundle most likely failed to import one of its modules",
            proc.returncode,
            stdout_text,
            stderr_text,
        )

    summary = next(
        (line.strip() for line in stdout_text.splitlines() if line.strip()),
        "(no help output)",
    )
    return True, summary, proc.returncode, stdout_text, stderr_text


# --------------------------------------------------------------------------
# Check 2: real MCP JSON-RPC stdio handshake
# --------------------------------------------------------------------------


def read_stdout(stream, out_queue):
    """Drain stdout line by line into `out_queue`, then push a sentinel.

    Reading in a thread is what keeps this test from deadlocking: the pipe
    buffer is finite, so a server that writes more than it reads would block
    forever if we only read after `communicate()`.
    """
    try:
        for line in iter(stream.readline, ""):
            out_queue.put(line)
    except Exception:
        pass
    finally:
        out_queue.put(None)


def collect_responses(proc, out_queue, deadline, seen_lines, tracebacks):
    """Read JSON-RPC responses from the queue until both ids arrive or we run out.

    Returns (responses, stop_reason) where responses is keyed by stringified
    request id and stop_reason is one of "complete", "exited" or "timeout".
    The distinction matters: "exited" means the bundle crashed or closed
    stdout, "timeout" means it hung.
    """
    responses = {}
    while len(responses) < 2:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return responses, "timeout"
        try:
            line = out_queue.get(timeout=min(POLL_INTERVAL, remaining))
        except queue.Empty:
            # No output yet; if the process died we will never get more.
            if proc.poll() is not None:
                return responses, "exited"
            continue

        if line is None:
            # Reader thread hit EOF: the process closed stdout.
            return responses, "exited"

        stripped = line.strip()
        if not stripped:
            continue

        seen_lines.append(stripped)

        for marker in TRACEBACK_MARKERS:
            if marker in stripped:
                tracebacks.append(stripped)
                break

        try:
            message = json.loads(stripped)
        except ValueError:
            # Banner or other non-JSON noise on stdout; not fatal on its own.
            continue

        if isinstance(message, dict) and "id" in message:
            responses[str(message["id"])] = message

    return responses, "complete"


def check_mcp_handshake(binary):
    """Perform initialize + notifications/initialized + tools/list.

    Returns (ok, summary, exit_code, stdout_tail, stderr_tail). `exit_code` is
    the process' final return code, or None if it never exited.
    """
    proc = None
    seen_lines = []
    tracebacks = []

    with tempfile.TemporaryDirectory(prefix="colab-mcp-smoke-") as tmp:
        log_dir = os.path.join(tmp, "logs")
        os.makedirs(log_dir, exist_ok=True)
        stderr_path = os.path.join(tmp, "stderr.log")
        stderr_file = open(stderr_path, "w+", encoding="utf-8", errors="replace")

        try:
            proc = spawn(
                binary,
                ["--log", log_dir],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr_file,
            )
        except OSError as exc:
            stderr_file.seek(0)
            return (
                False,
                f"could not start the binary: {exc}",
                None,
                "",
                tail(stderr_file.read()),
            )

        out_queue = queue.Queue()
        reader = threading.Thread(
            target=read_stdout,
            args=(proc.stdout, out_queue),
            daemon=True,
        )
        reader.start()

        write_error = None
        try:
            for message in (INITIALIZE_MSG, INITIALIZED_MSG, TOOLS_LIST_MSG):
                proc.stdin.write(json.dumps(message) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            write_error = str(exc)

        deadline = time.monotonic() + HANDSHAKE_TIMEOUT
        if write_error:
            responses, stop_reason = {}, "write_failed"
        else:
            responses, stop_reason = collect_responses(
                proc, out_queue, deadline, seen_lines, tracebacks
            )

        # Always clean up before inspecting anything. Closing stdin is the
        # documented shutdown signal for an MCP stdio server, so give it a
        # short grace period to exit on its own before escalating to signals.
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=SHUTDOWN_GRACE)
        except subprocess.TimeoutExpired:
            pass
        terminate_tree(proc)

        # Only close stdout once the reader thread is done with it. If some
        # process still held the write end, close() would block forever on the
        # read lock the thread is holding.
        reader.join(timeout=SHUTDOWN_GRACE)
        if not reader.is_alive():
            try:
                if proc.stdout and not proc.stdout.closed:
                    proc.stdout.close()
            except OSError:
                pass

        stderr_file.flush()
        stderr_file.seek(0)
        stderr_text = stderr_file.read()

    exit_code = None if proc is None else proc.returncode
    stdout_tail = tail("\n".join(seen_lines))
    stderr_tail = tail(stderr_text)

    if write_error:
        return (
            False,
            f"could not send the MCP handshake (broken pipe: {write_error}); "
            "the process exited before reading stdin",
            exit_code,
            stdout_tail,
            stderr_tail,
        )

    if tracebacks:
        return (
            False,
            "a Python traceback was written to stdout: " + tracebacks[0],
            exit_code,
            stdout_tail,
            stderr_tail,
        )

    # Distinguish a bundle that crashed from one that hung: the two failures
    # have completely different causes and the wording must not mislead.
    if stop_reason == "exited":
        missing = (
            "no JSON-RPC response with id=2 (tools/list)"
            if "1" in responses
            else "no JSON-RPC response with id=1 (initialize)"
        )
        how = (
            f"the process exited after answering {sorted(responses)} "
            f"but before {missing}"
            if responses
            else "the process exited before answering the MCP initialize request"
        )
    elif stop_reason == "timeout":
        how = (
            f"no response within {HANDSHAKE_TIMEOUT:.0f}s; the process is alive "
            "but never answered"
        )
    else:
        how = "no response received"

    initialize = responses.get("1")
    if initialize is None:
        return (
            False,
            f"{how} (the MCP initialize handshake never completed)",
            exit_code,
            stdout_tail,
            stderr_tail,
        )

    if "error" in initialize:
        return (
            False,
            f"initialize returned a JSON-RPC error: {initialize['error']!r}",
            exit_code,
            stdout_tail,
            stderr_tail,
        )

    server_info = initialize.get("result", {}).get("serverInfo")
    if not isinstance(server_info, dict) or "name" not in server_info:
        return (
            False,
            f"initialize response id=1 has no result.serverInfo: "
            f"{initialize.get('result')!r}",
            exit_code,
            stdout_tail,
            stderr_tail,
        )

    tools_list = responses.get("2")
    if tools_list is None:
        return (
            False,
            f"initialize succeeded but tools/list was never answered ({how})",
            exit_code,
            stdout_tail,
            stderr_tail,
        )

    if "error" in tools_list:
        return (
            False,
            f"tools/list returned a JSON-RPC error: {tools_list['error']!r}",
            exit_code,
            stdout_tail,
            stderr_tail,
        )

    tools = tools_list.get("result", {}).get("tools")
    if not isinstance(tools, list) or not tools:
        return (
            False,
            f"tools/list response id=2 has no non-empty result.tools array: "
            f"{tools_list.get('result')!r}",
            exit_code,
            stdout_tail,
            stderr_tail,
        )

    tool_names = [tool.get("name", "?") for tool in tools if isinstance(tool, dict)]
    return (
        True,
        f"server={server_info.get('name')} "
        f"{server_info.get('version', '')}".strip()
        + f" tools={tool_names}",
        exit_code,
        stdout_tail,
        stderr_tail,
    )


# --------------------------------------------------------------------------


def main(argv):
    if len(argv) != 2:
        print(
            f"usage: {os.path.basename(argv[0] or 'smoke_test.py')} <path-to-binary>",
            file=sys.stderr,
        )
        return 2

    binary = argv[1]
    if not os.path.isfile(binary):
        print(f"::error title=Binary smoke test failed::binary not found: {binary}")
        print(f"smoke test: {binary} does not exist", file=sys.stderr)
        return 1

    binary = os.path.abspath(binary)
    print(
        f"Smoke testing {binary} "
        f"({platform.system()} {platform.machine()}, python {sys.version.split()[0]})"
    )

    ok, summary, exit_code, stdout_text, stderr_text = check_help(binary)
    if ok:
        print(f"  [1/2] --help OK: {summary}")
    else:
        report_failure(summary, binary, exit_code, tail(stdout_text), tail(stderr_text))
        return 1

    ok, summary, exit_code, stdout_tail, stderr_tail = check_mcp_handshake(binary)
    if ok:
        print(f"  [2/2] MCP handshake OK: {summary}")
        print("Smoke test passed.")
        return 0

    report_failure(summary, binary, exit_code, stdout_tail, stderr_tail)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
