"""
verifier_tools.py

Verifier tools for AGENTIC_AI_LAPTOP.

The Verifier is the ACTION agent. Every mutating action happens
inside a live terminal session that the user can watch.

Tools:

    get_session()                — reuse-or-open (preferred)
    open_terminal_session()      — force-open a new terminal
    run_in_shell(id, cmd)        — run a command in the terminal
    close_terminal_session(id)   — close a session

Sessions persist to /tmp/zoya_verifier/sessions.json so they
survive the short-lived tool_server process.
"""

from __future__ import annotations

import datetime
import functools
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Annotated

from pydantic import Field, validate_call


# ============================================================================
# STRUCTURED TOOL ERROR
# ============================================================================

class ToolError(Exception):
    """Structured failure understood by tool_server.py."""

    def __init__(self, code, message, retryable=False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


# ============================================================================
# TOOL REGISTRY
# ============================================================================

TOOLS = {}


def tool(group: str, risk: str = "HIGH"):
    """Register a Verifier tool."""

    def deco(fn):
        checked = validate_call(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return checked(*args, **kwargs)
            except ToolError:
                raise
            except PermissionError as exc:
                raise ToolError("PERMISSION_DENIED", str(exc))
            except FileNotFoundError as exc:
                raise ToolError("NOT_FOUND", str(exc))
            except OSError as exc:
                raise ToolError("OS_ERROR", str(exc))

        doc = (fn.__doc__ or "").strip().splitlines()
        TOOLS[fn.__name__] = {
            "fn": wrapper,
            "group": group,
            "mutating": True,
            "risk": risk,
            "doc": doc[0] if doc else "",
        }
        return wrapper

    return deco


# ============================================================================
# CONFIG
# ============================================================================

SOCKET_DIR = "/tmp/zoya_verifier"
SOCKET_READY_TIMEOUT = 15.0
DEFAULT_COMMAND_TIMEOUT = 120
MAX_COMMAND_TIMEOUT = 1800
MAX_OUTPUT = 32_000

SPAWN_STDERR_LOG = "/tmp/zoya_verifier_spawn_err.log"
OPS_LOG = os.path.abspath("runs/infra_ops.log")

_SESSIONS_FILE = os.path.join(SOCKET_DIR, "sessions.json")


# ============================================================================
# SESSION STATE
# ============================================================================

_SESSIONS: dict[str, dict] = {}


def _load_sessions():
    """Load sessions from disk on module import."""
    try:
        os.makedirs(SOCKET_DIR, exist_ok=True)
        with open(_SESSIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            _SESSIONS.update(data)
    except FileNotFoundError:
        pass
    except Exception:
        pass


def _save_sessions():
    """Persist sessions to disk after every change."""
    try:
        os.makedirs(SOCKET_DIR, exist_ok=True)
        with open(_SESSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(_SESSIONS, f, indent=2)
    except Exception:
        pass


_load_sessions()


# ============================================================================
# HELPERS
# ============================================================================

def _ops_log(line: str):
    try:
        os.makedirs(os.path.dirname(OPS_LOG), exist_ok=True)
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        with open(OPS_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {line}\n")
            f.flush()
    except Exception:
        pass


def _truncate(text: str, limit: int = MAX_OUTPUT) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[output truncated]..."


def _ensure_socket_dir():
    os.makedirs(SOCKET_DIR, exist_ok=True)


def _socket_path_for(session_id: str) -> str:
    return os.path.join(SOCKET_DIR, f"{session_id}.sock")


def _find_host_script() -> str:
    here = Path(__file__).resolve().parent
    candidate = here / "shell_host.py"
    if candidate.is_file():
        return str(candidate)
    raise ToolError(
        "HOST_NOT_FOUND",
        f"shell_host.py not found near {here}",
    )


def _pick_terminal() -> list[str] | None:
    import shutil

    candidates = [
        ("ghostty",             ["ghostty", "-e"]),
        ("x-terminal-emulator", ["x-terminal-emulator", "-e"]),
        ("konsole",             ["konsole", "-e"]),
        ("kitty",               ["kitty"]),
        ("alacritty",           ["alacritty", "-e"]),
        ("wezterm",             ["wezterm", "start", "--"]),
        ("xfce4-terminal",      ["xfce4-terminal", "-e"]),
        ("tilix",               ["tilix", "-e"]),
        ("xterm",               ["xterm", "-e"]),
        ("gnome-terminal",      ["gnome-terminal", "--"]),
    ]

    for binary, argv in candidates:
        if shutil.which(binary):
            return argv

    return None


def _wait_for_socket(sock_path: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(sock_path):
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.settimeout(0.5)
                s.connect(sock_path)
                s.close()
                return True
            except (ConnectionRefusedError, FileNotFoundError, OSError):
                time.sleep(0.2)
                continue
        time.sleep(0.2)
    return False


def _is_socket_alive(sock_path: str) -> bool:
    if not sock_path or not os.path.exists(sock_path):
        return False
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(0.5)
        s.connect(sock_path)
        s.close()
        return True
    except Exception:
        return False


def _send_request(sock_path: str, payload: dict, timeout: float) -> dict:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)

    try:
        s.connect(sock_path)
    except (ConnectionRefusedError, FileNotFoundError) as exc:
        raise ToolError(
            "SESSION_NOT_FOUND",
            f"terminal session is not reachable: {exc}",
        )

    try:
        line = json.dumps(payload) + "\n"
        s.sendall(line.encode())

        chunks = []
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
    finally:
        try:
            s.close()
        except Exception:
            pass

    raw = b"".join(chunks).decode("utf-8", errors="replace").strip()

    if not raw:
        raise ToolError(
            "SESSION_NOT_FOUND",
            "terminal session closed the connection",
        )

    first_line = raw.splitlines()[0]

    try:
        return json.loads(first_line)
    except json.JSONDecodeError as exc:
        raise ToolError(
            "BAD_SESSION_RESPONSE",
            f"terminal session returned invalid JSON: {exc}",
        )


def _ensure_display_env(env: dict) -> dict:
    uid = os.getuid()

    if not env.get("XDG_RUNTIME_DIR"):
        candidate = f"/run/user/{uid}"
        if os.path.isdir(candidate):
            env["XDG_RUNTIME_DIR"] = candidate

    if not env.get("DBUS_SESSION_BUS_ADDRESS"):
        bus = f"/run/user/{uid}/bus"
        if os.path.exists(bus):
            env["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={bus}"

    if not env.get("WAYLAND_DISPLAY"):
        for name in ("wayland-0", "wayland-1", "wayland-2"):
            if os.path.exists(f"/run/user/{uid}/{name}"):
                env["WAYLAND_DISPLAY"] = name
                break

    if not env.get("DISPLAY"):
        import glob as _glob
        candidates = sorted(_glob.glob("/tmp/.X11-unix/X*"))
        if candidates:
            n = os.path.basename(candidates[0])[1:]
            env["DISPLAY"] = f":{n}"

    if not env.get("XDG_SESSION_TYPE"):
        if env.get("WAYLAND_DISPLAY"):
            env["XDG_SESSION_TYPE"] = "wayland"
        elif env.get("DISPLAY"):
            env["XDG_SESSION_TYPE"] = "x11"

    return env


# ============================================================================
# TOOL 1 — GET SESSION (reuse-or-open)
# ============================================================================

@tool("session", "MEDIUM")
def get_session(
    label: Annotated[
        str,
        Field(
            min_length=0,
            max_length=64,
            description=(
                "Label used only if a new terminal must be opened."
            ),
        ),
    ] = "",
):
    """
    Get the current verifier terminal session.

    If a live terminal session already exists (from a previous
    request), returns it. Otherwise opens a new terminal.

    Prefer this over open_terminal_session().
    """
    for sid, meta in list(_SESSIONS.items()):
        sock_path = meta.get("socket_path", "")
        if sock_path and _is_socket_alive(sock_path):
            _ops_log(
                f"get_session: reusing session_id={sid} "
                f"label={meta.get('label', '')}"
            )
            return {
                "session_id": sid,
                "socket_path": sock_path,
                "label": meta.get("label", ""),
                "action": "reused",
            }
        else:
            _SESSIONS.pop(sid, None)

    _save_sessions()

    _ops_log("get_session: no live session, opening new")

    return open_terminal_session(label=label)


# ============================================================================
# TOOL 2 — OPEN TERMINAL SESSION
# ============================================================================

@tool("session", "MEDIUM")
def open_terminal_session(
    label: Annotated[
        str,
        Field(
            min_length=0,
            max_length=64,
            description=(
                "Optional short label shown in the terminal title."
            ),
        ),
    ] = "",
):
    """
    Force-open a NEW terminal session for the Verifier.

    Prefer get_session() which reuses an existing session.

    Returns a session_id. Pass it to run_in_shell().
    """
    _ensure_socket_dir()

    session_id = str(uuid.uuid4())
    sock_path = _socket_path_for(session_id)

    try:
        os.unlink(sock_path)
    except FileNotFoundError:
        pass

    host_script = _find_host_script()

    term_prefix = _pick_terminal()
    if term_prefix is None:
        raise ToolError(
            "NO_TERMINAL",
            "no supported terminal found",
        )

    inner_argv = [
        sys.executable,
        host_script,
        "--socket",
        sock_path,
        "--label",
        label or "zoya-verifier",
    ]

    import shutil as _shutil
    setsid = _shutil.which("setsid")

    if setsid:
        full_argv = [setsid] + term_prefix + inner_argv
    else:
        full_argv = term_prefix + inner_argv

    env = _ensure_display_env(os.environ.copy())

    if not env.get("DISPLAY") and not env.get("WAYLAND_DISPLAY"):
        raise ToolError(
            "NO_DISPLAY",
            (
                "No graphical display env found. Launch Zoya from "
                "a real terminal (ghostty), not from an IDE."
            ),
        )

    _ops_log(
        "SPAWN: " + " ".join(repr(a) for a in full_argv)
    )
    _ops_log(
        f"SPAWN cwd={os.getcwd()}  sys.executable={sys.executable}"
    )
    for var in (
        "DISPLAY",
        "WAYLAND_DISPLAY",
        "DBUS_SESSION_BUS_ADDRESS",
        "XDG_RUNTIME_DIR",
        "XDG_SESSION_TYPE",
    ):
        _ops_log(f"SPAWN env {var}={env.get(var, '<unset>')}")

    try:
        open(SPAWN_STDERR_LOG, "w").close()
    except Exception:
        pass

    spawn_err = open(SPAWN_STDERR_LOG, "a")

    try:
        proc = subprocess.Popen(
            full_argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=spawn_err,
            start_new_session=True,
            close_fds=True,
            env=env,
        )
    except OSError as exc:
        try:
            spawn_err.close()
        except Exception:
            pass
        raise ToolError(
            "SPAWN_FAILED",
            f"could not spawn terminal: {exc}",
        )

    _ops_log(f"SPAWN pid={proc.pid}")

    ready = _wait_for_socket(sock_path, SOCKET_READY_TIMEOUT)

    try:
        spawn_err.close()
    except Exception:
        pass

    if not ready:
        try:
            with open(SPAWN_STDERR_LOG, "r", encoding="utf-8",
                      errors="replace") as f:
                spawn_stderr = f.read()
        except Exception:
            spawn_stderr = ""

        if spawn_stderr:
            _ops_log("SPAWN stderr:\n" + spawn_stderr[:4000])
        else:
            _ops_log("SPAWN stderr: <empty>")

        try:
            still_running = proc.poll() is None
        except Exception:
            still_running = False

        _ops_log(
            f"SPAWN terminal alive={still_running} "
            f"returncode={proc.returncode}"
        )

        try:
            os.unlink(sock_path)
        except Exception:
            pass

        raise ToolError(
            "SESSION_TIMEOUT",
            "terminal did not open its socket in time",
            retryable=True,
        )

    _SESSIONS[session_id] = {
        "socket_path": sock_path,
        "spawned_at": time.time(),
        "pid": proc.pid,
        "label": label,
    }
    _save_sessions()

    _ops_log(f"SPAWN OK session_id={session_id}")

    return {
        "session_id": session_id,
        "socket_path": sock_path,
        "terminal_pid": proc.pid,
        "label": label or "zoya-verifier",
        "action": "terminal_opened",
    }


# ============================================================================
# TOOL 3 — RUN IN SHELL
# ============================================================================

@tool("session", "HIGH")
def run_in_shell(
    session_id: Annotated[
        str,
        Field(
            min_length=1,
            max_length=128,
            description=(
                "The session_id returned by get_session()."
            ),
        ),
    ],
    command: Annotated[
        str,
        Field(
            min_length=1,
            max_length=8000,
            description=(
                "The shell command to run in the terminal."
            ),
        ),
    ],
    timeout: Annotated[
        int,
        Field(
            ge=1,
            le=MAX_COMMAND_TIMEOUT,
            description=(
                "Maximum seconds to wait for the command to "
                "finish before giving up."
            ),
        ),
    ] = DEFAULT_COMMAND_TIMEOUT,
):
    """
    Run a shell command in an open terminal session.

    If the returned error code is SESSION_NOT_FOUND, the
    terminal was closed. Call get_session() to obtain a
    fresh session, then retry.
    """
    if not command.strip():
        raise ToolError(
            "INVALID_ARGUMENTS",
            "command cannot be empty",
        )

    meta = _SESSIONS.get(session_id)

    if meta is None:
        raise ToolError(
            "SESSION_NOT_FOUND",
            f"unknown session_id: {session_id}. "
            "Call get_session() to open a new one.",
        )

    sock_path = meta["socket_path"]

    if not _is_socket_alive(sock_path):
        _SESSIONS.pop(session_id, None)
        _save_sessions()
        raise ToolError(
            "SESSION_NOT_FOUND",
            "terminal session is no longer running. "
            "Call get_session() to open a new one.",
        )

    payload = {
        "type": "run",
        "command": command,
        "timeout": timeout,
    }

    client_timeout = timeout + 15.0

    try:
        response = _send_request(sock_path, payload, client_timeout)
    except ToolError:
        _SESSIONS.pop(session_id, None)
        _save_sessions()
        raise

    status = response.get("status")

    if status == "ok":
        return {
            "session_id": session_id,
            "command": command,
            "exit_code": response.get("exit_code"),
            "stdout": _truncate(response.get("stdout", "")),
            "stderr": _truncate(response.get("stderr", "")),
            "duration_ms": response.get("duration_ms"),
            "action": "ran",
        }

    if status == "denied":
        return {
            "session_id": session_id,
            "command": command,
            "denied": True,
            "reason": response.get("reason", "denied by user"),
            "action": "denied",
        }

    if status == "timeout":
        return {
            "session_id": session_id,
            "command": command,
            "timeout": True,
            "duration_ms": response.get("duration_ms"),
            "stdout": _truncate(response.get("stdout", "")),
            "stderr": _truncate(response.get("stderr", "")),
            "action": "timeout",
        }

    raise ToolError(
        "BAD_SESSION_RESPONSE",
        f"unexpected status from terminal: {status}",
    )


# ============================================================================
# TOOL 4 — CLOSE SESSION
# ============================================================================

@tool("session", "LOW")
def close_terminal_session(
    session_id: Annotated[
        str,
        Field(min_length=1, max_length=128),
    ],
):
    """Close a terminal session if it is still open."""
    meta = _SESSIONS.pop(session_id, None)
    _save_sessions()

    if meta is None:
        return {
            "session_id": session_id,
            "action": "already_closed",
        }

    sock_path = meta["socket_path"]

    if _is_socket_alive(sock_path):
        try:
            _send_request(
                sock_path,
                {"type": "shutdown"},
                timeout=3.0,
            )
        except Exception:
            pass

    try:
        os.unlink(sock_path)
    except Exception:
        pass

    return {
        "session_id": session_id,
        "action": "closed",
    }


# ============================================================================
# SELF TEST
# ============================================================================

if __name__ == "__main__":
    print(f"{len(TOOLS)} Verifier tools registered")
    for name in sorted(TOOLS):
        meta = TOOLS[name]
        print(
            f"- {name} [group={meta['group']}, "
            f"mutating={meta['mutating']}, risk={meta['risk']}]"
        )