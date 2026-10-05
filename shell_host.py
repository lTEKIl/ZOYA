"""
shell_host.py

Persistent terminal host for the Zoya Verifier.

Runs INSIDE a real terminal window. Opens a Unix socket that the
verifier MCP tool connects to. Executes shell commands on behalf
of the verifier. Prompts the user for approval on dangerous
commands. Also lets the user type their own commands in the same
interactive shell.

Usage (spawned by verifier_tools.open_terminal_session):

    python shell_host.py --socket /tmp/zoya_verifier/<id>.sock \
                         --label zoya-verifier

Protocol (newline-delimited JSON over a Unix domain socket):

    Request:
        {"type": "run", "command": "...", "timeout": 120}
        {"type": "shutdown"}

    Response:
        {"status": "ok", "exit_code": 0, "stdout": "...",
         "stderr": "...", "duration_ms": 42}

        {"status": "denied", "reason": "denied by user"}

        {"status": "timeout", "duration_ms": 120000,
         "stdout": "...", "stderr": "..."}
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import select
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid


# ============================================================
# CONFIG
# ============================================================

DANGEROUS_PATTERNS = [
    # (regex, human-readable label)
    (re.compile(r"(^|[|;&]\s*)rm\s"),     "delete file"),
    (re.compile(r"(^|[|;&]\s*)rmdir\s"),  "delete directory"),
    (re.compile(r"(^|[|;&]\s*)shred\s"),  "securely delete"),
    (re.compile(r"(^|[|;&]\s*)dd\s"),     "raw disk operation"),
    (re.compile(r"(^|[|;&]\s*)mkfs"),     "format filesystem"),
    (re.compile(r"(^|[|;&]\s*)killall\s"),"kill process group"),
    (re.compile(r"(^|[|;&]\s*)pkill\s"),  "kill by name"),
    (re.compile(r"(^|[|;&]\s*)reboot\b"), "reboot machine"),
    (re.compile(r"(^|[|;&]\s*)shutdown\b"),"shutdown machine"),
    (re.compile(r"(^|[|;&]\s*)poweroff\b"),"power off machine"),
    (re.compile(r"(^|[|;&]\s*)halt\b"),   "halt machine"),
    (re.compile(r"(^|[|;&]\s*)sudo\s"),   "sudo (root privileges)"),
    (re.compile(r"(^|[|;&]\s*)iptables\b"),"firewall change"),
    (re.compile(r"(^|[|;&]\s*)nft\b"),    "firewall change"),
    (re.compile(r"(^|[|;&]\s*)ufw\b"),    "firewall change"),
]

DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 1800
MAX_OUTPUT = 32000
PROMPT_TIMEOUT = 120


# ============================================================
# LOGGING
# ============================================================

OPS_LOG = os.path.abspath("runs/infra_ops.log")


def ops_log(message):
    try:
        os.makedirs(os.path.dirname(OPS_LOG), exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        with open(OPS_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
            f.flush()
    except Exception:
        pass


# ============================================================
# PRINT LOCK — stops REPL prompt and command output from
# interleaving on the same terminal
# ============================================================

_PRINT_LOCK = threading.RLock()


def _safe_print(*args, **kwargs):
    with _PRINT_LOCK:
        try:
            print(*args, **kwargs)
            sys.stdout.flush()
        except Exception:
            pass


def _prompt(text: str):
    with _PRINT_LOCK:
        try:
            sys.stdout.write(text)
            sys.stdout.flush()
        except Exception:
            pass


# ============================================================
# STDIN COORDINATION
#
# Only one thread may read from stdin at a time. The approval
# prompt takes priority: while it is active, the REPL must not
# steal keystrokes.
# ============================================================

_STDIN_LOCK = threading.Lock()
_PROMPT_ACTIVE = threading.Event()


# ============================================================
# ANSI COLORS
# ============================================================

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"


def _c(text, color):
    return f"{color}{text}{RESET}"


# ============================================================
# DANGEROUS COMMAND DETECTION
# ============================================================

def is_dangerous(command: str) -> tuple[bool, str]:
    """Return (is_dangerous, human_label)."""
    if not command:
        return False, ""

    for pattern, label in DANGEROUS_PATTERNS:
        if pattern.search(command):
            return True, label

    if re.search(r">\s*/dev/(sd|nvme|hd|mmcblk)", command):
        return True, "redirection to block device"

    return False, ""


# ============================================================
# PERSISTENT BASH
# ============================================================

class PersistentBash:
    """
    A single long-lived bash process we talk to.

    Started with --norc --noprofile so user prompt configs and
    aliases don't inject noise into stdout. TERM=dumb disables
    any remaining color/prompt escapes. Job control is disabled
    with `set +m`. Echo is disabled with `stty -echo` so the
    command we write to stdin doesn't come back as output.
    """

    def __init__(self):
        self.lock = threading.RLock()
        self.proc = None
        self._spawn()

    def _spawn(self):
        env = os.environ.copy()
        env["PS1"] = ""
        env["PROMPT_COMMAND"] = ""
        env["TERM"] = "dumb"

        self.proc = subprocess.Popen(
            ["/bin/bash", "--norc", "--noprofile"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        try:
            self.proc.stdin.write("set +m\n")
            self.proc.stdin.write("stty -echo 2>/dev/null || true\n")
            self.proc.stdin.flush()
        except Exception:
            pass

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def run(self, command: str, timeout: int) -> dict:
        with self.lock:

            if not self.is_alive():
                self._spawn()

            sentinel = f"__ZOYA_DONE_{uuid.uuid4().hex}__"

            wrapped = (
                f"{command}\n"
                f"__ZOYA_EXIT=$?\n"
                f"printf '%s %s\\n' '{sentinel}' \"$__ZOYA_EXIT\"\n"
            )

            started = time.time()

            try:
                self.proc.stdin.write(wrapped)
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError):
                self._spawn()
                return {
                    "exit_code": -1,
                    "output": "",
                    "timed_out": False,
                    "duration_ms": int((time.time() - started) * 1000),
                    "error": "bash process died and was respawned",
                }

            output_lines = []
            exit_code = None
            timed_out = False

            import queue
            line_q: "queue.Queue[str]" = queue.Queue()

            def _reader():
                try:
                    for line in self.proc.stdout:
                        line_q.put(line)
                        if sentinel in line:
                            return
                except Exception:
                    return

            reader = threading.Thread(target=_reader, daemon=True)
            reader.start()

            deadline = time.time() + timeout

            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    timed_out = True
                    break

                try:
                    line = line_q.get(timeout=min(0.5, remaining))
                except queue.Empty:
                    continue

                if sentinel in line:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            exit_code = int(parts[1])
                        except ValueError:
                            exit_code = None
                    break

                output_lines.append(line)

            if timed_out:
                try:
                    self.proc.send_signal(signal.SIGINT)
                except Exception:
                    pass

            output = "".join(output_lines)

            if len(output) > MAX_OUTPUT:
                output = output[:MAX_OUTPUT] + "\n...[truncated]..."

            return {
                "exit_code": exit_code,
                "output": output,
                "timed_out": timed_out,
                "duration_ms": int((time.time() - started) * 1000),
            }

    def shutdown(self):
        try:
            if self.is_alive():
                self.proc.stdin.write("exit\n")
                self.proc.stdin.flush()
        except Exception:
            pass

        try:
            self.proc.wait(timeout=2)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


# ============================================================
# APPROVAL PROMPT
# ============================================================

def prompt_approval(command: str, reason: str):
    """
    Returns True (once), "session", or False.

    Owns stdin while active so the REPL can't steal the answer.
    """
    _PROMPT_ACTIVE.set()

    try:
        with _STDIN_LOCK:
            with _PRINT_LOCK:
                print()
                print(_c("┌─────────────────────────────────────────────────────────────┐", YELLOW))
                print(_c("│  ZOYA VERIFIER — ACTION REQUEST                             │", YELLOW))
                print(_c("└─────────────────────────────────────────────────────────────┘", YELLOW))
                print()
                print(_c("  Command:", BOLD))
                print()
                for line in command.splitlines() or [command]:
                    print(f"    {line}")
                print()
                if reason:
                    print(_c(f"  Reason: {reason}", DIM))
                    print()
                print(_c("  Allow? [y] yes  [n] no  [a] yes and don't ask again this session", BOLD))
                print()

            try:
                _prompt("  > ")

                rlist, _, _ = select.select([sys.stdin], [], [], PROMPT_TIMEOUT)

                if not rlist:
                    _safe_print(_c("  (no answer — denied)", RED))
                    return False

                answer = sys.stdin.readline().strip().lower()

            except (KeyboardInterrupt, EOFError):
                _safe_print()
                _safe_print(_c("  (interrupted — denied)", RED))
                return False

            if answer in ("y", "yes"):
                _safe_print(_c("  ✓ approved", GREEN))
                return True

            if answer in ("a", "always"):
                _safe_print(_c("  ✓ approved (session)", GREEN))
                return "session"

            _safe_print(_c("  ✗ denied", RED))
            return False

    finally:
        _PROMPT_ACTIVE.clear()


# ============================================================
# SESSION STATE
# ============================================================

class Session:
    def __init__(self, label: str):
        self.label = label
        self.started_at = time.time()
        self.session_approvals: set[str] = set()
        self.shutdown = threading.Event()
        self.lock = threading.Lock()
        self.bash = PersistentBash()

    def is_session_approved(self, command: str) -> bool:
        with self.lock:
            return command in self.session_approvals

    def add_session_approval(self, command: str):
        with self.lock:
            self.session_approvals.add(command)

    def close(self):
        try:
            self.bash.shutdown()
        except Exception:
            pass


# ============================================================
# OUTPUT ROUTING
# ============================================================

def _banner(session: Session, sock_path: str):
    _safe_print()
    _safe_print(_c("═" * 64, CYAN))
    _safe_print(_c(f"  ZOYA VERIFIER TERMINAL  —  {session.label}", BOLD))
    _safe_print(_c("═" * 64, CYAN))
    _safe_print()
    _safe_print(f"  Socket: {_c(sock_path, DIM)}")
    _safe_print(f"  Started: {datetime.datetime.now().strftime('%H:%M:%S')}")
    _safe_print()
    _safe_print("  This terminal is controlled by the Verifier.")
    _safe_print("  Every command the verifier runs appears here.")
    _safe_print()
    _safe_print("  You can also type your own commands below.")
    _safe_print("  Type 'exit' to close the terminal.")
    _safe_print()
    _safe_print(_c("─" * 64, DIM))
    _safe_print()


def _log_line(prefix: str, color: str, text: str):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    first = True
    for line in text.splitlines() or [""]:
        if first:
            _safe_print(f"{_c(f'[{timestamp}]', DIM)} {_c(prefix, color)} {line}")
            first = False
        else:
            _safe_print(f"            {line}")


def _log_run(command: str):
    _safe_print()
    _safe_print(_c("─" * 64, DIM))
    _log_line("ZOYA ▶", BLUE, command)
    _safe_print()


def _log_result(response: dict):
    status = response.get("status")
    duration = response.get("duration_ms", 0)

    if status == "ok":
        code = response.get("exit_code")
        if code == 0:
            _log_line("✓ done", GREEN, f"exit 0  ({duration}ms)")
        else:
            _log_line("✗ failed", RED, f"exit {code}  ({duration}ms)")
    elif status == "timeout":
        _log_line("⏱ timeout", YELLOW, f"({duration}ms)")
    elif status == "denied":
        _log_line("✗ denied", RED, response.get("reason", ""))

    stdout = response.get("stdout") or ""
    stderr = response.get("stderr") or ""

    if stdout:
        _safe_print()
        for line in stdout.splitlines():
            _safe_print(f"  {line}")

    if stderr:
        _safe_print()
        for line in stderr.splitlines():
            _safe_print(f"  {_c(line, RED)}")

    _safe_print()


def _log_user_command(command: str):
    _safe_print()
    _safe_print(_c("─" * 64, DIM))
    _log_line("you ▶", MAGENTA, command)
    _safe_print()


# ============================================================
# SOCKET SERVER
# ============================================================

def _handle_client(session: Session, conn: socket.socket):
    try:
        conn.settimeout(MAX_TIMEOUT + 30)
        buf = b""
        while b"\n" not in buf:
            chunk = conn.recv(65536)
            if not chunk:
                return
            buf += chunk

        line = buf.split(b"\n", 1)[0].decode("utf-8", errors="replace")

        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            response = {"status": "bad_request", "error": str(exc)}
            conn.sendall((json.dumps(response) + "\n").encode())
            return

        req_type = request.get("type")

        if req_type == "shutdown":
            session.shutdown.set()
            conn.sendall(b'{"status":"ok"}\n')
            return

        if req_type != "run":
            response = {
                "status": "bad_request",
                "error": f"unknown type: {req_type}",
            }
            conn.sendall((json.dumps(response) + "\n").encode())
            return

        command = str(request.get("command", ""))
        timeout = int(request.get("timeout", DEFAULT_TIMEOUT))
        timeout = max(1, min(timeout, MAX_TIMEOUT))

        dangerous, reason = is_dangerous(command)

        if dangerous and not session.is_session_approved(command):
            _log_run(command)
            decision = prompt_approval(command, reason)
            if decision == "session":
                session.add_session_approval(command)
                decision = True
            if not decision:
                response = {
                    "status": "denied",
                    "reason": "denied by user",
                }
                _log_result(response)
                conn.sendall((json.dumps(response) + "\n").encode())
                return
        else:
            _log_run(command)

        result = session.bash.run(command, timeout)

        if result["timed_out"]:
            response = {
                "status": "timeout",
                "duration_ms": result["duration_ms"],
                "stdout": result["output"],
                "stderr": "",
            }
        else:
            response = {
                "status": "ok",
                "exit_code": result["exit_code"],
                "stdout": result["output"],
                "stderr": result.get("error", ""),
                "duration_ms": result["duration_ms"],
            }

        _log_result(response)

        conn.sendall((json.dumps(response) + "\n").encode())

    except Exception as exc:
        try:
            err = {"status": "error", "error": str(exc)}
            conn.sendall((json.dumps(err) + "\n").encode())
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def run_socket_server(session: Session, sock_path: str):
    try:
        os.unlink(sock_path)
    except FileNotFoundError:
        pass

    os.makedirs(os.path.dirname(sock_path), exist_ok=True)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(sock_path)
    server.listen(8)
    server.settimeout(1.0)

    os.chmod(sock_path, 0o600)

    ops_log(f"shell_host listening on {sock_path}")

    try:
        while not session.shutdown.is_set():
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            t = threading.Thread(
                target=_handle_client,
                args=(session, conn),
                daemon=True,
            )
            t.start()
    finally:
        try:
            server.close()
        except Exception:
            pass
        try:
            os.unlink(sock_path)
        except Exception:
            pass


# ============================================================
# INTERACTIVE REPL
# ============================================================

def run_interactive_repl(session: Session):
    prompt_shown = False

    while not session.shutdown.is_set():

        if _PROMPT_ACTIVE.is_set():
            if prompt_shown:
                _safe_print()
                prompt_shown = False
            time.sleep(0.05)
            continue

        if not prompt_shown:
            _prompt(_c("shell> ", MAGENTA))
            prompt_shown = True

        try:
            rlist, _, _ = select.select([sys.stdin], [], [], 1.0)
        except (OSError, ValueError):
            break

        if not rlist:
            continue

        if _PROMPT_ACTIVE.is_set():
            # Prompt started. Blank line so the prompt area is clean.
            _safe_print()
            prompt_shown = False
            continue

        try:
            line = sys.stdin.readline()
        except (KeyboardInterrupt, EOFError):
            _safe_print()
            break

        if not line:
            break

        prompt_shown = False

        command = line.strip()
        if not command:
            continue

        if command in ("exit", "quit"):
            session.shutdown.set()
            break

        _log_user_command(command)

        result = session.bash.run(command, DEFAULT_TIMEOUT)

        if result["timed_out"]:
            response = {
                "status": "timeout",
                "duration_ms": result["duration_ms"],
                "stdout": result["output"],
                "stderr": "",
            }
        else:
            response = {
                "status": "ok",
                "exit_code": result["exit_code"],
                "stdout": result["output"],
                "stderr": result.get("error", ""),
                "duration_ms": result["duration_ms"],
            }

        _log_result(response)

# ============================================================
# SHUTDOWN HANDLING
# ============================================================

def install_signal_handlers(session: Session):
    def handler(signum, frame):
        session.shutdown.set()

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--label", default="zoya-verifier")
    args = parser.parse_args()

    session = Session(label=args.label)
    install_signal_handlers(session)

    _banner(session, args.socket)

    ops_log(
        f"shell_host started, socket={args.socket} label={args.label}"
    )

    server_thread = threading.Thread(
        target=run_socket_server,
        args=(session, args.socket),
        daemon=True,
    )
    server_thread.start()

    try:
        run_interactive_repl(session)
    finally:
        session.shutdown.set()
        _safe_print()
        _safe_print(_c("Session closing...", DIM))
        ops_log("shell_host shutting down")

        session.close()
        server_thread.join(timeout=2.0)

        _safe_print(_c("Goodbye.", DIM))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        sys.exit(1)