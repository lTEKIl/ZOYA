"""
verifier_agent.py

The Verifier is the ACTION agent. It runs every mutating action
inside a live terminal session that the user can watch.

Tools available:

    get_session()                — reuse-or-open a terminal, get a session_id
    open_terminal_session()      — force-open a fresh terminal (rare)
    run_in_shell(id, cmd)        — run a command in the terminal
    close_terminal_session(id)   — close a session

The terminal (shell_host.py) owns the approval prompt.
"""

import asyncio
import json
import os
import sys
import datetime
import traceback

from groq import Groq
from google.protobuf import json_format

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from a2a.types import (
    SendMessageRequest,
    SendMessageResponse,
)


# ============================================================
# CONFIG
# ============================================================

MODEL = "openai/gpt-oss-120b"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY environment variable is not set."
    )

groq = Groq(api_key=GROQ_API_KEY)

OPS_LOG = os.path.abspath("runs/verifier_ops.log")

SHARED_OPS_LOG = os.path.abspath("runs/infra_ops.log")

MAX_AGENT_ITERATIONS = 12


# ============================================================
# LOGGING
# ============================================================

def ops_log(message):
    try:
        os.makedirs(os.path.dirname(OPS_LOG), exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        with open(OPS_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
            f.flush()
    except Exception:
        pass


def shared_ops_log(message):
    try:
        os.makedirs(
            os.path.dirname(SHARED_OPS_LOG),
            exist_ok=True,
        )
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        with open(SHARED_OPS_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
            f.flush()
    except Exception:
        pass


def log_exception(prefix, exc):
    try:
        traceback_text = "".join(
            traceback.format_exception(
                type(exc),
                exc,
                exc.__traceback__,
            )
        )
    except Exception:
        traceback_text = repr(exc)

    ops_log(f"{prefix}\n{traceback_text}")
    print(
        f"{prefix}\n{traceback_text}",
        file=sys.stderr,
        flush=True,
    )


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are the Verifier / Action Agent in a local laptop automation
system. You are the ONLY agent that can change the state of the
machine.

The user is on Ubuntu Linux, running as a normal desktop user.

============================================================
YOUR ENVIRONMENT
============================================================

You have these tools:

  1. get_session()
     Returns the current verifier terminal session. If a
     terminal is already open from a previous request, the
     SAME session_id is returned and no new window is opened.
     If no terminal exists, a new one is spawned.
     Call this FIRST on every task. Store the returned
     session_id.

  2. run_in_shell(session_id, command, timeout)
     Runs a shell command inside that terminal.
     The user sees it live and can approve or deny it there.
     Returns stdout, stderr, exit_code.

  3. open_terminal_session()
     Force-opens a NEW terminal even if one already exists.
     Only use this when the user explicitly asks for a
     fresh terminal, or when get_session() keeps failing.

  4. close_terminal_session(session_id)
     Closes a session. Rarely needed.

Everything you do is a shell command. There are no structured
tools. You express every action as a real command, the same way
a human would type it into a terminal.

============================================================
THE SHELL IS REAL
============================================================

- It is a real interactive bash session on the user's machine.
- The user can type commands in the same shell at the same time.
- The user sees every command you run.
- For dangerous commands (rm, sudo, killall, dd, mkfs, etc.)
  the terminal will ask the user to approve before running.
- sudo prompts appear in the terminal. The user types the
  password there. You do NOT need to handle passwords.

============================================================
WORKFLOW
============================================================

For every task:

1. Call get_session() FIRST. This returns the terminal
   session. If a terminal is already open from a previous
   request, the same session_id is returned. Otherwise a
   new terminal is opened.

   get_session() returns something like:

       {
         "session_id": "18308d30-7502-4e22-ac77-716c99853e19",
         "socket_path": "/tmp/zoya_verifier/18308d30-....sock",
         "label": "zoya-verifier",
         "action": "reused"   <- or "terminal_opened"
       }

   Remember only the session_id. You do not need to
   remember it across user messages — get_session() will
   return it again next time.

2. Run the commands needed to complete the task, one at a
   time. Wait for each result before deciding the next
   command.

3. If run_in_shell returns an error with code
   SESSION_NOT_FOUND, the terminal was closed. Call
   get_session() to obtain a fresh one, then retry.

4. When done, return a final response that is plain text
   describing:
   - what you ran
   - the outcome (success or failure)
   - any important output
   - any follow-up the user should know about

============================================================
COMMAND GUIDELINES
============================================================

- Prefer simple, focused commands over long one-liners.
- Prefer absolute paths.
- Use the user's home directory as needed; `$HOME` expands
  in the shell, that is fine.
- Do not invent PIDs, paths or usernames. If you need a PID,
  run `pgrep`, `ps aux | grep` or similar to find it first.
- Do not use `su`, `doas` or `pkexec`. Use `sudo` when needed.
- Do not try to pipe commands together with `;` unless it is
  genuinely simpler. Prefer multiple run_in_shell calls.
- Do not run interactive programs (vim, nano, htop, less)
  unless the user explicitly wants them. They will block.

============================================================
SAFETY RULES
============================================================

- The terminal owns approval. If a command is denied there,
  run_in_shell will return {"denied": true}. Do NOT retry the
  same command. Either choose a safer alternative or report
  the denial in your final answer.

- Never claim success for a command that returned a non-zero
  exit code without explaining what happened.

- If a command times out, report that. Do not pretend it
  succeeded.

============================================================
OUTPUT FORMAT
============================================================

Return a plain-text final response, not JSON. Something like:

    Ran: rm /home/rahul-teki/Documents/hello.py
    Result: success (exit code 0)
    The file was deleted.

    Ran: systemctl restart docker
    Result: success (exit code 0)
    Docker restarted cleanly.

If a command failed or was denied, say so clearly.
"""


# ============================================================
# MCP -> GROQ TOOL FORMAT
# ============================================================

def mcp_tool_to_groq(tool):
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.input_schema or {
                "type": "object",
                "properties": {},
            },
        },
    }


# ============================================================
# MCP RESULT EXTRACTION
# ============================================================

def extract_mcp_result(mcp_response):
    results = []

    for content in mcp_response.content:
        if hasattr(content, "text"):
            try:
                results.append(json.loads(content.text))
            except json.JSONDecodeError:
                results.append(content.text)
        else:
            results.append(str(content))

    if len(results) == 1:
        return results[0]

    return results


def safe_json(value):
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )
    except Exception:
        return str(value)


def result_preview(result, limit=2000):
    text = safe_json(result)
    if len(text) > limit:
        return text[:limit] + "... [truncated]"
    return text


# ============================================================
# MCP SERVER
# ============================================================

def _spawn_mcp_server_params():
    return StdioServerParameters(
        command=sys.executable,
        args=["tool_server.py", "verifier"],
        env=dict(os.environ),
    )


# ============================================================
# VERIFIER AGENT INNER LOOP
# ============================================================

async def _run_verifier_agent_inner(request_text):

    ops_log("=" * 70)
    ops_log("▶ VERIFIER AGENT TASK")
    ops_log(f"Request: {request_text}")

    import os as _os
    ops_log(
        "[VERIFIER ENV] "
        f"DISPLAY={_os.environ.get('DISPLAY', '<unset>')} "
        f"WAYLAND_DISPLAY={_os.environ.get('WAYLAND_DISPLAY', '<unset>')} "
        f"DBUS={_os.environ.get('DBUS_SESSION_BUS_ADDRESS', '<unset>')} "
        f"XDG_RUNTIME_DIR={_os.environ.get('XDG_RUNTIME_DIR', '<unset>')}"
    )

    server_params = _spawn_mcp_server_params()

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            ops_log("→ Initializing verifier MCP connection...")
            await session.initialize()

            ops_log("→ Discovering verifier MCP tools...")
            mcp_result = await session.list_tools()
            mcp_tools = list(mcp_result.tools)

            groq_tools = [
                mcp_tool_to_groq(tool)
                for tool in mcp_tools
            ]

            ops_log(
                "✓ Connected to verifier MCP Tool Server "
                f"({len(mcp_tools)} tools available)"
            )

            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": request_text,
                },
            ]

            for iteration in range(1, MAX_AGENT_ITERATIONS + 1):

                ops_log(f"→ Verifier LLM iteration {iteration}")

                response = groq.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=groq_tools,
                    tool_choice="auto",
                    temperature=0.1,
                )

                message = response.choices[0].message
                tool_calls = message.tool_calls or []

                if tool_calls:

                    messages.append(message)

                    for call in tool_calls:

                        tool_name = call.function.name

                        try:
                            arguments = json.loads(
                                call.function.arguments or "{}"
                            )
                        except json.JSONDecodeError as exc:
                            ops_log(
                                f"✗ Invalid JSON arguments for "
                                f"{tool_name}: {exc}"
                            )
                            messages.append({
                                "role": "tool",
                                "tool_call_id": call.id,
                                "content": safe_json({
                                    "ok": False,
                                    "error": {
                                        "code": "INVALID_TOOL_ARGUMENTS",
                                        "message": str(exc),
                                    },
                                }),
                            })
                            continue

                        ops_log(f"→ TOOL: {tool_name}")
                        ops_log(f"  args: {safe_json(arguments)}")

                        try:
                            mcp_response = await session.call_tool(
                                tool_name,
                                arguments,
                            )
                            result = extract_mcp_result(mcp_response)

                        except Exception as exc:
                            log_exception(
                                f"✗ MCP TOOL ERROR: {tool_name}",
                                exc,
                            )
                            result = {
                                "ok": False,
                                "tool": tool_name,
                                "error": {
                                    "code": "MCP_TOOL_ERROR",
                                    "message": str(exc),
                                    "exception_type": type(exc).__name__,
                                },
                            }

                        ops_log("← RESULT: " + result_preview(result))

                        messages.append({
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": safe_json(result),
                        })

                    continue

                final_text = (message.content or "").strip()

                if not final_text:
                    final_text = (
                        "The task completed but no final summary "
                        "was produced."
                    )

                ops_log("✓ VERIFIER AGENT COMPLETE")
                ops_log("  Final: " + final_text[:1500])
                ops_log("=" * 70)

                return final_text

            ops_log("✗ VERIFIER ITERATION LIMIT")
            ops_log("=" * 70)

            return (
                "I reached my reasoning limit before completing "
                "this task. Please try again with a more focused "
                "request, or check the terminal for what ran."
            )


async def run_verifier_agent(request_text):
    """Outer wrapper. Never let an exception escape."""
    try:
        return await _run_verifier_agent_inner(request_text)
    except BaseException as e:

        try:
            import traceback as _tb
            traceback_text = "".join(
                _tb.format_exception(
                    type(e), e, e.__traceback__
                )
            )
        except Exception:
            traceback_text = repr(e)

        header = (
            "✗ VERIFIER OUTER ERROR "
            f"({type(e).__name__}):\n"
        )

        try:
            ops_log(header + traceback_text)
        except Exception:
            pass

        try:
            shared_ops_log(header + traceback_text)
        except Exception:
            pass

        try:
            print(
                header + traceback_text,
                file=sys.stderr,
                flush=True,
            )
        except Exception:
            pass

        return (
            "The verifier could not complete this request "
            "due to an internal error. See "
            "runs/verifier_ops.log for details."
        )


# ============================================================
# A2A HELPERS
# ============================================================

def proto_to_dict(message):
    return json_format.MessageToDict(
        message,
        preserving_proto_field_name=True,
    )


def dict_to_proto(data, message_type):
    message = message_type()
    json_format.ParseDict(data, message)
    return message


def build_a2a_response(request_dict, result_text):
    request = dict_to_proto(
        request_dict,
        SendMessageRequest,
    )

    response = SendMessageResponse()
    response.message.message_id = request.message.message_id
    response.message.role = 2

    part = response.message.parts.add()
    part.text = result_text

    return proto_to_dict(response)


# ============================================================
# A2A REQUEST HANDLER
# ============================================================

async def handle_a2a_request(request_dict):
    request = dict_to_proto(
        request_dict,
        SendMessageRequest,
    )

    parts = [
        part.text
        for part in request.message.parts
        if part.text
    ]

    request_text = "\n".join(parts).strip()

    if not request_text:
        return build_a2a_response(
            request_dict,
            "No verifier request was provided.",
        )

    ops_log("Received A2A verifier request.")

    result_text = await run_verifier_agent(request_text)

    return build_a2a_response(
        request_dict,
        result_text,
    )


# ============================================================
# A2A STDIO SERVER
# ============================================================

async def a2a_server():
    print(
        "[Verifier Agent] A2A stdio server started.",
        file=sys.stderr,
        flush=True,
    )

    ops_log("Verifier Agent process started.")

    for line in sys.stdin:
        line = line.strip()

        if not line:
            continue

        request = {}

        try:
            request = json.loads(line)

            request_id = request.get("id")
            method = request.get("method")
            params = request.get("params", {})

            print(
                f"[Verifier Agent] A2A request: {method}",
                file=sys.stderr,
                flush=True,
            )

            if method == "SendMessage":
                result = await handle_a2a_request(params)

                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": result,
                }

            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unsupported method: {method}",
                    },
                }

        except Exception as exc:
            log_exception("✗ A2A ERROR", exc)

            response = {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": -32000,
                    "message": str(exc),
                    "type": type(exc).__name__,
                },
            }

        print(
            json.dumps(response, ensure_ascii=False),
            flush=True,
        )


# ============================================================
# MAIN
# ============================================================

async def main():
    await a2a_server()


if __name__ == "__main__":
    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        ops_log("Verifier Agent stopped by user.")

    except Exception as exc:
        log_exception(
            "✗ FATAL VERIFIER AGENT ERROR",
            exc,
        )
        raise