import asyncio
import json
import os
import sys
import datetime
import traceback
import re

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

groq = Groq(
    api_key=GROQ_API_KEY
)

OPS_LOG = os.path.abspath(
    "runs/infra_ops.log"
)

MAX_AGENT_ITERATIONS = 12

# Maximum number of times we allow the LLM to retry after
# Groq rejects a generated tool call because of schema rules.
MAX_TOOL_VALIDATION_RETRIES = 3
MAX_MALFORMED_TOOL_RETRIES = 3


# ============================================================
# OPS LOGGING
# ============================================================

def ops_log(message):
    """
    Write Infra Agent activity to the shared Ops log.

    stdout is reserved for A2A protocol messages.
    """

    try:
        os.makedirs(
            os.path.dirname(OPS_LOG),
            exist_ok=True,
        )

        timestamp = datetime.datetime.now().strftime(
            "%H:%M:%S"
        )

        with open(
            OPS_LOG,
            "a",
            encoding="utf-8",
        ) as f:

            f.write(
                f"[{timestamp}] {message}\n"
            )

            f.flush()

    except Exception:
        # Logging must never break the agent.
        pass


def log_exception(prefix, exc):
    """
    Log the complete traceback.

    Useful for:
    - ExceptionGroup
    - TaskGroup
    - MCP failures
    - Groq failures
    - A2A failures
    """

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

    ops_log(
        f"{prefix}\n{traceback_text}"
    )

    print(
        f"{prefix}\n{traceback_text}",
        file=sys.stderr,
        flush=True,
    )


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are the Infra/Ops Agent in a production incident-response
system.

Your job is to INVESTIGATE the user's actual local machine using
MCP tools and report factual findings to the Planner.

============================================================
CORE ROLE
============================================================

You are a READ-ONLY investigation agent.

All MCP tools available to you are read-only.

You DO NOT modify the machine.

You DO NOT restart processes.

You DO NOT stop processes.

You DO NOT kill processes.

You DO NOT renice processes.

You DO NOT clear caches.

You DO NOT change volume.

You DO NOT restart services.

Do not invent or assume the existence of action tools.

Your responsibility is:

    OBSERVE
    INVESTIGATE
    CORRELATE
    REPORT

============================================================
ABSOLUTE GROUNDING RULES
============================================================

The MCP tools provide real information from the user's machine.

You MUST ground your answer strictly in actual MCP results.

NEVER invent system information.

NEVER assume the operating system.

NEVER assume the machine is:

- Windows
- Linux
- Ubuntu
- Debian
- Fedora
- macOS
- or any other operating system

If the operating system was not explicitly returned by a tool,
do not state an operating system as fact.

============================================================
PROCESS RULES
============================================================

Preserve process names exactly as returned by MCP.

If MCP returns:

{
    "pid": 1234,
    "name": "code",
    "rss_mb": 512.4
}

report:

code

Do NOT transform it into:

code.exe

Do NOT invent:

explorer.exe
svchost.exe
RuntimeBroker.exe

or any other process.

Never fabricate:

- PIDs
- process names
- CPU percentages
- memory values
- disk values
- network values
- services
- applications
- hardware
- operating systems
- timestamps

============================================================
CURRENT MACHINE INVESTIGATION
============================================================

When the user asks about the current machine:

1. Use MCP tools.
2. Investigate before answering.
3. Use only tools relevant to the question.
4. Do not blindly call every tool.
5. Use the actual returned evidence.
6. If information is unavailable, say it is unavailable.
7. Never fill missing information with assumptions.

============================================================
APPLICATION INVESTIGATION
============================================================

For questions such as:

- Is application X running?
- Is process X running?
- What is application X doing?
- What process belongs to application X?

Use appropriate process/application tools.

For example:

find_process
get_process_info
get_process_tree
get_process_usage
get_open_files
get_network_connections

depending on the question.

Preserve the exact application/process name returned by MCP.

============================================================
TIME / HISTORY RULE
============================================================

Be extremely careful with historical questions.

Examples:

- When was an application opened?
- When was an application closed?
- When did a process start?
- What happened yesterday?
- When did an application last run?
- What happened before the current process?
- What was the previous crash?

A current process lookup does NOT prove when an application
was opened.

A process being absent does NOT prove exactly when it was closed.

A service log does NOT automatically prove an application's
launch time.

get_logs is for recent journal/service log information. Do not
reinterpret arbitrary log timestamps as application launch
timestamps unless the actual returned log explicitly provides
evidence of that event.

Only state a timestamp if an MCP tool actually provides evidence
for that timestamp.

NEVER invent a timestamp.

If the available tools cannot determine the requested historical
event, explicitly say that the historical information is
unavailable from the available evidence.

============================================================
GET_LOGS LIMIT
============================================================

The get_logs tool has a strict:

minutes <= 240

constraint.

NEVER request:

minutes = 1440
minutes = 720
minutes = 480

or any other value greater than 240.

If you need information covering a longer period, do not violate
the schema.

Use the maximum allowed value of 240 when appropriate, or use
another relevant tool.

============================================================
TOOL SCHEMA RULES
============================================================

You MUST obey the exact JSON schema supplied for every MCP tool.

CRITICAL ZERO-ARGUMENT TOOL RULE:
If a tool has no input parameters, ALWAYS generate:
    arguments: {}
Never generate {"": ""}, {""}, {"": null}, or any other malformed
empty argument representation.

If a parameter has:

- maximum
- minimum
- enum
- required fields
- type restrictions
- other constraints

follow those constraints exactly.

Never generate arguments outside the supplied schema.

============================================================
TOOL FAILURE RULE
============================================================

If a tool returns an error:

1. Understand the error.
2. Do not fabricate a successful result.
3. Try another appropriate read-only tool if useful.
4. Otherwise explain the limitation.

For NOT_FOUND results, do not convert the result into an invented
successful finding.

============================================================
EVIDENCE RULE
============================================================

Every important factual claim about the machine must be supported
by tool output.

Example:

Tool says:

{
    "ok": false,
    "error": {
        "code": "NOT_FOUND"
    }
}

You may conclude:

"The requested process was not found running."

You may NOT conclude:

"The application was uninstalled."

unless another tool provides evidence for that.

============================================================
FINAL RESPONSE
============================================================

Always provide a final answer.

Base the answer only on evidence obtained from MCP tools.

Clearly distinguish:

- observed facts
- conclusions directly supported by evidence
- unavailable information

Never present assumptions as machine facts.

You are investigating the user's real machine,
not a hypothetical machine.
"""


# ============================================================
# MCP TOOL -> GROQ TOOL
# ============================================================

def mcp_tool_to_groq(tool):
    """
    Convert an MCP tool definition into Groq's function-tool
    format.

    The original MCP JSON schema is preserved.
    """

    schema = tool.input_schema

    try:

        schema = json.loads(
            json.dumps(schema)
        )

    except Exception:

        schema = tool.input_schema

    description = tool.description or ""

    # Add explicit safety/context to particularly important tools.
    if tool.name == "get_logs":

        description += (
            "\nIMPORTANT: `minutes` MUST be between 1 and 240. "
            "NEVER use a value greater than 240."
        )

    if tool.name == "find_process":

        description += (
            "\nIMPORTANT: Returns currently running matching "
            "processes. It does NOT provide historical launch time."
        )

    if tool.name == "get_process_info":

        description += (
            "\nIMPORTANT: Use returned process metadata as-is. "
            "Do not invent missing timestamps."
        )

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": description,
            "parameters": schema,
        },
    }


# ============================================================
# TOOL DESCRIPTION / CONSTRAINTS
# ============================================================

def build_tool_constraints_text(mcp_tools):

    lines = []

    for tool in mcp_tools:

        schema = tool.input_schema or {}

        properties = schema.get(
            "properties",
            {},
        )

        if not properties:
            continue

        constraints = []

        for name, prop in properties.items():

            if "maximum" in prop:

                constraints.append(
                    f"{name} maximum={prop['maximum']}"
                )

            if "minimum" in prop:

                constraints.append(
                    f"{name} minimum={prop['minimum']}"
                )

            if "enum" in prop:

                constraints.append(
                    f"{name} allowed={prop['enum']}"
                )

            if "type" in prop:

                constraints.append(
                    f"{name} type={prop['type']}"
                )

        if constraints:

            lines.append(
                f"{tool.name}: "
                + ", ".join(constraints)
            )

    if not lines:

        return ""

    return (
        "\n\n"
        "IMPORTANT MCP TOOL CONSTRAINTS:\n"
        + "\n".join(
            f"- {line}"
            for line in lines
        )
    )


# ============================================================
# GROQ CALL
# ============================================================

def ask_groq(
    messages,
    tools,
    disabled_tools=None,
):

    disabled_tools = disabled_tools or set()

    usable_tools = [
        tool
        for tool in tools
        if tool["function"]["name"]
        not in disabled_tools
    ]

    return groq.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=usable_tools,
        tool_choice="auto",
        temperature=0.1,
    ).choices[0].message


# ============================================================
# EXTRACT TOOL VALIDATION FAILURE
# ============================================================

def extract_tool_validation_error(exc):

    text = str(exc)

    tool_name = None
    parameter = None
    value = None
    maximum = None
    minimum = None

    # --------------------------------------------------------
    # Tool name
    # --------------------------------------------------------

    match = re.search(
        r"parameters for tool\s+([A-Za-z0-9_\-]+)",
        text,
        re.IGNORECASE,
    )

    if match:

        tool_name = match.group(1)

    # --------------------------------------------------------
    # Maximum
    # --------------------------------------------------------

    match = re.search(
        r"`/([^`]+)`:\s*maximum:\s*got\s*([0-9,.-]+),"
        r"\s*want\s*([0-9,.-]+)",
        text,
        re.IGNORECASE,
    )

    if match:

        parameter = match.group(1)

        try:

            value = float(
                match.group(2).replace(",", "")
            )

        except Exception:
            pass

        try:

            maximum = float(
                match.group(3).replace(",", "")
            )

        except Exception:
            pass

    # --------------------------------------------------------
    # Minimum
    # --------------------------------------------------------

    match = re.search(
        r"`/([^`]+)`:\s*minimum:\s*got\s*([0-9,.-]+),"
        r"\s*want\s*([0-9,.-]+)",
        text,
        re.IGNORECASE,
    )

    if match:

        parameter = match.group(1)

        try:

            value = float(
                match.group(2).replace(",", "")
            )

        except Exception:
            pass

        try:

            minimum = float(
                match.group(3).replace(",", "")
            )

        except Exception:
            pass

    return {
        "tool": tool_name,
        "parameter": parameter,
        "value": value,
        "maximum": maximum,
        "minimum": minimum,
        "raw": text,
    }


# ============================================================
# MALFORMED GROQ TOOL-CALL RECOVERY
# ============================================================

def extract_malformed_tool_error(exc):
    text = str(exc)

    if "Failed to parse tool call arguments as JSON" not in text:
        return {"tool": None, "raw": text}

    match = re.search(
        r'"name"\s*:\s*"([A-Za-z0-9_-]+)"',
        text,
    )

    return {
        "tool": match.group(1) if match else None,
        "raw": text,
    }


def build_malformed_tool_recovery_message(error_info):
    tool = error_info.get("tool")

    if tool:
        return (
            "TOOL CALL JSON CORRECTION.\n\n"
            f"The previous `{tool}` call contained malformed JSON "
            "arguments and Groq rejected it before execution.\n\n"
            "If this tool has no parameters, the arguments MUST be "
            "exactly {}.\n"
            'Never generate {"": ""}, {""}, {"": null}, or similar.\n\n'
            f"Generate a new valid `{tool}` call using the exact MCP schema."
        )

    return (
        "TOOL CALL JSON CORRECTION.\n\n"
        "The previous tool call contained malformed JSON arguments.\n"
        "Review the exact MCP schema and generate valid JSON.\n"
        "For a zero-argument tool, use exactly {}."
    )


# ============================================================
# BUILD TOOL VALIDATION RECOVERY MESSAGE
# ============================================================

def build_validation_recovery_message(error_info):

    tool = error_info.get(
        "tool"
    )

    parameter = error_info.get(
        "parameter"
    )

    value = error_info.get(
        "value"
    )

    maximum = error_info.get(
        "maximum"
    )

    minimum = error_info.get(
        "minimum"
    )

    if (
        tool
        and parameter
        and maximum is not None
    ):

        return (
            "TOOL VALIDATION CORRECTION.\n\n"
            f"The previous attempted `{tool}` call was invalid.\n"
            f"`{parameter}` was {value}.\n"
            f"The maximum allowed value is {maximum}.\n\n"
            f"Do NOT use {value} again.\n"
            f"Use `{parameter}` <= {maximum}.\n\n"
            "Generate a new valid tool call."
        )

    if (
        tool
        and parameter
        and minimum is not None
    ):

        return (
            "TOOL VALIDATION CORRECTION.\n\n"
            f"The previous attempted `{tool}` call was invalid.\n"
            f"`{parameter}` was {value}.\n"
            f"The minimum allowed value is {minimum}.\n\n"
            f"Do NOT use {value} again.\n"
            f"Use `{parameter}` >= {minimum}.\n\n"
            "Generate a new valid tool call."
        )

    return (
        "TOOL VALIDATION CORRECTION.\n\n"
        "The previous generated tool call violated the supplied "
        "MCP JSON schema.\n\n"
        "Review the exact schema and generate a valid tool call. "
        "Respect all maximum, minimum, enum and type constraints."
    )


# ============================================================
# MCP RESULT EXTRACTION
# ============================================================

def extract_mcp_result(
    mcp_response
):

    results = []

    for content in mcp_response.content:

        if hasattr(
            content,
            "text",
        ):

            try:

                results.append(
                    json.loads(
                        content.text
                    )
                )

            except json.JSONDecodeError:

                results.append(
                    content.text
                )

        else:

            results.append(
                str(content)
            )

    if len(results) == 1:

        return results[0]

    return results


# ============================================================
# SAFE JSON
# ============================================================

def safe_json(
    value
):

    try:

        return json.dumps(
            value,
            default=str,
            ensure_ascii=False,
        )

    except Exception:

        return str(value)


# ============================================================
# RESULT PREVIEW
# ============================================================

def result_preview(
    result,
    limit=4000,
):

    text = safe_json(
        result
    )

    if len(text) > limit:

        return (
            text[:limit]
            + "... [truncated]"
        )

    return text


# ============================================================
# RUN INFRA AGENT
# ============================================================

async def run_infra_agent(
    incident
):

    ops_log("=" * 70)

    ops_log(
        "▶ INFRA AGENT TASK"
    )

    ops_log(
        f"Request: {incident}"
    )

    # --------------------------------------------------------
    # MCP SERVER
    # --------------------------------------------------------

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[
            "tool_server.py", "infra"
        ],
    )

    try:

        async with stdio_client(
            server_params
        ) as (
            read,
            write,
        ):

            async with ClientSession(
                read,
                write,
            ) as session:

                # ------------------------------------------------
                # INITIALIZE MCP
                # ------------------------------------------------

                ops_log(
                    "→ Initializing MCP connection..."
                )

                await session.initialize()

                # ------------------------------------------------
                # DISCOVER TOOLS
                # ------------------------------------------------

                ops_log(
                    "→ Discovering MCP tools..."
                )

                mcp_result = (
                    await session.list_tools()
                )

                mcp_tools = list(
                    mcp_result.tools
                )

                tools = [
                    mcp_tool_to_groq(
                        tool
                    )
                    for tool in mcp_tools
                ]

                ops_log(
                    "✓ Connected to MCP Tool Server "
                    f"({len(tools)} tools available)"
                )

                tool_names = [
                    tool.name
                    for tool in mcp_tools
                ]

                ops_log(
                    "  Available tools: "
                    + ", ".join(
                        tool_names
                    )
                )

                # ------------------------------------------------
                # TOOL CONSTRAINTS
                # ------------------------------------------------

                constraints_text = (
                    build_tool_constraints_text(
                        mcp_tools
                    )
                )

                if constraints_text:

                    ops_log(
                        "  MCP schema constraints loaded."
                    )

                # ------------------------------------------------
                # INITIAL MESSAGES
                # ------------------------------------------------

                messages = [
                    {
                        "role":
                            "system",

                        "content":
                            SYSTEM_PROMPT
                            + constraints_text,
                    },
                    {
                        "role":
                            "user",

                        "content":
                            incident,
                    },
                ]

                # ------------------------------------------------
                # LOOP
                # ------------------------------------------------

                iteration = 0

                disabled_tools = set()

                while True:

                    iteration += 1

                    ops_log(
                        f"→ LLM reasoning iteration "
                        f"{iteration}"
                    )

                    if iteration > MAX_AGENT_ITERATIONS:

                        ops_log(
                            "✗ Maximum agent iterations reached."
                        )

                        return (
                            "I could not complete the investigation "
                            "within the allowed number of steps."
                        )

                    # =================================================
                    # GROQ REQUEST
                    # =================================================

                    validation_retry = 0

                    while True:

                        try:

                            message = ask_groq(
                                messages,
                                tools,
                                disabled_tools,
                            )

                            validation_retry = 0

                            break

                        except Exception as e:

                            # Groq can reject malformed tool-call JSON
                            # before MCP ever receives the call.
                            malformed_info = (
                                extract_malformed_tool_error(e)
                            )

                            if malformed_info["tool"] is not None:

                                validation_retry += 1
                                bad_tool = malformed_info["tool"]

                                ops_log(
                                    "⚠ GROQ MALFORMED TOOL ARGUMENTS"
                                )
                                ops_log(
                                    f"  Tool: {bad_tool}"
                                )

                                if validation_retry <= MAX_MALFORMED_TOOL_RETRIES:

                                    messages.append(
                                        {
                                            "role": "system",
                                            "content":
                                                build_malformed_tool_recovery_message(
                                                    malformed_info
                                                ),
                                        }
                                    )

                                    ops_log(
                                        "→ Asking LLM to regenerate "
                                        "the malformed tool call."
                                    )

                                    continue

                                disabled_tools.add(bad_tool)

                                ops_log(
                                    f"⚠ Disabling `{bad_tool}` after "
                                    "repeated malformed arguments."
                                )

                                messages.append(
                                    {
                                        "role": "system",
                                        "content": (
                                            f"The tool `{bad_tool}` has been "
                                            "disabled because the model "
                                            "repeatedly generated malformed "
                                            "JSON arguments. Do not call it "
                                            "again. Use another read-only "
                                            "tool if possible."
                                        ),
                                    }
                                )

                                validation_retry = 0
                                continue

                            error_info = (
                                extract_tool_validation_error(
                                    e
                                )
                            )

                            is_validation_error = (
                                error_info["tool"]
                                is not None
                            )

                            # ------------------------------------------------
                            # Normal Groq error
                            # ------------------------------------------------

                            if not is_validation_error:

                                log_exception(
                                    "✗ GROQ ERROR",
                                    e,
                                )

                                # Do NOT let a Groq failure become
                                # an unexplained MCP TaskGroup failure.
                                return (
                                    "The Infra investigation could not "
                                    "continue because the reasoning "
                                    "model returned an error."
                                )

                            # ------------------------------------------------
                            # Tool schema validation error
                            # ------------------------------------------------

                            validation_retry += 1

                            ops_log(
                                "⚠ GROQ TOOL VALIDATION ERROR"
                            )

                            ops_log(
                                "  Tool: "
                                + str(
                                    error_info["tool"]
                                )
                            )

                            ops_log(
                                "  Parameter: "
                                + str(
                                    error_info["parameter"]
                                )
                            )

                            ops_log(
                                "  Attempted value: "
                                + str(
                                    error_info["value"]
                                )
                            )

                            ops_log(
                                "  Maximum: "
                                + str(
                                    error_info["maximum"]
                                )
                            )

                            ops_log(
                                "  Minimum: "
                                + str(
                                    error_info["minimum"]
                                )
                            )

                            # ------------------------------------------------
                            # Retry with correction
                            # ------------------------------------------------

                            if validation_retry <= (
                                MAX_TOOL_VALIDATION_RETRIES
                            ):

                                recovery_message = (
                                    build_validation_recovery_message(
                                        error_info
                                    )
                                )

                                ops_log(
                                    "→ Sending schema correction "
                                    "back to LLM."
                                )

                                messages.append(
                                    {
                                        "role":
                                            "system",

                                        "content":
                                            recovery_message,
                                    }
                                )

                                continue

                            # ------------------------------------------------
                            # Repeated invalid generation
                            # ------------------------------------------------

                            bad_tool = (
                                error_info["tool"]
                            )

                            disabled_tools.add(
                                bad_tool
                            )

                            ops_log(
                                f"⚠ Disabling tool `{bad_tool}` "
                                "for the remainder of this "
                                "investigation."
                            )

                            messages.append(
                                {
                                    "role":
                                        "system",

                                    "content":
                                        (
                                            f"The tool `{bad_tool}` "
                                            "has been disabled because "
                                            "valid arguments could not "
                                            "be generated.\n"
                                            "Do not attempt to call it "
                                            "again.\n"
                                            "Use another available "
                                            "read-only tool or report "
                                            "that the requested "
                                            "information is unavailable."
                                        ),
                                }
                            )

                            validation_retry = 0

                            # Retry Groq without that tool.
                            continue

                    # =================================================
                    # TOOL CALLS
                    # =================================================

                    if message.tool_calls:

                        ops_log(
                            f"→ LLM requested "
                            f"{len(message.tool_calls)} tool call(s)"
                        )

                        assistant_tool_calls = []

                        for call in message.tool_calls:

                            assistant_tool_calls.append(
                                {
                                    "id":
                                        call.id,

                                    "type":
                                        "function",

                                    "function":
                                        {
                                            "name":
                                                call.function.name,

                                            "arguments":
                                                (
                                                    call.function.arguments
                                                    or "{}"
                                                ),
                                        },
                                }
                            )

                        messages.append(
                            {
                                "role":
                                    "assistant",

                                "content":
                                    (
                                        message.content
                                        or ""
                                    ),

                                "tool_calls":
                                    assistant_tool_calls,
                            }
                        )

                        # ------------------------------------------------
                        # EXECUTE MCP TOOLS
                        # ------------------------------------------------

                        for call in message.tool_calls:

                            tool_name = (
                                call.function.name
                            )

                            raw_arguments = (
                                call.function.arguments
                                or "{}"
                            )

                            ops_log(
                                f"→ TOOL: {tool_name}"
                            )

                            ops_log(
                                "  Raw args: "
                                + raw_arguments
                            )

                            # --------------------------------------------
                            # Parse arguments
                            # --------------------------------------------

                            try:

                                arguments = json.loads(
                                    raw_arguments
                                )

                                if not isinstance(
                                    arguments,
                                    dict,
                                ):

                                    raise ValueError(
                                        "Tool arguments must "
                                        "be a JSON object."
                                    )

                            except Exception as e:

                                result = {
                                    "ok":
                                        False,

                                    "tool":
                                        tool_name,

                                    "error":
                                        {
                                            "code":
                                                "BAD_TOOL_ARGUMENTS",

                                            "message":
                                                str(e),
                                        },
                                }

                                ops_log(
                                    f"✗ {tool_name}: "
                                    "invalid JSON arguments"
                                )

                            else:

                                ops_log(
                                    "  Args: "
                                    + safe_json(
                                        arguments
                                    )
                                )

                                # ----------------------------------------
                                # Execute MCP tool
                                # ----------------------------------------

                                try:

                                    mcp_response = (
                                        await session.call_tool(
                                            tool_name,
                                            arguments,
                                        )
                                    )

                                    result = (
                                        extract_mcp_result(
                                            mcp_response
                                        )
                                    )

                                except Exception as e:

                                    log_exception(
                                        (
                                            f"✗ MCP TOOL ERROR: "
                                            f"{tool_name}"
                                        ),
                                        e,
                                    )

                                    result = {
                                        "ok":
                                            False,

                                        "tool":
                                            tool_name,

                                        "error":
                                            {
                                                "code":
                                                    "MCP_TOOL_ERROR",

                                                "message":
                                                    str(e),

                                                "exception_type":
                                                    type(e).__name__,
                                            },
                                    }

                            # --------------------------------------------
                            # LOG RESULT
                            # --------------------------------------------

                            ops_log(
                                "← RESULT: "
                                + result_preview(
                                    result
                                )
                            )

                            # --------------------------------------------
                            # Return result to LLM
                            # --------------------------------------------

                            messages.append(
                                {
                                    "role":
                                        "tool",

                                    "tool_call_id":
                                        call.id,

                                    "content":
                                        safe_json(
                                            result
                                        ),
                                }
                            )

                        # Continue reasoning.
                        continue

                    # =================================================
                    # FINAL ANSWER
                    # =================================================

                    final_text = (
                        message.content
                        or ""
                    ).strip()

                    if not final_text:

                        final_text = (
                            "The investigation completed, "
                            "but no final response was produced."
                        )

                    ops_log(
                        "✓ INFRA AGENT INVESTIGATION COMPLETE"
                    )

                    ops_log(
                        "  Final response: "
                        + result_preview(
                            final_text,
                            limit=3000,
                        )
                    )

                    ops_log(
                        "=" * 70
                    )

                    return final_text

    except Exception as e:

        log_exception(
            "✗ INFRA AGENT RUN ERROR",
            e,
        )

        raise


# ============================================================
# PROTOBUF HELPERS
# ============================================================

def proto_to_dict(
    message
):

    return json_format.MessageToDict(
        message,
        preserving_proto_field_name=True,
    )


def dict_to_proto(
    data,
    message_type,
):

    message = message_type()

    json_format.ParseDict(
        data,
        message,
    )

    return message


# ============================================================
# BUILD A2A RESPONSE
# ============================================================

def build_a2a_response(
    request_dict,
    final_text,
):

    request = dict_to_proto(
        request_dict,
        SendMessageRequest,
    )

    response = SendMessageResponse()

    response.message.message_id = (
        request.message.message_id
    )

    response.message.role = 2

    part = response.message.parts.add()

    part.text = final_text

    return proto_to_dict(
        response
    )


# ============================================================
# HANDLE A2A REQUEST
# ============================================================

async def handle_a2a_request(
    request_dict
):

    request = dict_to_proto(
        request_dict,
        SendMessageRequest,
    )

    incident_parts = [
        part.text
        for part in request.message.parts
        if part.text
    ]

    incident = "\n".join(
        incident_parts
    ).strip()

    if not incident:

        response = SendMessageResponse()

        response.message.message_id = (
            request.message.message_id
        )

        response.message.role = 2

        part = response.message.parts.add()

        part.text = (
            "No incident description was provided."
        )

        return proto_to_dict(
            response
        )

    ops_log(
        "Received A2A request."
    )

    final_text = await run_infra_agent(
        incident
    )

    return build_a2a_response(
        request_dict,
        final_text,
    )


# ============================================================
# A2A STDIO SERVER
# ============================================================

async def a2a_server():

    print(
        "[Infra Agent] A2A stdio server started.",
        file=sys.stderr,
        flush=True,
    )

    ops_log(
        "Infra Agent process started."
    )

    # stdout is ONLY JSON protocol.
    for line in sys.stdin:

        line = line.strip()

        if not line:
            continue

        request = {}

        try:

            request = json.loads(
                line
            )

            request_id = request.get(
                "id"
            )

            method = request.get(
                "method"
            )

            params = request.get(
                "params",
                {},
            )

            print(
                f"[Infra Agent] A2A request: {method}",
                file=sys.stderr,
                flush=True,
            )

            ops_log(
                f"A2A request received: {method}"
            )

            # ------------------------------------------------
            # SendMessage
            # ------------------------------------------------

            if method == "SendMessage":

                result = (
                    await handle_a2a_request(
                        params
                    )
                )

                response = {
                    "jsonrpc":
                        "2.0",

                    "id":
                        request_id,

                    "result":
                        result,
                }

            # ------------------------------------------------
            # Unsupported method
            # ------------------------------------------------

            else:

                response = {
                    "jsonrpc":
                        "2.0",

                    "id":
                        request_id,

                    "error":
                        {
                            "code":
                                -32601,

                            "message":
                                (
                                    f"Unsupported method: "
                                    f"{method}"
                                ),
                        },
                }

        except Exception as e:

            log_exception(
                "✗ A2A ERROR",
                e,
            )

            response = {
                "jsonrpc":
                    "2.0",

                "id":
                    request.get(
                        "id"
                    ),

                "error":
                    {
                        "code":
                            -32000,

                        "message":
                            str(e),

                        "type":
                            type(e).__name__,
                    },
            }

        # ----------------------------------------------------
        # A2A response
        # ----------------------------------------------------

        print(
            json.dumps(
                response,
                ensure_ascii=False,
            ),
            flush=True,
        )


# ============================================================
# MAIN
# ============================================================

async def main():

    await a2a_server()


if __name__ == "__main__":

    try:

        asyncio.run(
            main()
        )

    except KeyboardInterrupt:

        ops_log(
            "Infra Agent stopped by user."
        )

    except Exception as e:

        log_exception(
            "✗ FATAL INFRA AGENT ERROR",
            e,
        )

        raise