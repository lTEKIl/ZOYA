import asyncio
import json
import os
import re
import sys
import datetime

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

OPS_LOG = os.path.abspath("runs/infra_ops.log")

MAX_ITERATIONS = 6
MAX_TOOL_RESULT_CHARS = 4000
MAX_TOTAL_CONTEXT_CHARS = 20000

MAX_GROQ_RECOVERY_RETRIES = 4


# ============================================================
# OPS LOGGING
# ============================================================

def ops_log(message):
    """
    Shared Ops terminal log.

    stdout is reserved exclusively for A2A JSON-RPC.
    """

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


# ============================================================
# SYSTEM PROMPT
# ============================================================

# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are the Investigator / RAG Agent in a local incident-response
and forensic investigation system.

Your job is to investigate the user's request using the available
read-only Investigator MCP tools.

IMPORTANT:

- All Investigator tools are READ-ONLY.
- Never claim that you changed, deleted, restarted, killed, modified,
  or repaired anything.
- Use actual tool results as evidence.
- Never invent files, processes, users, services, timestamps, paths,
  network connections, packages, or system state.
- Prefer the smallest number of tools necessary to answer the request.
- Do not blindly enumerate every available tool.
- If a tool returns no results, report that accurately.
- Distinguish "not found" from "not checked".
- Do not infer historical facts unless the evidence supports them.
- For filesystem questions, inspect the filesystem directly.
- For process questions, inspect actual processes.
- For network questions, inspect actual network state.
- For security questions, inspect relevant security configuration
  and evidence.
- For binary analysis, use the binary-analysis tools when appropriate.
- For persistence questions, inspect startup entries, cron entries,
  systemd timers, services, etc.
- For logs, respect the tool's supported time range.
- Do not invent timestamps.
- Do not expose internal orchestration details to the user.

SPECIAL RULE FOR ZERO-ARGUMENT TOOLS:

Some tools take no arguments.

When calling a zero-argument tool, the arguments object MUST be:

{}

Never generate an empty-string argument such as:

{"": ""}

or:

{"": null}

If a tool is not necessary, do not call it.

============================================================
TRUNCATION RULE — VERY IMPORTANT
============================================================

If a tool result contains any of:

  - "truncated": true
  - the string "[truncated]"
  - the string "output truncated"

...then you have reached the tool's HARD LIMIT. The limit is a
constraint declared in the tool's schema and it cannot be
exceeded. Asking again with a bigger number will fail.

Do NOT:

  - Retry the same tool with a larger max_entries, max_results
    or limit parameter.
  - Re-request the same list with a different limit to try to
    see more.
  - Try a different list-like tool hoping for a bigger result.
  - Loop. Looping wastes tokens and delays the answer.

DO:

  - Work with the data you have.
  - If the data is sufficient to answer, answer.
  - If the data is insufficient, say so plainly and return
    what you found.
  - Prefer summarization over enumeration when the list is
    large.

Example of the correct behavior:

    User asks: "list my Downloads folder"

    You call: list_directory("/home/.../Downloads", max_entries=500)

    Result: truncated=true, 500 entries shown

    CORRECT next step:
        Summarize the entries you received and return.

    WRONG next step:
        Call list_directory again with max_entries=1000
        (this will be rejected)
        Call find_files with max_results=1000
        (also rejected)

The tool limits exist to keep token usage bounded. Respect
them.

============================================================
SCHEMA VALIDATION ERRORS
============================================================

If a tool call fails with INVALID_ARGUMENTS or a pydantic
validation error, it means the parameter you sent was outside
the allowed range or the wrong type. Do NOT retry with a
larger value. Read the error, adjust to a value that is
within the declared bounds, and try once. If it still fails,
move on.

============================================================
FINAL ANSWER FORMAT
============================================================

After completing the investigation, ALWAYS provide a concise
final answer containing:

1. What was checked.
2. The actual evidence found.
3. The answer/conclusion supported by that evidence.
4. Any important limitation or uncertainty.

If the list of evidence is large, summarize it. Do not paste
hundreds of rows.
"""


# ============================================================
# MCP -> GROQ TOOL FORMAT
# ============================================================

def mcp_tool_to_groq(tool):
    """
    Convert an MCP Tool definition to Groq/OpenAI function format.
    """

    schema = tool.input_schema or {
        "type": "object",
        "properties": {},
    }

    if not isinstance(schema, dict):
        schema = {
            "type": "object",
            "properties": {},
        }

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": (
                tool.description
                or "Read-only investigation tool."
            ),
            "parameters": schema,
        },
    }


# ============================================================
# TOOL SCHEMA HELPERS
# ============================================================

def is_zero_argument_tool(tool):
    """
    Detect tools whose MCP schema exposes no usable parameters.
    """

    schema = tool.input_schema or {}

    properties = schema.get(
        "properties",
        {},
    )

    required = schema.get(
        "required",
        [],
    )

    return (
        not properties
        and not required
    )


def build_tool_constraints_text(tools):
    """
    Build a compact textual description of important parameter
    constraints. This helps reduce invalid Groq tool calls.
    """

    lines = []

    for tool in tools:

        schema = tool.input_schema or {}

        properties = schema.get(
            "properties",
            {},
        )

        required = schema.get(
            "required",
            [],
        )

        if not properties:
            lines.append(
                f"- {tool.name}: no arguments; "
                f"arguments MUST be {{}}"
            )
            continue

        parts = []

        for name, prop in properties.items():

            if not isinstance(prop, dict):
                continue

            ptype = prop.get(
                "type",
                "value",
            )

            minimum = prop.get("minimum")
            maximum = prop.get("maximum")

            enum = prop.get("enum")

            text = f"{name}: {ptype}"

            if minimum is not None:
                text += f", min={minimum}"

            if maximum is not None:
                text += f", max={maximum}"

            if enum:
                text += (
                    ", allowed="
                    + ",".join(
                        str(x)
                        for x in enum
                    )
                )

            if name in required:
                text += ", required"

            parts.append(text)

        lines.append(
            f"- {tool.name}: "
            + "; ".join(parts)
        )

    return "\n".join(lines)


# ============================================================
# GROQ CALL
# ============================================================

def ask_groq(messages, tools):
    """
    Single Groq request.
    """

    return groq.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.1,
    )


# ============================================================
# ERROR PARSING
# ============================================================

def extract_failed_tool_name(error):
    """
    Extract the tool name from Groq's malformed tool-call error.

    Example:

    Failed generation:
    {"name": "current_user", "arguments": {""}}

    -> current_user
    """

    text = str(error)

    patterns = [
        r'"name"\s*:\s*"([^"]+)"',
        r"'name'\s*:\s*'([^']+)'",
        r'tool[ _-]?name["\']?\s*[:=]\s*["\']([^"\']+)',
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE,
        )

        if match:
            return match.group(1)

    return None


def is_malformed_tool_call_error(error):
    """
    Detect Groq errors where the model generated invalid JSON
    for a function/tool call.
    """

    text = str(error).lower()

    indicators = [
        "failed to parse tool call arguments as json",
        "tool_use_failed",
        "failed_generation",
        "invalid_request_error",
    ]

    return any(
        indicator in text
        for indicator in indicators
    )


def extract_validation_details(error):
    """
    Extract useful information from MCP/Groq validation errors.

    This is intentionally generic because exact error formatting
    can vary between versions.
    """

    text = str(error)

    tool_name = None
    parameter = None
    maximum = None
    minimum = None

    tool_patterns = [
        r'tool[ =:"\']+([A-Za-z0-9_.-]+)',
        r'function[ =:"\']+([A-Za-z0-9_.-]+)',
    ]

    for pattern in tool_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE,
        )

        if match:
            tool_name = match.group(1)
            break

    parameter_patterns = [
        r'parameter[ =:"\']+([A-Za-z0-9_.-]+)',
        r'field[ =:"\']+([A-Za-z0-9_.-]+)',
    ]

    for pattern in parameter_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE,
        )

        if match:
            parameter = match.group(1)
            break

    match = re.search(
        r'(?:maximum|max)[^0-9]*([0-9]+)',
        text,
        re.IGNORECASE,
    )

    if match:
        maximum = match.group(1)

    match = re.search(
        r'(?:minimum|min)[^0-9]*([0-9]+)',
        text,
        re.IGNORECASE,
    )

    if match:
        minimum = match.group(1)

    return {
        "tool": tool_name,
        "parameter": parameter,
        "maximum": maximum,
        "minimum": minimum,
    }


# ============================================================
# MCP RESULT EXTRACTION
# ============================================================

def compact_value(value, limit=MAX_TOOL_RESULT_CHARS):
    """
    Keep MCP evidence from exploding the Groq context.
    """

    try:
        text = json.dumps(
            value,
            default=str,
            ensure_ascii=False,
        )
    except Exception:
        text = str(value)

    if len(text) <= limit:
        return value

    return (
        text[:limit]
        + "\n...[RESULT TRUNCATED]..."
    )


def extract_mcp_result(mcp_response):
    """
    Convert MCP CallToolResult into JSON-friendly data.
    """

    results = []

    content_items = getattr(
        mcp_response,
        "content",
        [],
    )

    for content in content_items:

        if hasattr(content, "text"):

            text = content.text

            try:
                results.append(
                    json.loads(text)
                )

            except Exception:
                results.append(text)

        elif hasattr(content, "data"):

            results.append(
                content.data
            )

        else:

            results.append(
                str(content)
            )

    # Some MCP implementations may expose structured content.
    structured = getattr(
        mcp_response,
        "structuredContent",
        None,
    )

    if structured is not None:

        if not results:
            return compact_value(
                structured
            )

        results.append(
            structured
        )

    if len(results) == 1:
        return compact_value(
            results[0]
        )

    return compact_value(
        results
    )


# ============================================================
# CONTEXT MANAGEMENT
# ============================================================

def estimate_message_size(messages):
    total = 0

    for message in messages:

        try:
            total += len(
                json.dumps(
                    message,
                    default=str,
                    ensure_ascii=False,
                )
            )
        except Exception:
            total += len(
                str(message)
            )

    return total


def compact_messages(messages):
    """
    Compact old tool results if the investigation becomes large.

    Keep:
    - system prompt
    - recent assistant/tool interaction
    - user request

    Replace older tool outputs with a short marker.
    """

    if estimate_message_size(
        messages
    ) <= MAX_TOTAL_CONTEXT_CHARS:

        return messages

    ops_log(
        "⚠ Groq context became large; "
        "compacting older tool results."
    )

    compacted = []

    for index, message in enumerate(
        messages
    ):

        if index < 2:
            compacted.append(
                message
            )
            continue

        role = message.get(
            "role"
        )

        if role == "tool":

            content = message.get(
                "content",
                "",
            )

            if len(str(content)) > 2500:

                compacted.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.get(
                            "tool_call_id"
                        ),
                        "content": (
                            "[Older tool result "
                            "compacted. Use newer "
                            "evidence or call the "
                            "tool again if needed.]"
                        ),
                    }
                )

            else:

                compacted.append(
                    message
                )

        else:

            compacted.append(
                message
            )

    return compacted


# ============================================================
# SAFE TOOL ARGUMENT NORMALIZATION
# ============================================================

def normalize_tool_arguments(
    tool_name,
    arguments,
    zero_argument_tools,
):
    """
    Normalize harmless malformed arguments before MCP execution.

    Specifically handles:

        {"": ""}

        {"": null}

    for tools that genuinely accept no arguments.
    """

    if arguments is None:
        return {}

    if not isinstance(
        arguments,
        dict,
    ):
        return arguments

    if tool_name not in zero_argument_tools:
        return arguments

    if not arguments:
        return {}

    # Groq sometimes produces:
    #
    # {"": ""}
    #
    # or:
    #
    # {"": null}
    #
    # for zero-argument functions.

    if all(
        str(key).strip() == ""
        for key in arguments.keys()
    ):
        return {}

    return arguments


# ============================================================
# RUN INVESTIGATOR AGENT
# ============================================================

async def _run_investigator_agent_inner(
    investigation_task
):

    ops_log("=" * 70)
    ops_log("▶ INVESTIGATOR AGENT TASK")
    ops_log(
        f"Request: {investigation_task}"
    )

    # --------------------------------------------------------
    # Start Investigator MCP server
    # --------------------------------------------------------

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[
            "tool_server.py",
            "investigator",
        ],
    )

    try:

        async with stdio_client(
            server_params
        ) as (
            read_stream,
            write_stream,
        ):

            async with ClientSession(
                read_stream,
                write_stream,
            ) as session:

                # ------------------------------------------------
                # Initialize MCP
                # ------------------------------------------------

                ops_log(
                    "→ Initializing MCP connection..."
                )

                await session.initialize()

                ops_log(
                    "✓ MCP connection initialized."
                )

                # ------------------------------------------------
                # Discover tools
                # ------------------------------------------------

                ops_log(
                    "→ Discovering Investigator MCP tools..."
                )

                tool_result = (
                    await session.list_tools()
                )

                mcp_tools = tool_result.tools

                ops_log(
                    "✓ Connected to Investigator MCP "
                    f"({len(mcp_tools)} tools available)"
                )

                ops_log(
                    "Available tools: "
                    + ", ".join(
                        tool.name
                        for tool in mcp_tools
                    )
                )

                # ------------------------------------------------
                # Convert MCP tools to Groq tools
                # ------------------------------------------------

                groq_tools = [
                    mcp_tool_to_groq(
                        tool
                    )
                    for tool in mcp_tools
                ]

                # ------------------------------------------------
                # Track zero-argument tools
                # ------------------------------------------------

                zero_argument_tools = {
                    tool.name
                    for tool in mcp_tools
                    if is_zero_argument_tool(
                        tool
                    )
                }

                ops_log(
                    "Zero-argument tools: "
                    + (
                        ", ".join(
                            sorted(
                                zero_argument_tools
                            )
                        )
                        if zero_argument_tools
                        else "none"
                    )
                )

                # ------------------------------------------------
                # Tool constraints
                # ------------------------------------------------

                constraints = (
                    build_tool_constraints_text(
                        mcp_tools
                    )
                )

                # ------------------------------------------------
                # Messages
                # ------------------------------------------------

                messages = [
                    {
                        "role": "system",
                        "content": (
                            SYSTEM_PROMPT
                            + "\n\n"
                            + "AVAILABLE TOOL CONSTRAINTS:\n"
                            + constraints
                        ),
                    },
                    {
                        "role": "user",
                        "content": investigation_task,
                    },
                ]

                # ------------------------------------------------
                # Disabled tools for this investigation
                # ------------------------------------------------

                disabled_tools = set()

                # ------------------------------------------------
                # Main reasoning loop
                # ------------------------------------------------

                for iteration in range(
                    1,
                    MAX_ITERATIONS + 1,
                ):

                    ops_log(
                        f"→ LLM reasoning iteration "
                        f"{iteration}"
                    )

                    # --------------------------------------------
                    # Remove disabled tools
                    # --------------------------------------------

                    active_groq_tools = [
                        tool
                        for tool in groq_tools
                        if tool["function"]["name"]
                        not in disabled_tools
                    ]

                    # --------------------------------------------
                    # Keep context under control
                    # --------------------------------------------

                    messages = compact_messages(
                        messages
                    )

                    # --------------------------------------------
                    # Groq recovery loop
                    # --------------------------------------------

                    message = None

                    recovery_count = 0

                    while (
                        recovery_count
                        <= MAX_GROQ_RECOVERY_RETRIES
                    ):

                        try:

                            response = await asyncio.to_thread(
                                ask_groq,
                                messages,
                                active_groq_tools,
                            )

                            message = (
                                response.choices[0].message
                            )

                            break

                        except Exception as e:

                            error_text = str(e)

                            # ====================================
                            # FIX: MALFORMED GROQ TOOL CALL
                            # ====================================

                            if is_malformed_tool_call_error(
                                e
                            ):

                                failed_tool = (
                                    extract_failed_tool_name(
                                        e
                                    )
                                )

                                if failed_tool:

                                    ops_log(
                                        "⚠ Groq generated malformed "
                                        f"arguments for tool: "
                                        f"{failed_tool}"
                                    )

                                    if (
                                        failed_tool
                                        not in disabled_tools
                                    ):

                                        disabled_tools.add(
                                            failed_tool
                                        )

                                        ops_log(
                                            "→ Temporarily disabling "
                                            f"tool '{failed_tool}' "
                                            "for this investigation "
                                            "and retrying."
                                        )

                                        messages.append(
                                            {
                                                "role": "system",
                                                "content": (
                                                    f"The tool "
                                                    f"'{failed_tool}' "
                                                    "generated an invalid "
                                                    "tool call. Do NOT "
                                                    f"call '{failed_tool}' "
                                                    "again during this "
                                                    "investigation. "
                                                    "Use another appropriate "
                                                    "read-only tool."
                                                ),
                                            }
                                        )

                                        active_groq_tools = [
                                            tool
                                            for tool in groq_tools
                                            if tool[
                                                "function"
                                            ][
                                                "name"
                                            ]
                                            not in disabled_tools
                                        ]

                                        recovery_count += 1

                                        continue

                                # --------------------------------
                                # No tool name could be extracted.
                                #
                                # Disable all zero-argument tools
                                # because this failure is commonly
                                # caused by malformed no-arg calls.
                                # --------------------------------

                                if zero_argument_tools:

                                    newly_disabled = (
                                        zero_argument_tools
                                        - disabled_tools
                                    )

                                    if newly_disabled:

                                        disabled_tools.update(
                                            newly_disabled
                                        )

                                        ops_log(
                                            "⚠ Could not identify "
                                            "the malformed tool."
                                        )

                                        ops_log(
                                            "→ Temporarily disabling "
                                            "zero-argument tools: "
                                            + ", ".join(
                                                sorted(
                                                    newly_disabled
                                                )
                                            )
                                        )

                                        messages.append(
                                            {
                                                "role": "system",
                                                "content": (
                                                    "A previous tool "
                                                    "call contained "
                                                    "malformed JSON. "
                                                    "Do not call "
                                                    "zero-argument "
                                                    "tools during the "
                                                    "next attempt. "
                                                    "Use another "
                                                    "read-only tool."
                                                ),
                                            }
                                        )

                                        active_groq_tools = [
                                            tool
                                            for tool in groq_tools
                                            if tool[
                                                "function"
                                            ][
                                                "name"
                                            ]
                                            not in disabled_tools
                                        ]

                                        recovery_count += 1

                                        continue

                            # ====================================
                            # OTHER GROQ ERROR
                            # ====================================

                            ops_log(
                                f"✗ GROQ ERROR: {error_text}"
                            )

                            raise

                    # ------------------------------------------------
                    # No response after recovery
                    # ------------------------------------------------

                    if message is None:

                        ops_log(
                            "✗ Groq failed to produce a response "
                            "after recovery attempts."
                        )

                        return (
                            "The investigation could not be completed "
                            "because the reasoning model repeatedly "
                            "generated invalid tool calls."
                        )

                    # =================================================
                    # TOOL CALLS
                    # =================================================

                    if message.tool_calls:

                        assistant_tool_calls = []

                        for call in message.tool_calls:

                            assistant_tool_calls.append(
                                {
                                    "id": call.id,
                                    "type": "function",
                                    "function": {
                                        "name": (
                                            call.function.name
                                        ),
                                        "arguments": (
                                            call.function.arguments
                                        ),
                                    },
                                }
                            )

                        messages.append(
                            {
                                "role": "assistant",
                                "content": (
                                    message.content
                                    or ""
                                ),
                                "tool_calls": (
                                    assistant_tool_calls
                                ),
                            }
                        )

                        # --------------------------------------------
                        # Execute each requested tool
                        # --------------------------------------------

                        for call in message.tool_calls:

                            tool_name = (
                                call.function.name
                            )

                            raw_arguments = (
                                call.function.arguments
                            )

                            # ----------------------------------------
                            # Parse arguments
                            # ----------------------------------------

                            try:

                                arguments = json.loads(
                                    raw_arguments
                                    or "{}"
                                )

                            except json.JSONDecodeError:

                                ops_log(
                                    f"✗ {tool_name}: "
                                    "invalid JSON arguments"
                                )

                                result = {
                                    "ok": False,
                                    "error": {
                                        "code": (
                                            "INVALID_TOOL_ARGUMENTS"
                                        ),
                                        "message": (
                                            "The model generated "
                                            "invalid JSON arguments "
                                            "for this tool."
                                        ),
                                    },
                                }

                            else:

                                # ------------------------------------
                                # Normalize zero-arg malformed calls
                                # ------------------------------------

                                arguments = (
                                    normalize_tool_arguments(
                                        tool_name,
                                        arguments,
                                        zero_argument_tools,
                                    )
                                )

                                # ------------------------------------
                                # Skip tool if disabled
                                # ------------------------------------

                                if (
                                    tool_name
                                    in disabled_tools
                                ):

                                    ops_log(
                                        f"⚠ Skipping disabled "
                                        f"tool: {tool_name}"
                                    )

                                    result = {
                                        "ok": False,
                                        "error": {
                                            "code": (
                                                "TOOL_TEMPORARILY_DISABLED"
                                            ),
                                            "message": (
                                                "This tool was "
                                                "temporarily disabled "
                                                "because the reasoning "
                                                "model generated an "
                                                "invalid call for it."
                                            ),
                                        },
                                    }

                                else:

                                    # --------------------------------
                                    # Log tool call
                                    # --------------------------------

                                    ops_log(
                                        f"→ TOOL: "
                                        f"{tool_name}"
                                    )

                                    ops_log(
                                        "  Args: "
                                        + json.dumps(
                                            arguments,
                                            default=str,
                                        )
                                    )

                                    # --------------------------------
                                    # Execute MCP tool
                                    # --------------------------------

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

                                        validation = (
                                            extract_validation_details(
                                                e
                                            )
                                        )

                                        ops_log(
                                            f"✗ MCP TOOL ERROR: "
                                            f"{tool_name}: {e}"
                                        )

                                        # --------------------------------
                                        # Validation recovery
                                        # --------------------------------

                                        if (
                                            validation["tool"]
                                            and validation[
                                                "tool"
                                            ]
                                            == tool_name
                                        ):

                                            correction = (
                                                "The previous call "
                                                f"to '{tool_name}' "
                                                "had invalid parameters."
                                            )

                                            if validation[
                                                "parameter"
                                            ]:

                                                correction += (
                                                    " Parameter "
                                                    f"'{validation['parameter']}' "
                                                    "was invalid."
                                                )

                                            if validation[
                                                "maximum"
                                            ]:

                                                correction += (
                                                    " Maximum allowed "
                                                    f"value is "
                                                    f"{validation['maximum']}."
                                                )

                                            if validation[
                                                "minimum"
                                            ]:

                                                correction += (
                                                    " Minimum allowed "
                                                    f"value is "
                                                    f"{validation['minimum']}."
                                                )

                                            messages.append(
                                                {
                                                    "role": "system",
                                                    "content": (
                                                        correction
                                                        + " Retry with "
                                                        "parameters that "
                                                        "satisfy the tool "
                                                        "schema."
                                                    ),
                                                }
                                            )

                                        result = {
                                            "ok": False,
                                            "error": {
                                                "code": (
                                                    "MCP_TOOL_ERROR"
                                                ),
                                                "message": str(e),
                                                "details": validation,
                                            },
                                        }
                                    if isinstance(result, dict) and result.get("truncated") is True:
                                        result["_agent_note"] = (
                                                            "This result is truncated at the tool's hard limit. "
                                                            "Do NOT retry with a larger limit. Work with this data "
                                                            "and return your answer."
                                                            )   

                                    # --------------------------------
                                    # Log result
                                    # --------------------------------

                                    try:

                                        result_preview = (
                                            json.dumps(
                                                result,
                                                default=str,
                                                ensure_ascii=False,
                                            )
                                        )

                                    except Exception:

                                        result_preview = str(
                                            result
                                        )

                                    if (
                                        len(
                                            result_preview
                                        )
                                        > 4000
                                    ):

                                        result_preview = (
                                            result_preview[:4000]
                                            + "... [truncated]"
                                        )

                                    ops_log(
                                        "← RESULT: "
                                        + result_preview
                                    )

                            # ----------------------------------------
                            # Return result to Groq
                            # ----------------------------------------

                            try:

                                tool_content = json.dumps(
                                    result,
                                    default=str,
                                    ensure_ascii=False,
                                )

                            except Exception:

                                tool_content = str(
                                    result
                                )

                            # Prevent giant individual tool
                            # responses from entering the context.

                            if (
                                len(tool_content)
                                > MAX_TOOL_RESULT_CHARS
                            ):

                                tool_content = (
                                    tool_content[
                                        :MAX_TOOL_RESULT_CHARS
                                    ]
                                    + "\n...[TRUNCATED]..."
                                )

                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": call.id,
                                    "content": tool_content,
                                }
                            )

                        # --------------------------------------------
                        # Continue reasoning
                        # --------------------------------------------

                        continue

                    # =================================================
                    # FINAL ANSWER
                    # =================================================

                    final_text = (
                        message.content
                        or ""
                    ).strip()

                    if not final_text:

                        ops_log(
                            "⚠ Investigator returned an empty "
                            "final response."
                        )

                        final_text = (
                            "The investigation completed, but "
                            "the reasoning model did not return "
                            "a textual summary."
                        )

                    ops_log(
                        "✓ INVESTIGATOR RESPONSE RECEIVED"
                    )

                    ops_log("=" * 70)

                    return final_text

                # =====================================================
                # ITERATION LIMIT
                # =====================================================

                ops_log(
                    "⚠ Investigator reached maximum reasoning "
                    f"iterations ({MAX_ITERATIONS})."
                )

                return (
                    "The investigation reached its reasoning "
                    "limit before a final conclusion could be produced."
                )

    except Exception as e:

        ops_log(
            f"✗ INVESTIGATOR AGENT ERROR: {e}"
        )

        return (
            "The investigation failed: "
            + str(e)
        )

async def run_investigator_agent(investigation_task):
    """
    Outer wrapper: NEVER let an async-context-manager exception
    escape without returning a plain string.
    """
    try:
        return await _run_investigator_agent_inner(
            investigation_task
        )
    except BaseException as e:
        try:
            ops_log(
                "✗ INVESTIGATOR OUTER ERROR: "
                f"{type(e).__name__}: {e}"
            )
        except Exception:
            pass
        return (
            "The investigation could not be completed "
            "due to an internal error."
        )

# ============================================================
# PROTOBUF HELPERS
# ============================================================

def proto_to_dict(message):

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

    # A2A Role enum:
    # 1 = user
    # 2 = agent
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

    investigation_task = (
        "\n".join(
            incident_parts
        ).strip()
    )

    ops_log(
        "Received Investigator A2A request."
    )

    ops_log(
        "Investigation task: "
        + investigation_task
    )

    if not investigation_task:

        response = SendMessageResponse()

        response.message.message_id = (
            request.message.message_id
        )

        response.message.role = 2

        part = response.message.parts.add()

        part.text = (
            "No investigation task was provided."
        )

        return proto_to_dict(
            response
        )

    final_text = (
        await run_investigator_agent(
            investigation_task
        )
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
        "[Investigator Agent] "
        "A2A stdio server started.",
        file=sys.stderr,
    )

    ops_log(
        "Investigator Agent process started."
    )

    for line in sys.stdin:

        line = line.strip()

        if not line:
            continue

        request = {}

        try:

            # ------------------------------------------------
            # Parse outer JSON-RPC envelope
            # ------------------------------------------------

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
                "[Investigator Agent] "
                f"A2A request received: {method}",
                file=sys.stderr,
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
                        "message": (
                            f"Unsupported method: "
                            f"{method}"
                        ),
                    },
                }

        except Exception as e:

            print(
                "[Investigator Agent] "
                f"ERROR: {e}",
                file=sys.stderr,
            )

            ops_log(
                f"✗ A2A ERROR: {e}"
            )

            response = {
                "jsonrpc": "2.0",
                "id": request.get(
                    "id"
                ),
                "error": {
                    "code": -32000,
                    "message": str(e),
                },
            }

        # IMPORTANT:
        # stdout is ONLY for A2A protocol.

        print(
            json.dumps(
                response,
                default=str,
            ),
            flush=True,
        )


# ============================================================
# MAIN
# ============================================================

async def main():

    await a2a_server()


if __name__ == "__main__":

    asyncio.run(
        main()
    )