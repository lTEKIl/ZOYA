"""
Generic MCP stdio tool server.

Usage:
    python tool_server.py infra
    python tool_server.py investigator
    python tool_server.py verifier
"""

import asyncio
import importlib
import inspect
import json
import sys
import time
import traceback
import types
from typing import (
    Annotated,
    Any,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    ListToolsResult,
    TextContent,
    Tool,
)

import os as _os
import glob as _glob


def _fix_display_env():
    """
    Guarantee the display env vars are set even if an upstream
    process stripped them. Reads them from the filesystem.
    Idempotent — safe to call multiple times.
    """
    uid = _os.getuid()

    if not _os.environ.get("XDG_RUNTIME_DIR"):
        candidate = f"/run/user/{uid}"
        if _os.path.isdir(candidate):
            _os.environ["XDG_RUNTIME_DIR"] = candidate

    if not _os.environ.get("DBUS_SESSION_BUS_ADDRESS"):
        bus = f"/run/user/{uid}/bus"
        if _os.path.exists(bus):
            _os.environ["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={bus}"

    if not _os.environ.get("WAYLAND_DISPLAY"):
        for name in ("wayland-0", "wayland-1", "wayland-2"):
            if _os.path.exists(f"/run/user/{uid}/{name}"):
                _os.environ["WAYLAND_DISPLAY"] = name
                break

    if not _os.environ.get("DISPLAY"):
        candidates = sorted(_glob.glob("/tmp/.X11-unix/X*"))
        if candidates:
            n = _os.path.basename(candidates[0])[1:]
            _os.environ["DISPLAY"] = f":{n}"

    if not _os.environ.get("XDG_SESSION_TYPE"):
        if _os.environ.get("WAYLAND_DISPLAY"):
            _os.environ["XDG_SESSION_TYPE"] = "wayland"
        elif _os.environ.get("DISPLAY"):
            _os.environ["XDG_SESSION_TYPE"] = "x11"


_fix_display_env()
# ============================================================
# CONFIGURATION
# ============================================================

AGENT_MODULES = {
    "infra": "custom_tools",
    "custom": "custom_tools",
    "investigator": "investigator_tools",
    "verifier": "verifier_tools",
}

ENV = "laptop-local"
TENANT = "local"


# ============================================================
# REGISTRY LOADING
# ============================================================

def load_registry(agent: str):
    """
    Dynamically load the TOOLS registry for an agent.
    """

    module_name = AGENT_MODULES.get(agent)

    if not module_name:
        raise RuntimeError(
            f"Unknown agent '{agent}'. "
            f"Available: {', '.join(sorted(AGENT_MODULES))}"
        )

    module = importlib.import_module(module_name)

    if not hasattr(module, "TOOLS"):
        raise RuntimeError(
            f"{module_name} does not define TOOLS"
        )

    tools = module.TOOLS

    if not isinstance(tools, dict):
        raise RuntimeError(
            f"{module_name}.TOOLS must be a dictionary"
        )

    return tools


# ============================================================
# JSON SAFETY
# ============================================================

def json_safe(value):
    """
    Convert arbitrary Python values into JSON-safe structures.
    """

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            json_safe(v)
            for v in value
        ]

    try:
        json.dumps(value)
        return value

    except Exception:
        return str(value)


# ============================================================
# TYPE -> JSON SCHEMA
# ============================================================

def annotation_to_schema(annotation):
    """
    Convert a Python type annotation into a basic JSON schema.

    Handles:
        str
        int
        float
        bool
        list
        dict
        Literal
        Optional
        Union
        Annotated
    """

    if annotation is inspect.Parameter.empty:
        return {
            "type": "string"
        }

    # --------------------------------------------------------
    # None / NoneType
    # --------------------------------------------------------

    if annotation is type(None):
        return {
            "type": "null"
        }

    # --------------------------------------------------------
    # Resolve typing origin
    # --------------------------------------------------------

    origin = get_origin(annotation)
    args = get_args(annotation)

    # --------------------------------------------------------
    # Annotated[T, ...]
    # --------------------------------------------------------

    if origin is Annotated:
        if args:
            base_type = args[0]
            schema = annotation_to_schema(base_type)

            # Preserve useful Annotated metadata where possible.
            metadata = args[1:]

            for item in metadata:
                # Pydantic FieldInfo-like objects may expose
                # useful constraints.
                if hasattr(item, "ge") and item.ge is not None:
                    schema["minimum"] = item.ge

                if hasattr(item, "gt") and item.gt is not None:
                    schema["exclusiveMinimum"] = item.gt

                if hasattr(item, "le") and item.le is not None:
                    schema["maximum"] = item.le

                if hasattr(item, "lt") and item.lt is not None:
                    schema["exclusiveMaximum"] = item.lt

                if hasattr(item, "description") and item.description:
                    schema["description"] = item.description

                if hasattr(item, "min_length") and item.min_length is not None:
                    schema["minLength"] = item.min_length

                if hasattr(item, "max_length") and item.max_length is not None:
                    schema["maxLength"] = item.max_length

            return schema

    # --------------------------------------------------------
    # Literal["x", "y"] / Literal[1, 2]
    # --------------------------------------------------------

    if origin is Literal:

        values = list(args)

        if not values:
            return {
                "type": "string"
            }

        first = values[0]

        if isinstance(first, bool):
            schema_type = "boolean"

        elif isinstance(first, int):
            schema_type = "integer"

        elif isinstance(first, float):
            schema_type = "number"

        elif first is None:
            schema_type = "null"

        else:
            schema_type = "string"

        return {
            "type": schema_type,
            "enum": values,
        }

    # --------------------------------------------------------
    # Optional[T] / Union[T, None]
    # --------------------------------------------------------

    if origin in (Union, types.UnionType):

        non_none = [
            arg
            for arg in args
            if arg is not type(None)
        ]

        if len(non_none) == 1:
            return annotation_to_schema(non_none[0])

        schemas = [
            annotation_to_schema(arg)
            for arg in non_none
        ]

        if schemas:
            return {
                "anyOf": schemas
            }

    # --------------------------------------------------------
    # Basic Python types
    # --------------------------------------------------------

    if annotation is str:
        return {
            "type": "string"
        }

    if annotation is int:
        return {
            "type": "integer"
        }

    if annotation is float:
        return {
            "type": "number"
        }

    if annotation is bool:
        return {
            "type": "boolean"
        }

    if annotation is list:
        return {
            "type": "array"
        }

    if annotation is dict:
        return {
            "type": "object"
        }

    # --------------------------------------------------------
    # Generic list[T]
    # --------------------------------------------------------

    if origin is list:
        schema = {
            "type": "array"
        }

        if args:
            schema["items"] = annotation_to_schema(args[0])

        return schema

    # --------------------------------------------------------
    # Generic dict[K, V]
    # --------------------------------------------------------

    if origin is dict:
        return {
            "type": "object"
        }

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    return {
        "type": "string"
    }


def get_input_schema(fn):
    """
    Build JSON schema from a tool function's signature.

    Uses get_type_hints(..., include_extras=True) so that
    postponed annotations such as:

        pid: int
        minutes: int
        by: Literal["cpu", "mem", "io"]

    are resolved correctly.
    """

    try:
        signature = inspect.signature(fn)

    except Exception:
        return {
            "type": "object",
            "properties": {},
        }

    # --------------------------------------------------------
    # Resolve annotations properly.
    # --------------------------------------------------------

    try:
        type_hints = get_type_hints(
            fn,
            include_extras=True,
        )

    except Exception:
        # Fall back to raw signature annotations.
        type_hints = {}

    properties = {}
    required = []

    for name, parameter in signature.parameters.items():

        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        # Prefer resolved type hints.
        annotation = type_hints.get(
            name,
            parameter.annotation,
        )

        schema = annotation_to_schema(annotation)

        # ----------------------------------------------------
        # Default value
        # ----------------------------------------------------

        if parameter.default is not inspect.Parameter.empty:

            schema["default"] = json_safe(
                parameter.default
            )

        else:
            required.append(name)

        properties[name] = schema

    result = {
        "type": "object",
        "properties": properties,
    }

    if required:
        result["required"] = required

    return result


# ============================================================
# MCP TOOL DEFINITIONS
# ============================================================

def build_mcp_tools(registry):
    """
    Convert the internal TOOLS registry into MCP Tool objects.
    """

    output = []

    for name in sorted(registry):

        meta = registry[name]

        if not isinstance(meta, dict):
            continue

        fn = meta.get("fn")

        if not callable(fn):
            continue

        description = meta.get(
            "doc",
            "",
        )

        group = meta.get(
            "group",
            "",
        )

        mutating = meta.get(
            "mutating",
            False,
        )

        flags = []

        if group:
            flags.append(
                f"group={group}"
            )

        flags.append(
            "mutating"
            if mutating
            else "read-only"
        )

        if flags:

            description = (
                f"{description} "
                f"[{', '.join(flags)}]"
            ).strip()

        output.append(
            Tool(
                name=name,
                description=description,
                inputSchema=get_input_schema(fn),
            )
        )

    return output


# ============================================================
# STRUCTURED RESPONSES
# ============================================================

def make_error(
    tool_name,
    agent,
    code,
    message,
    retryable=False,
    elapsed_ms=None,
):
    """
    Create a structured tool error envelope.
    """

    result = {
        "ok": False,
        "tool": tool_name,
        "agent": agent,
        "env": ENV,
        "tenant": TENANT,
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
        },
    }

    if elapsed_ms is not None:

        result["meta"] = {
            "elapsed_ms": round(
                elapsed_ms,
                2,
            )
        }

    return result


def make_success(
    tool_name,
    agent,
    result,
    elapsed_ms,
):
    """
    Create a structured successful tool envelope.
    """

    return {
        "ok": True,
        "tool": tool_name,
        "agent": agent,
        "env": ENV,
        "tenant": TENANT,
        "result": json_safe(result),
        "meta": {
            "elapsed_ms": round(
                elapsed_ms,
                2,
            )
        },
    }


# ============================================================
# TOOL EXECUTION
# ============================================================

async def execute_tool(
    registry,
    tool_name,
    arguments,
    agent,
):
    """
    Execute one registry tool safely.
    """

    started = time.perf_counter()

    # --------------------------------------------------------
    # Tool lookup
    # --------------------------------------------------------

    meta = registry.get(tool_name)

    if meta is None:

        return make_error(
            tool_name,
            agent,
            "NOT_FOUND",
            f"unknown tool: {tool_name}",
            False,
            (
                time.perf_counter()
                - started
            ) * 1000,
        )

    # --------------------------------------------------------
    # Registry validation
    # --------------------------------------------------------

    if not isinstance(meta, dict):

        return make_error(
            tool_name,
            agent,
            "INVALID_TOOL",
            "tool registry entry is not a dictionary",
            False,
            (
                time.perf_counter()
                - started
            ) * 1000,
        )

    fn = meta.get("fn")

    if not callable(fn):

        return make_error(
            tool_name,
            agent,
            "INVALID_TOOL",
            "tool registry entry has no callable 'fn'",
            False,
            (
                time.perf_counter()
                - started
            ) * 1000,
        )

    # --------------------------------------------------------
    # Arguments validation
    # --------------------------------------------------------

    if arguments is None:
        arguments = {}

    if not isinstance(arguments, dict):

        return make_error(
            tool_name,
            agent,
            "INVALID_ARGUMENTS",
            "arguments must be an object",
            False,
            (
                time.perf_counter()
                - started
            ) * 1000,
        )

    # --------------------------------------------------------
    # Execute
    # --------------------------------------------------------

    try:

        if inspect.iscoroutinefunction(fn):

            result = await fn(
                **arguments
            )

        else:

            result = await asyncio.to_thread(
                fn,
                **arguments,
            )

        elapsed_ms = (
            time.perf_counter()
            - started
        ) * 1000

        return make_success(
            tool_name,
            agent,
            result,
            elapsed_ms,
        )

    # --------------------------------------------------------
    # Structured tool errors
    # --------------------------------------------------------

    except Exception as exc:

        elapsed_ms = (
            time.perf_counter()
            - started
        ) * 1000

        code = getattr(
            exc,
            "code",
            None,
        )

        message = getattr(
            exc,
            "message",
            None,
        )

        retryable = getattr(
            exc,
            "retryable",
            False,
        )

        # Custom ToolError-style exception.
        if code:

            return make_error(
                tool_name,
                agent,
                code,
                message or str(exc),
                bool(retryable),
                elapsed_ms,
            )

        # Pydantic validation errors.
        if exc.__class__.__name__ in (
            "ValidationError",
            "PydanticValidationError",
        ):

            return make_error(
                tool_name,
                agent,
                "INVALID_ARGUMENTS",
                str(exc),
                False,
                elapsed_ms,
            )

        # Unexpected error.
        traceback.print_exc(
            file=sys.stderr
        )

        return make_error(
            tool_name,
            agent,
            "INTERNAL_ERROR",
            str(exc),
            False,
            elapsed_ms,
        )


# ============================================================
# MCP SERVER
# ============================================================

def create_server(
    agent,
    registry,
):
    """
    Create an MCP Server using the low-level callback API
    supported by the installed MCP SDK.
    """

    async def on_list_tools(
        ctx,
        params,
    ):
        return ListToolsResult(
            tools=build_mcp_tools(
                registry
            )
        )

    async def on_call_tool(
        ctx,
        params,
    ):
        tool_name = params.name

        arguments = (
            params.arguments
            or {}
        )

        result = await execute_tool(
            registry=registry,
            tool_name=tool_name,
            arguments=arguments,
            agent=agent,
        )

        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps(
                        result,
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    ),
                )
            ],
            isError=not result.get(
                "ok",
                False,
            ),
        )

    return Server(
        f"laptop-{agent}-tools",
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


# ============================================================
# SERVER RUNTIME
# ============================================================

async def run_server(agent):
    """
    Start the generic MCP stdio server.
    """

    registry = load_registry(
        agent
    )

    server = create_server(
        agent,
        registry,
    )

    async with stdio_server() as (
        read_stream,
        write_stream,
    ):

        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


# ============================================================
# CLI
# ============================================================

def main():

    if len(sys.argv) < 2:

        print(
            "Usage: python tool_server.py "
            "<infra|investigator|verifier>",
            file=sys.stderr,
        )

        sys.exit(2)

    agent = (
        sys.argv[1]
        .strip()
        .lower()
    )

    if agent not in AGENT_MODULES:

        print(
            f"Unknown agent: {agent}",
            file=sys.stderr,
        )

        print(
            "Available:",
            ", ".join(
                sorted(
                    AGENT_MODULES
                )
            ),
            file=sys.stderr,
        )

        sys.exit(2)

    try:

        asyncio.run(
            run_server(agent)
        )

    except KeyboardInterrupt:
        pass

    except Exception:

        traceback.print_exc(
            file=sys.stderr
        )

        sys.exit(1)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()