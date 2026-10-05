"""
test_mcp.py

Diagnostic client for the generic MCP tool server.

Usage:

    python test_mcp.py
    python test_mcp.py infra
    python test_mcp.py investigator
    python test_mcp.py verifier
"""

import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_AGENT = "infra"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_json(value):

    print(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

async def main():

    agent = (
        sys.argv[1]
        if len(sys.argv) > 1
        else DEFAULT_AGENT
    ).strip().lower()

    print(
        "=" * 70
    )

    print(
        f"MCP TEST CLIENT"
    )

    print(
        f"Agent: {agent}"
    )

    print(
        f"Python: {sys.executable}"
    )

    print(
        "=" * 70
    )

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[
            "tool_server.py",
            agent,
        ],
    )

    print(
        "\nStarting MCP server..."
    )

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

            # ---------------------------------------------------------------
            # Initialize
            # ---------------------------------------------------------------

            print(
                "Initializing MCP session..."
            )

            await session.initialize()

            print(
                "MCP session initialized."
            )

            # ---------------------------------------------------------------
            # List tools
            # ---------------------------------------------------------------

            print(
                "\n"
                + "=" * 70
            )

            print(
                "AVAILABLE TOOLS"
            )

            print(
                "=" * 70
            )

            response = await session.list_tools()

            tools = response.tools

            print(
                f"\nFound {len(tools)} tools.\n"
            )

            for index, tool in enumerate(
                tools,
                start=1,
            ):

                print(
                    f"{index:02d}. {tool.name}"
                )

                print(
                    f"    {tool.description}"
                )

                print(
                    "    schema:"
                )

                print(
                    json.dumps(
                        tool.input_schema,
                        indent=6,
                        ensure_ascii=False,
                    )
                )

                print()

            # ---------------------------------------------------------------
            # Infra-specific execution test
            # ---------------------------------------------------------------

            if agent in (
                "infra",
                "custom",
            ):

                print(
                    "=" * 70
                )

                print(
                    "EXECUTION TEST"
                )

                print(
                    "=" * 70
                )

                print(
                    "\nCalling:"
                )

                print(
                    "  list_processes"
                )

                print(
                    '  arguments = {"limit": 5}'
                )

                result = await session.call_tool(
                    "list_processes",
                    {
                        "limit": 5
                    },
                )

                print(
                    "\nRaw MCP result:"
                )

                for content in result.content:

                    if hasattr(
                        content,
                        "text",
                    ):

                        try:

                            parsed = json.loads(
                                content.text
                            )

                            print_json(
                                parsed
                            )

                        except Exception:

                            print(
                                content.text
                            )

                    else:

                        print(
                            content
                        )

                print(
                    "\n"
                    + "=" * 70
                )

                print(
                    "TEST COMPLETE"
                )

                print(
                    "=" * 70
                )

            else:

                print(
                    "\nNo agent-specific execution "
                    "test configured for this agent."
                )

                print(
                    "Tool discovery succeeded."
                )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    try:

        asyncio.run(
            main()
        )

    except KeyboardInterrupt:

        print(
            "\nInterrupted."
        )

    except Exception as exc:

        print(
            "\nMCP TEST FAILED:"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise