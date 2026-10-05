import os
import time
import json
import re

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.text import Text


LOG_FILE = os.path.abspath(
    "runs/infra_ops.log"
)

console = Console()


# ============================================================
# TABLE FORMATTER
# ============================================================

def flatten_dict(data, prefix=""):

    result = {}

    if isinstance(data, dict):

        for key, value in data.items():

            new_key = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            if isinstance(value, (dict, list)):
                result.update(
                    flatten_dict(
                        value,
                        new_key,
                    )
                )
            else:
                result[new_key] = value

    elif isinstance(data, list):

        for index, value in enumerate(data):

            new_key = (
                f"{prefix}[{index}]"
            )

            if isinstance(value, (dict, list)):
                result.update(
                    flatten_dict(
                        value,
                        new_key,
                    )
                )
            else:
                result[new_key] = value

    else:

        result[prefix] = data

    return result


def json_table(data):

    # --------------------------------------------------------
    # Dict
    # --------------------------------------------------------

    if isinstance(data, dict):

        table = Table(
            show_header=True,
            header_style="bold",
            border_style="dim",
            expand=False,
        )

        table.add_column(
            "FIELD",
            style="bold",
        )

        table.add_column(
            "VALUE",
        )

        flat = flatten_dict(data)

        for key, value in flat.items():

            table.add_row(
                str(key),
                str(value),
            )

        return table

    # --------------------------------------------------------
    # List
    # --------------------------------------------------------

    if isinstance(data, list):

        if not data:

            return Text(
                "[]",
            )

        if all(
            isinstance(item, dict)
            for item in data
        ):

            columns = set()

            for item in data:
                columns.update(
                    item.keys()
                )

            columns = list(columns)

            table = Table(
                show_header=True,
                header_style="bold",
                border_style="dim",
                expand=True,
            )

            for column in columns:

                table.add_column(
                    str(column)
                )

            for item in data:

                table.add_row(
                    *[
                        str(
                            item.get(
                                column,
                                "",
                            )
                        )
                        for column in columns
                    ]
                )

            return table

        table = Table(
            show_header=True,
            header_style="bold",
            border_style="dim",
        )

        table.add_column(
            "#",
            justify="right",
        )

        table.add_column(
            "VALUE"
        )

        for index, item in enumerate(data):

            table.add_row(
                str(index),
                str(item),
            )

        return table

    return Text(str(data))


# ============================================================
# PRINT TOOL RESULT
# ============================================================

def render_tool_result(
    timestamp,
    result_text,
):

    try:

        data = json.loads(
            result_text
        )

    except Exception:

        console.print(
            Text(
                result_text,
            )
        )

        return

    console.print()

    console.print(
        Panel(
            json_table(data),
            title=f"[bold]TOOL RESULT[/bold]  {timestamp}",
            border_style="white",
            padding=(0, 1),
        )
    )

    console.print()


# ============================================================
# FORMAT NORMAL LOG LINE
# ============================================================

def print_log_line(line):

    line = line.rstrip()

    if not line:
        return

    # --------------------------------------------------------
    # Tool result
    # --------------------------------------------------------

    match = re.match(
        r"^\[(.*?)\] ← RESULT: (.*)$",
        line,
    )

    if match:

        timestamp = match.group(1)
        payload = match.group(2)

        render_tool_result(
            timestamp,
            payload,
        )

        return

    # --------------------------------------------------------
    # Tool call
    # --------------------------------------------------------

    match = re.match(
        r"^\[(.*?)\] → TOOL: (.*)$",
        line,
    )

    if match:

        timestamp = match.group(1)
        tool = match.group(2)

        console.print(
            f"[dim][{timestamp}][/dim] "
            f"[bold white]→ TOOL[/bold white] "
            f"[white]{tool}[/white]"
        )

        return

    # --------------------------------------------------------
    # Arguments
    # --------------------------------------------------------

    match = re.match(
        r"^\[(.*?)\]   Args: (.*)$",
        line,
    )

    if match:

        timestamp = match.group(1)
        payload = match.group(2)

        try:

            data = json.loads(
                payload
            )

            table = json_table(data)

            console.print(
                Panel(
                    table,
                    title="ARGS",
                    border_style="dim",
                    padding=(0, 1),
                )
            )

        except Exception:

            console.print(
                f"[dim][{timestamp}][/dim] "
                f"[white]{payload}[/white]"
            )

        return

    # --------------------------------------------------------
    # Agent task
    # --------------------------------------------------------

    if "▶ INFRA AGENT TASK" in line:

        console.print()

        console.print(
            Panel(
                "INFRA / OPS AGENT",
                border_style="white",
                title="AGENT START",
            )
        )

        return

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    if "✓ INFRA AGENT INVESTIGATION COMPLETE" in line:

        console.print()

        console.print(
            Panel(
                "INVESTIGATION COMPLETE",
                border_style="white",
                title="✓ DONE",
            )
        )

        return

    # --------------------------------------------------------
    # MCP
    # --------------------------------------------------------

    if "✓ Connected to MCP" in line:

        console.print(
            f"[dim]{line}[/dim]"
        )

        return

    # --------------------------------------------------------
    # Generic
    # --------------------------------------------------------

    console.print(
        line
    )


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        os.path.dirname(LOG_FILE),
        exist_ok=True,
    )

    open(
        LOG_FILE,
        "a",
    ).close()

    console.clear()

    console.print(
        Panel(
            Text(
                "AGENTIC LAPTOP",
                justify="center",
            ),
            subtitle="INFRA / OPS MONITOR",
            border_style="white",
        )
    )

    console.print(
        f"[dim]Watching {LOG_FILE}[/dim]"
    )

    console.print()

    with open(
        LOG_FILE,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:

        f.seek(
            0,
            os.SEEK_END,
        )

        while True:

            line = f.readline()

            if line:

                print_log_line(
                    line
                )

            else:

                time.sleep(
                    0.08
                )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()