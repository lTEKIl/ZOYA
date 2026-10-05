"""
custom_tools.py

Read-only tools for the Infra Agent.

Groups:
    A - Process identification
    B - Process/resource usage
    C - Files, sockets and connections
    D - System health
    E - Logs and events
    F - Application information
    G - Audio
    H - Network

IMPORTANT:
    This file intentionally contains NO action tools.

    Infra Agent can:
        - inspect
        - measure
        - query
        - diagnose
        - collect evidence

    Infra Agent CANNOT:
        - kill processes
        - restart processes
        - stop services
        - restart services
        - modify files
        - change network configuration
        - change system configuration
        - change volume
        - install/remove software
        - execute arbitrary shell commands

Dangerous / mutating operations will eventually live behind
the separate Verifier system.
"""

import functools
import glob
import os
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Annotated, Literal, Optional

import psutil
from pydantic import Field, validate_call


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

ENV = "laptop-local"


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Pid = Annotated[
    int,
    Field(gt=0),
]

Name = Annotated[
    str,
    Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._ +-]*$",
    ),
]

Host = Annotated[
    str,
    Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9.-]{0,252}$",
    ),
]

Unit = Annotated[
    str,
    Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9@._:-]{0,79}$",
    ),
]

Minutes = Annotated[
    int,
    Field(
        ge=1,
        le=240,
    ),
]

Lines = Annotated[
    int,
    Field(
        ge=1,
        le=200,
    ),
]


# ---------------------------------------------------------------------------
# Structured tool error
# ---------------------------------------------------------------------------

class ToolError(Exception):
    """
    Structured failure.

    The tool server converts this into an error envelope.
    """

    def __init__(
        self,
        code,
        message,
        retryable=False,
    ):
        super().__init__(message)

        self.code = code
        self.message = message
        self.retryable = retryable


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOLS = {}


def tool(group):
    """
    Register a read-only tool.

    Every Infra tool is non-mutating.

    Argument validation is performed through Pydantic's validate_call.
    Common psutil failures are converted into ToolError.
    """

    def deco(fn):

        checked = validate_call(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):

            try:

                return checked(
                    *args,
                    **kwargs,
                )

            except psutil.NoSuchProcess:

                raise ToolError(
                    "NOT_FOUND",
                    "that process no longer exists",
                )

            except psutil.AccessDenied:

                raise ToolError(
                    "ACCESS_DENIED",
                    "not allowed to read that process",
                )

        doc = (
            fn.__doc__
            or ""
        ).strip().splitlines()

        TOOLS[fn.__name__] = {
            "fn": wrapper,
            "group": group,
            "mutating": False,
            "doc": doc[0] if doc else "",
        }

        return wrapper

    return deco


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(
    fn,
    default=None,
):
    try:

        return fn()

    except (
        psutil.Error,
        OSError,
        AttributeError,
    ):

        return default


def _mb(
    value,
):
    return round(
        value / 2**20,
        1,
    )


def _run(
    cmd,
    timeout=6,
):
    """
    Run a fixed command.

    IMPORTANT:
        Commands are NEVER executed through a shell.

    This is intentionally limited to internally-defined commands.
    It is NOT an arbitrary command execution interface.
    """

    exe = shutil.which(
        cmd[0]
    )

    if not exe:

        raise ToolError(
            "UNAVAILABLE",
            f"{cmd[0]} is not installed",
        )

    try:

        result = subprocess.run(
            [
                exe,
                *cmd[1:],
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    except subprocess.TimeoutExpired:

        raise ToolError(
            "TIMEOUT",
            f"{cmd[0]} timed out",
            True,
        )

    if (
        result.returncode != 0
        and not result.stdout
    ):

        raise ToolError(
            "COMMAND_FAILED",
            (
                result.stderr
                or f"{cmd[0]} failed"
            ).strip()[:200],
        )

    return result.stdout


def _proc(
    pid,
):
    return psutil.Process(
        pid
    )


def _match(
    name,
    exact=False,
):
    """
    Find processes by name.

    exact=False:
        substring matching for discovery.

    exact=True:
        exact process/cmd basename matching.

    Infra currently uses this only for observation.
    """

    search = name.lower()

    my_pid = os.getpid()

    output = []

    for process in psutil.process_iter(
        [
            "pid",
            "name",
        ]
    ):

        try:

            if process.pid == my_pid:
                continue

            process_name = (
                process.info["name"]
                or ""
            ).lower()

            cmdline = _safe(
                process.cmdline,
                [],
            )

            cmd0 = os.path.basename(
                cmdline[0]
                if cmdline
                else ""
            ).lower()

            if exact:

                hit = (
                    search
                    in (
                        process_name,
                        cmd0,
                    )
                )

            else:

                hit = (
                    search in process_name
                    or search in cmd0
                )

            if hit:

                output.append(
                    process
                )

        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
        ):

            pass

    return output


def _pids(
    name,
    exact=True,
):

    return [
        process.pid
        for process in _match(
            name,
            exact,
        )
    ]


def _label(
    process,
):

    return (
        f"{_safe(process.name, '?')}"
        f":{process.pid}"
    )


def _dir_mb(
    path,
    max_seconds=3.0,
):

    total = 0
    start = time.time()

    for root, _dirs, files in os.walk(
        path
    ):

        for filename in files:

            total += _safe(
                lambda: os.lstat(
                    os.path.join(
                        root,
                        filename,
                    )
                ).st_size,
                0,
            )

        if (
            time.time() - start
            > max_seconds
        ):

            break

    return _mb(
        total
    )


# ===========================================================================
# A. IDENTIFY
# ===========================================================================

@tool("A")
def list_processes(
    limit: Annotated[
        int,
        Field(
            ge=1,
            le=50,
        ),
    ] = 10,
) -> list:
    """
    Top running processes by memory.
    """

    rows = []

    for process in psutil.process_iter(
        [
            "pid",
            "name",
            "memory_info",
        ]
    ):

        memory_info = (
            process.info.get(
                "memory_info"
            )
        )

        if memory_info:

            rows.append(
                {
                    "pid":
                        process.pid,

                    "name":
                        process.info["name"],

                    "rss_mb":
                        _mb(
                            memory_info.rss
                        ),
                }
            )

    return sorted(
        rows,
        key=lambda row: -row["rss_mb"],
    )[:limit]


@tool("A")
def find_process(
    name: Name,
) -> list:
    """
    App name -> matching PIDs.
    """

    output = [
        {
            "pid":
                process.pid,

            "name":
                _safe(
                    process.name
                ),

            "cmd":
                " ".join(
                    _safe(
                        process.cmdline,
                        [],
                    )[:3]
                )[:120],
        }

        for process in _match(
            name
        )
    ]

    if not output:

        raise ToolError(
            "NOT_FOUND",
            f"no running process matches {name!r}",
        )

    return output


@tool("A")
def get_process_info(
    pid: Pid,
) -> dict:
    """
    Name, state, user, uptime, command line and parent of one PID.
    """

    process = _proc(
        pid
    )

    with process.oneshot():

        parent = _safe(
            process.parent
        )

        return {
            "pid":
                pid,

            "name":
                process.name(),

            "status":
                process.status(),

            "user":
                _safe(
                    process.username
                ),

            "uptime_s":
                int(
                    time.time()
                    - process.create_time()
                ),

            "cmdline":
                " ".join(
                    _safe(
                        process.cmdline,
                        [],
                    )
                )[:300],

            "exe":
                _safe(
                    process.exe
                ),

            "ppid":
                process.ppid(),

            "parent":
                (
                    _safe(
                        parent.name
                    )
                    if parent
                    else None
                ),

            "threads":
                _safe(
                    process.num_threads
                ),

            "nice":
                _safe(
                    process.nice
                ),
        }


@tool("A")
def get_process_tree(
    pid: Pid,
) -> dict:
    """
    Parent and all children of a PID.
    Edges point dependency -> dependent.
    """

    process = _proc(
        pid
    )

    parent = _safe(
        process.parent
    )

    children = process.children(
        recursive=True
    )

    edges = []

    if parent:

        edges.append(
            [
                _label(parent),
                _label(process),
            ]
        )

    for child in children:

        child_parent = _safe(
            child.parent
        )

        if child_parent:

            edges.append(
                [
                    _label(child_parent),
                    _label(child),
                ]
            )

    return {
        "pid":
            pid,

        "parent":
            (
                _label(parent)
                if parent
                else None
            ),

        "children":
            [
                _label(child)
                for child in children
            ],

        "edges":
            edges,
    }


# ===========================================================================
# B. USAGE
# ===========================================================================

@tool("B")
def get_process_usage(
    pid: Pid,
) -> dict:
    """
    CPU, memory, threads and disk I/O of one PID.
    """

    process = _proc(
        pid
    )

    process.cpu_percent(
        None
    )

    time.sleep(
        0.3
    )

    with process.oneshot():

        memory_info = (
            process.memory_info()
        )

        io = _safe(
            process.io_counters
        )

        return {
            "pid":
                pid,

            "name":
                process.name(),

            "cpu_pct":
                process.cpu_percent(
                    None
                ),

            "rss_mb":
                _mb(
                    memory_info.rss
                ),

            "vms_mb":
                _mb(
                    memory_info.vms
                ),

            "threads":
                process.num_threads(),

            "io_read_mb":
                (
                    _mb(
                        io.read_bytes
                    )
                    if io
                    else None
                ),

            "io_write_mb":
                (
                    _mb(
                        io.write_bytes
                    )
                    if io
                    else None
                ),

            "fds":
                _safe(
                    process.num_fds
                ),
        }


@tool("B")
def get_usage_trend(
    pid: Pid,
    seconds: Annotated[
        int,
        Field(
            ge=1,
            le=8,
        ),
    ] = 5,
) -> dict:
    """
    CPU and memory samples over a few seconds.
    """

    process = _proc(
        pid
    )

    process.cpu_percent(
        None
    )

    samples = []

    start = time.time()

    for _ in range(
        seconds
    ):

        time.sleep(
            1
        )

        samples.append(
            {
                "t":
                    round(
                        time.time()
                        - start,
                        1,
                    ),

                "cpu_pct":
                    process.cpu_percent(
                        None
                    ),

                "rss_mb":
                    _mb(
                        process.memory_info().rss
                    ),
            }
        )

    cpus = [
        sample["cpu_pct"]
        for sample in samples
    ]

    return {
        "pid":
            pid,

        "samples":
            samples,

        "rss_growth_mb":
            round(
                samples[-1]["rss_mb"]
                - samples[0]["rss_mb"],
                1,
            ),

        "cpu_avg":
            round(
                sum(cpus)
                / len(cpus),
                1,
            ),

        "cpu_max":
            max(cpus),
    }


@tool("B")
def get_top_processes(
    by: Literal[
        "cpu",
        "mem",
        "io",
    ] = "cpu",

    n: Annotated[
        int,
        Field(
            ge=1,
            le=20,
        ),
    ] = 5,
) -> dict:
    """
    Biggest consumers of CPU, memory or disk I/O.
    """

    processes = list(
        psutil.process_iter(
            [
                "pid",
                "name",
            ]
        )
    )

    if by == "cpu":

        for process in processes:

            _safe(
                lambda: process.cpu_percent(
                    None
                )
            )

        time.sleep(
            0.4
        )

    rows = []

    for process in processes:

        try:

            if by == "cpu":

                value = (
                    process.cpu_percent(
                        None
                    )
                )

            elif by == "mem":

                value = _mb(
                    process.memory_info().rss
                )

            else:

                io = process.io_counters()

                value = _mb(
                    io.read_bytes
                    + io.write_bytes
                )

        except (
            psutil.Error,
            OSError,
            AttributeError,
        ):

            continue

        rows.append(
            {
                "pid":
                    process.pid,

                "name":
                    process.info["name"],

                "value":
                    value,
            }
        )

    unit = {
        "cpu": "%",
        "mem": "MB",
        "io": "MB total",
    }[by]

    return {
        "by":
            by,

        "unit":
            unit,

        "top":
            sorted(
                rows,
                key=lambda row: -row["value"],
            )[:n],
    }


# ===========================================================================
# C. WHAT IT TOUCHES
# ===========================================================================

@tool("C")
def get_open_files(
    pid: Pid,
) -> dict:
    """
    Open files of a PID and file-descriptor information.
    """

    process = _proc(
        pid
    )

    files = [
        item.path
        for item in process.open_files()
    ]

    limit = _safe(
        lambda: process.rlimit(
            psutil.RLIMIT_NOFILE
        )
    )

    return {
        "pid":
            pid,

        "open_files":
            len(files),

        "num_fds":
            _safe(
                process.num_fds
            ),

        "fd_limit_soft":
            (
                limit[0]
                if limit
                else None
            ),

        "files":
            files[:30],
    }


@tool("C")
def get_network_connections(
    pid: Pid,
) -> dict:
    """
    Network sockets and remote hosts a PID is talking to.
    """

    process = _proc(
        pid
    )

    fn = (
        getattr(
            process,
            "net_connections",
            None,
        )
        or process.connections
    )

    rows = []

    for connection in fn(
        kind="inet"
    ):

        rows.append(
            {
                "status":
                    connection.status,

                "local":
                    (
                        f"{connection.laddr.ip}:"
                        f"{connection.laddr.port}"
                        if connection.laddr
                        else None
                    ),

                "remote":
                    (
                        f"{connection.raddr.ip}:"
                        f"{connection.raddr.port}"
                        if connection.raddr
                        else None
                    ),
            }
        )

    return {
        "pid":
            pid,

        "count":
            len(rows),

        "established":
            sum(
                1
                for row in rows
                if row["status"]
                == "ESTABLISHED"
            ),

        "connections":
            rows[:30],
    }


@tool("C")
def get_listening_ports() -> dict:
    """
    What is listening on this machine.
    """

    try:

        connections = psutil.net_connections(
            kind="inet"
        )

    except psutil.AccessDenied:

        raise ToolError(
            "ACCESS_DENIED",
            "listing all sockets needs elevated rights",
        )

    rows = []

    for connection in connections:

        if connection.status != "LISTEN":
            continue

        name = (
            _safe(
                lambda:
                    psutil.Process(
                        connection.pid
                    ).name()
            )
            if connection.pid
            else None
        )

        rows.append(
            {
                "port":
                    connection.laddr.port,

                "ip":
                    connection.laddr.ip,

                "pid":
                    connection.pid,

                "name":
                    name,
            }
        )

    return {
        "count":
            len(rows),

        "listening":
            sorted(
                rows,
                key=lambda row:
                    row["port"],
            )[:50],
    }


# ===========================================================================
# D. SYSTEM HEALTH
# ===========================================================================

@tool("D")
def get_system_overview() -> dict:
    """
    CPU, load, RAM, swap, uptime.
    """

    psutil.cpu_percent(
        None
    )

    time.sleep(
        0.3
    )

    virtual_memory = (
        psutil.virtual_memory()
    )

    swap = (
        psutil.swap_memory()
    )

    return {
        "cpu_pct":
            psutil.cpu_percent(
                None
            ),

        "cores":
            psutil.cpu_count(),

        "load_avg":
            [
                round(
                    value,
                    2,
                )
                for value in os.getloadavg()
            ],

        "mem_pct":
            virtual_memory.percent,

        "mem_used_mb":
            _mb(
                virtual_memory.used
            ),

        "mem_available_mb":
            _mb(
                virtual_memory.available
            ),

        "swap_pct":
            swap.percent,

        "uptime_min":
            int(
                (
                    time.time()
                    - psutil.boot_time()
                )
                / 60
            ),

        "process_count":
            len(
                psutil.pids()
            ),
    }


@tool("D")
def get_disk_usage() -> dict:
    """
    Free space on every real mount.
    """

    rows = []

    for partition in psutil.disk_partitions(
        all=False
    ):

        if (
            partition.fstype
            in (
                "squashfs",
                "tmpfs",
            )
            or partition.mountpoint.startswith(
                "/snap"
            )
        ):
            continue

        usage = _safe(
            lambda:
                psutil.disk_usage(
                    partition.mountpoint
                )
        )

        if usage:

            rows.append(
                {
                    "mount":
                        partition.mountpoint,

                    "device":
                        partition.device,

                    "fstype":
                        partition.fstype,

                    "total_gb":
                        round(
                            usage.total
                            / 2**30,
                            1,
                        ),

                    "free_gb":
                        round(
                            usage.free
                            / 2**30,
                            1,
                        ),

                    "used_pct":
                        usage.percent,
                }
            )

    return {
        "disks":
            rows,
    }


@tool("D")
def get_temperature() -> dict:
    """
    Temperatures and CPU frequency.
    """

    temperatures = _safe(
        psutil.sensors_temperatures,
        {},
    )

    if not temperatures:

        raise ToolError(
            "UNAVAILABLE",
            "no temperature sensors found",
        )

    chips = {}

    for name, sensor_list in temperatures.items():

        chips[name] = [
            {
                "label":
                    sensor.label or name,

                "c":
                    sensor.current,

                "high":
                    sensor.high,

                "critical":
                    sensor.critical,
            }

            for sensor in sensor_list
        ]

    frequency = _safe(
        psutil.cpu_freq
    )

    all_temperatures = [
        item["c"]
        for sensor_list in chips.values()
        for item in sensor_list
    ]

    return {
        "hottest_c":
            max(
                all_temperatures
            ),

        "cpu_freq_mhz":
            (
                round(
                    frequency.current
                )
                if frequency
                else None
            ),

        "cpu_freq_max_mhz":
            (
                round(
                    frequency.max
                )
                if frequency
                and frequency.max
                else None
            ),

        "sensors":
            chips,
    }


@tool("D")
def get_battery_power() -> dict:
    """
    Battery level, charging state and power profile.
    """

    battery = _safe(
        psutil.sensors_battery
    )

    output = {
        "battery_present":
            battery is not None,
    }

    if battery:

        unknown = (
            psutil.POWER_TIME_UNLIMITED,
            psutil.POWER_TIME_UNKNOWN,
        )

        output.update(
            {
                "percent":
                    round(
                        battery.percent
                    ),

                "plugged_in":
                    battery.power_plugged,

                "minutes_left":
                    (
                        None
                        if battery.secsleft
                        in unknown
                        else
                        battery.secsleft // 60
                    ),
            }
        )

    try:

        output["power_profile"] = (
            _run(
                [
                    "powerprofilesctl",
                    "get",
                ]
            ).strip()
        )

    except ToolError:

        output[
            "power_profile"
        ] = None

    return output


@tool("D")
def get_gpu_status() -> dict:
    """
    GPU load and memory for NVIDIA,
    or detected GPU hardware.
    """

    try:

        output = _run(
            [
                "nvidia-smi",

                "--query-gpu="
                "name,"
                "utilization.gpu,"
                "memory.used,"
                "memory.total,"
                "temperature.gpu",

                "--format=csv,noheader,nounits",
            ]
        )

        gpus = []

        for line in output.strip().splitlines():

            (
                name,
                utilization,
                memory_used,
                memory_total,
                temperature,
            ) = [
                item.strip()
                for item in line.split(",")
            ]

            gpus.append(
                {
                    "name":
                        name,

                    "util_pct":
                        float(
                            utilization
                        ),

                    "mem_used_mb":
                        float(
                            memory_used
                        ),

                    "mem_total_mb":
                        float(
                            memory_total
                        ),

                    "temp_c":
                        float(
                            temperature
                        ),
                }
            )

        return {
            "vendor":
                "nvidia",

            "gpus":
                gpus,
        }

    except ToolError:

        pass

    try:

        cards = [
            line
            for line in _run(
                [
                    "lspci"
                ]
            ).splitlines()
            if re.search(
                r"VGA|3D|Display",
                line,
            )
        ]

    except ToolError:

        raise ToolError(
            "UNAVAILABLE",
            "no GPU information available",
        )

    return {
        "vendor":
            "unknown",

        "gpus_detected":
            cards[:4],

        "note":
            "live GPU load is only available for NVIDIA",
    }


# ===========================================================================
# E. LOGS & EVENTS
# ===========================================================================

_SEV = re.compile(
    r"\b("
    r"error|"
    r"fail(ed|ure)?|"
    r"crit(ical)?|"
    r"warn(ing)?|"
    r"oom|"
    r"out of memory|"
    r"segfault|"
    r"denied|"
    r"timeout|"
    r"xrun|"
    r"underrun"
    r")\b",
    re.I,
)

_KERNEL = re.compile(
    r"oom|"
    r"out of memory|"
    r"killed process|"
    r"segfault|"
    r"call trace|"
    r"thermal|"
    r"throttl|"
    r"i/o error|"
    r"failed|"
    r"firmware|"
    r"gpu hang|"
    r"usb .* disconnect",
    re.I,
)


@tool("E")
def get_logs(
    service: Name,
    minutes: Minutes = 30,
    max_lines: Lines = 40,
) -> dict:
    """
    Recent journal lines mentioning a service.

    IMPORTANT:
        minutes is limited to 240 minutes.
    """

    lines = _run(
        [
            "journalctl",
            "--no-pager",
            "-o",
            "short",
            "--since",
            f"{minutes} min ago",
        ],
        8,
    ).splitlines()

    search = service.lower()

    matched = [
        line
        for line in lines
        if search in line.lower()
    ]

    severity_lines = [
        line
        for line in matched
        if _SEV.search(line)
    ]

    selected = (
        severity_lines
        or matched
    )

    return {
        "service":
            service,

        "window_min":
            minutes,

        "matched_lines":
            len(matched),

        "error_like_lines":
            len(severity_lines),

        "lines":
            [
                line[:240]
                for line
                in selected[-max_lines:]
            ],
    }


@tool("E")
def get_kernel_events(
    minutes: Minutes = 60,
    max_lines: Lines = 40,
) -> dict:
    """
    Kernel messages: OOM kills, segfaults,
    driver errors and thermal events.

    IMPORTANT:
        minutes is limited to 240 minutes.
    """

    lines = _run(
        [
            "journalctl",
            "-k",
            "--no-pager",
            "-o",
            "short",
            "--since",
            f"{minutes} min ago",
        ],
        8,
    ).splitlines()

    hits = [
        line
        for line in lines
        if _KERNEL.search(line)
    ]

    return {
        "window_min":
            minutes,

        "matched_lines":
            len(hits),

        "lines":
            [
                line[:240]
                for line
                in hits[-max_lines:]
            ],
    }


@tool("E")
def get_crash_reports(
    app: Optional[Name] = None,
) -> dict:
    """
    Recent crash reports.
    """

    rows = []

    for file_path in glob.glob(
        "/var/crash/*.crash"
    ):

        if (
            app
            and app.lower()
            not in file_path.lower()
        ):
            continue

        rows.append(
            {
                "file":
                    file_path,

                "modified":
                    time.strftime(
                        "%Y-%m-%d %H:%M",
                        time.localtime(
                            _safe(
                                lambda:
                                    os.path.getmtime(
                                        file_path
                                    ),
                                0,
                            )
                        ),
                    ),
            }
        )

    try:

        core = _run(
            [
                "coredumpctl",
                "list",
                "--no-pager",
                "--no-legend",
            ],
            5,
        ).strip().splitlines()[-10:]

    except ToolError:

        core = []

    if app:

        core = [
            line
            for line in core
            if app.lower()
            in line.lower()
        ]

    return {
        "apport_crashes":
            sorted(
                rows,
                key=lambda row:
                    row["modified"],
            )[-10:],

        "coredumps":
            core,
    }


@tool("E")
def get_service_status(
    unit: Unit,
) -> dict:
    """
    systemd unit state.

    Tries user units first, then system units.
    """

    properties = (
        "ActiveState,"
        "SubState,"
        "LoadState,"
        "MainPID,"
        "NRestarts,"
        "Result,"
        "ExecMainStatus"
    )

    for scope in (
        ["--user"],
        [],
    ):

        try:

            output = _run(
                [
                    "systemctl",
                    *scope,
                    "show",
                    unit,
                    f"--property={properties}",
                ],
                5,
            )

        except ToolError:

            continue

        data = dict(
            line.split(
                "=",
                1,
            )
            for line
            in output.splitlines()
            if "=" in line
        )

        if (
            data.get(
                "LoadState"
            )
            != "not-found"
        ):

            return {
                "unit":
                    unit,

                "scope":
                    (
                        "user"
                        if scope
                        else "system"
                    ),

                **data,
            }

    raise ToolError(
        "NOT_FOUND",
        f"no systemd unit named {unit!r}",
    )


# ===========================================================================
# F. APP INFO
# ===========================================================================

@tool("F")
def get_app_info(
    app: Name,
) -> dict:
    """
    Where an app lives and how it was installed.
    """

    output = {
        "app":
            app,

        "binary":
            (
                shutil.which(
                    app.lower()
                )
                or shutil.which(
                    app
                )
            ),

        "installed_via":
            {},
    }

    try:

        version = _run(
            [
                "dpkg-query",
                "-W",
                "-f=${Version}",
                app.lower(),
            ],
            4,
        ).strip()

        if version:

            output[
                "installed_via"
            ]["apt"] = version

    except ToolError:

        pass

    try:

        for line in _run(
            [
                "snap",
                "list",
            ],
            5,
        ).splitlines()[1:]:

            parts = line.split()

            if (
                parts
                and parts[0].lower()
                == app.lower()
            ):

                output[
                    "installed_via"
                ]["snap"] = parts[1]

    except ToolError:

        pass

    try:

        for line in _run(
            [
                "flatpak",
                "list",
                "--app",
                "--columns=application,version",
            ],
            6,
        ).splitlines():

            if app.lower() in line.lower():

                output[
                    "installed_via"
                ]["flatpak"] = line.strip()

    except ToolError:

        pass

    return output


@tool("F")
def get_cache_size(
    app: Name,
) -> dict:
    """
    Size of an app's cache, config and data folders.
    """

    home = Path.home()

    candidates = [
        home / ".cache" / app,
        home / ".config" / app,
        home / ".local/share" / app,
        home / "snap" / app.lower(),
    ]

    candidates += list(
        (
            home
            / ".var/app"
        ).glob(
            f"*{app}*"
        )
    )

    rows = [
        {
            "path":
                str(
                    path
                ).replace(
                    str(home),
                    "~",
                ),

            "size_mb":
                _dir_mb(
                    path
                ),
        }

        for path
        in candidates
        if path.is_dir()
    ]

    if not rows:

        raise ToolError(
            "NOT_FOUND",
            f"no cache/config folders found for {app!r}",
        )

    return {
        "app":
            app,

        "folders":
            rows,
    }


@tool("F")
def get_startup_apps() -> dict:
    """
    Apps that start automatically at login.
    """

    rows = []

    directories = (
        Path.home()
        / ".config/autostart",

        Path(
            "/etc/xdg/autostart"
        ),
    )

    for directory in directories:

        if not directory.is_dir():
            continue

        for file_path in sorted(
            directory.glob(
                "*.desktop"
            )
        ):

            text = _safe(
                lambda:
                    file_path.read_text(
                        errors="ignore"
                    ),
                "",
            )

            name = re.search(
                r"^Name=(.+)$",
                text,
                re.M,
            )

            disabled = re.search(
                r"^(Hidden=true|"
                r"X-GNOME-Autostart-enabled=false)",
                text,
                re.M,
            )

            rows.append(
                {
                    "name":
                        (
                            name.group(1)
                            if name
                            else file_path.stem
                        ),

                    "file":
                        file_path.name,

                    "enabled":
                        disabled is None,

                    "source":
                        (
                            "user"
                            if "home"
                            in str(directory)
                            else "system"
                        ),
                }
            )

    return {
        "count":
            len(rows),

        "apps":
            rows[:60],
    }


# ===========================================================================
# G. AUDIO
# ===========================================================================

@tool("G")
def get_audio_status() -> dict:
    """
    Default output device, volume and mute state.
    """

    sink = _run(
        [
            "pactl",
            "get-default-sink",
        ]
    ).strip()

    volume = _run(
        [
            "pactl",
            "get-sink-volume",
            "@DEFAULT_SINK@",
        ]
    )

    mute = _run(
        [
            "pactl",
            "get-sink-mute",
            "@DEFAULT_SINK@",
        ]
    )

    percentages = [
        int(value)
        for value
        in re.findall(
            r"(\d+)%",
            volume,
        )
    ]

    return {
        "default_sink":
            sink,

        "volume_pct":
            (
                max(percentages)
                if percentages
                else None
            ),

        "channels_pct":
            percentages,

        "muted":
            "yes"
            in mute.lower(),
    }


@tool("G")
def list_audio_streams() -> dict:
    """
    Which apps are playing sound and at what volume.
    """

    output = _run(
        [
            "pactl",
            "list",
            "sink-inputs",
        ]
    )

    streams = []

    blocks = re.split(
        r"\n(?=Sink Input #)",
        output.strip(),
    )

    for block in blocks:

        if not block.startswith(
            "Sink Input"
        ):
            continue

        app = re.search(
            r'application\.name = "(.*)"',
            block,
        )

        pid = re.search(
            r'application\.process\.id = "(\d+)"',
            block,
        )

        volume = re.search(
            r"^\s*Volume:(.*)$",
            block,
            re.M,
        )

        percentages = (
            re.findall(
                r"(\d+)%",
                volume.group(1),
            )
            if volume
            else []
        )

        mute = re.search(
            r"Mute: (\w+)",
            block,
        )

        streams.append(
            {
                "app":
                    (
                        app.group(1)
                        if app
                        else None
                    ),

                "pid":
                    (
                        int(
                            pid.group(1)
                        )
                        if pid
                        else None
                    ),

                "volume_pct":
                    (
                        int(
                            percentages[0]
                        )
                        if percentages
                        else None
                    ),

                "muted":
                    bool(
                        mute
                        and mute.group(1)
                        == "yes"
                    ),
            }
        )

    return {
        "count":
            len(streams),

        "streams":
            streams,
    }


# ===========================================================================
# H. NETWORK
# ===========================================================================

@tool("H")
def get_network_status() -> dict:
    """
    Active interfaces, Wi-Fi signal,
    DNS servers and error counters.
    """

    stats = (
        psutil.net_if_stats()
    )

    addresses = (
        psutil.net_if_addrs()
    )

    interfaces = []

    for name, status in stats.items():

        if (
            name == "lo"
            or not status.isup
        ):
            continue

        ipv4 = next(
            (
                address.address
                for address
                in addresses.get(
                    name,
                    [],
                )
                if address.family
                == socket.AF_INET
            ),
            None,
        )

        interfaces.append(
            {
                "name":
                    name,

                "ip":
                    ipv4,

                "speed_mbps":
                    status.speed,
            }
        )

    wifi = None

    try:

        for line in _run(
            [
                "nmcli",
                "-t",
                "-f",
                "ACTIVE,SSID,SIGNAL",
                "dev",
                "wifi",
            ],
            5,
        ).splitlines():

            if line.startswith(
                "yes:"
            ):

                parts = line.split(
                    ":"
                )

                wifi = {
                    "ssid":
                        ":".join(
                            parts[1:-1]
                        ),

                    "signal_pct":
                        int(
                            parts[-1]
                        ),
                }

    except (
        ToolError,
        ValueError,
    ):

        pass

    dns = [
        line.split()[1]
        for line
        in _safe(
            lambda:
                Path(
                    "/etc/resolv.conf"
                ).read_text().splitlines(),
            [],
        )
        if line.startswith(
            "nameserver"
        )
    ]

    io = (
        psutil.net_io_counters()
    )

    return {
        "interfaces":
            interfaces,

        "wifi":
            wifi,

        "dns_servers":
            dns,

        "errors":
            {
                "in":
                    io.errin,

                "out":
                    io.errout,

                "dropped_in":
                    io.dropin,

                "dropped_out":
                    io.dropout,
            },
    }


@tool("H")
def ping_test(
    host: Host,

    count: Annotated[
        int,
        Field(
            ge=1,
            le=4,
        ),
    ] = 3,
) -> dict:
    """
    Latency and packet loss to a host.
    """

    output = _run(
        [
            "ping",
            "-c",
            str(count),
            "-W",
            "2",
            host,
        ],
        timeout=10,
    )

    loss = re.search(
        r"(\d+(?:\.\d+)?)% packet loss",
        output,
    )

    rtt = re.search(
        r"= ([\d.]+)/([\d.]+)/([\d.]+)",
        output,
    )

    return {
        "host":
            host,

        "packet_loss_pct":
            (
                float(
                    loss.group(1)
                )
                if loss
                else None
            ),

        "rtt_min_ms":
            (
                float(
                    rtt.group(1)
                )
                if rtt
                else None
            ),

        "rtt_avg_ms":
            (
                float(
                    rtt.group(2)
                )
                if rtt
                else None
            ),

        "rtt_max_ms":
            (
                float(
                    rtt.group(3)
                )
                if rtt
                else None
            ),
    }


# ===========================================================================
# SELF CHECK
# ===========================================================================

if __name__ == "__main__":

    print(
        f"{len(TOOLS)} read-only tools registered\n"
    )

    for group in sorted(
        {
            meta["group"]
            for meta in TOOLS.values()
        }
    ):

        for name, meta in TOOLS.items():

            if meta["group"] == group:

                print(
                    f"  {group}  READ_ONLY  "
                    f"{name:24} "
                    f"{meta['doc']}"
                )