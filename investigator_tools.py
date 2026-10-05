"""
Investigator tools for AGENTIC_AI_LAPTOP.

Purpose:
    Read-only investigation of the local Ubuntu/Linux machine.

Architecture:
    Investigator discovers, inspects, correlates and reports.
    It does NOT perform mutating or privileged actions.

Registry format:
    TOOLS[name] = {
        "fn": wrapped_function,
        "group": "...",
        "mutating": False,
        "doc": "...",
    }

Used by:
    tool_server.py investigator
"""

from __future__ import annotations
from ddgs import DDGS
import functools
import glob
import json
import os
import re
import shlex
import shutil
import socket
import stat
import subprocess
import time
from pathlib import Path
from typing import Annotated, Literal, Optional

import psutil
from pydantic import Field, validate_call


# ============================================================
# REGISTRY
# ============================================================

TOOLS = {}


# ============================================================
# ERRORS
# ============================================================

class ToolError(Exception):
    """
    Structured tool error understood by tool_server.py.
    """

    def __init__(
        self,
        code: str,
        message: str,
        retryable: bool = False,
    ):
        super().__init__(message)

        self.code = code
        self.message = message
        self.retryable = retryable


# ============================================================
# TOOL DECORATOR
# ============================================================

def tool(
    group: str,
    mutating: bool = False,
):
    """
    Register a function as an Investigator tool.
    """

    def decorator(fn):

        wrapped = validate_call(
            fn
        )

        wrapped = functools.wraps(fn)(
            wrapped
        )

        TOOLS[fn.__name__] = {
            "fn": wrapped,
            "group": group,
            "mutating": mutating,
            "doc": (fn.__doc__ or "").strip(),
        }

        return wrapped

    return decorator


# ============================================================
# CONSTANTS / LIMITS
# ============================================================

MAX_READ_BYTES = 256 * 1024
MAX_SEARCH_RESULTS = 200
MAX_COMMAND_OUTPUT = 64 * 1024
MAX_FIND_RESULTS = 200

# Directories that are commonly useful during investigation.
SAFE_SEARCH_ROOTS = {
    "/tmp",
    "/var/tmp",
    "/var/log",
    "/etc",
    "/opt",
    "/usr/local",
    "/home",
}

# Commands that are useful for investigation.
READONLY_COMMANDS = {
    # Files
    "ls",
    "find",
    "file",
    "stat",
    "readlink",
    "realpath",
    "basename",
    "dirname",
    "tree",
    "du",
    "df",
    "cat",
    "head",
    "tail",
    "less",
    "more",
    "nl",
    "wc",
    "strings",
    "od",
    "xxd",
    "hexdump",

    # Text/search
    "grep",
    "egrep",
    "fgrep",
    "awk",
    "sed",
    "cut",
    "tr",
    "sort",
    "uniq",
    "diff",
    "cmp",

    # Processes
    "ps",
    "pgrep",
    "pidof",
    "pstree",
    "lsof",
    "fuser",
    "top",
    "free",
    "uptime",
    "vmstat",
    "iostat",
    "mpstat",
    "pidstat",

    # System
    "uname",
    "hostname",
    "lsblk",
    "blkid",
    "lscpu",
    "lsmem",
    "lsusb",
    "lspci",
    "lsmod",
    "lshw",
    "dmidecode",
    "inxi",
    "sensors",
    "uptime",

    # Network
    "ip",
    "ss",
    "ping",
    "tracepath",
    "traceroute",
    "dig",
    "nslookup",
    "host",
    "curl",
    "wget",
    "nmcli",
    "resolvectl",
    "ethtool",
    "arp",
    "route",

    # Services/logs
    "systemctl",
    "journalctl",
    "service",
    "loginctl",
    "systemd-analyze",
    "systemd-cgls",
    "systemd-cgtop",
    "dmesg",

    # Users/security
    "who",
    "w",
    "whoami",
    "id",
    "groups",
    "users",
    "last",
    "lastlog",
    "getfacl",
    "getcap",
    "capsh",

    # Packages
    "apt",
    "apt-cache",
    "apt-file",
    "dpkg",
    "dpkg-query",
    "snap",
    "flatpak",
    "pip",
    "pip3",
    "npm",
    "npx",

    # Storage
    "mount",
    "findmnt",
    "fdisk",
    "parted",
    "smartctl",
    "hdparm",

    # Binary analysis
    "readelf",
    "objdump",
    "nm",
    "ldd",
    "ldconfig",

    # Git
    "git",

    # Shell/environment
    "env",
    "printenv",
    "echo",
    "printenv",
    "which",
    "whereis",
    "type",
}

# Commands that are never allowed through the generic
# read-only command executor.
DANGEROUS_COMMANDS = {
    "rm",
    "rmdir",
    "cp",
    "mv",
    "mkdir",
    "touch",
    "ln",
    "chmod",
    "chown",
    "chgrp",
    "setfacl",
    "setcap",

    "kill",
    "killall",
    "pkill",
    "renice",

    "systemctl",
    "service",

    "apt-get",
    "aptitude",

    "mount",
    "umount",

    "fdisk",
    "parted",

    "iptables",
    "ip6tables",
    "nft",
    "ufw",

    "useradd",
    "usermod",
    "userdel",
    "groupadd",
    "groupmod",
    "groupdel",

    "passwd",
    "chpasswd",

    "sudo",
    "su",

    "ssh",
    "scp",
    "rsync",

    "git",
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def _run(
    command,
    args=None,
    timeout: int = 15,
):
    """
    Run a local command without invoking a shell.
    """

    args = args or []

    cmd = [
        command,
        *[
            str(x)
            for x in args
        ],
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )

    except FileNotFoundError:
        raise ToolError(
            "COMMAND_NOT_FOUND",
            f"command not found: {command}",
        )

    except subprocess.TimeoutExpired:
        raise ToolError(
            "TIMEOUT",
            f"command timed out after {timeout}s",
            retryable=True,
        )

    except PermissionError:
        raise ToolError(
            "PERMISSION_DENIED",
            f"permission denied executing: {command}",
        )

    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _truncate(
    text: str,
    maximum: int = MAX_COMMAND_OUTPUT,
):
    if len(text) <= maximum:
        return text

    return (
        text[:maximum]
        + "\n...[output truncated]..."
    )


def _read_text(
    path: str,
    max_bytes: int = MAX_READ_BYTES,
):
    """
    Safely read a bounded amount of text.
    """

    try:
        with open(
            path,
            "rb",
        ) as f:

            data = f.read(
                max_bytes
            )

    except FileNotFoundError:
        raise ToolError(
            "NOT_FOUND",
            f"path does not exist: {path}",
        )

    except PermissionError:
        raise ToolError(
            "PERMISSION_DENIED",
            f"permission denied: {path}",
        )

    except IsADirectoryError:
        raise ToolError(
            "NOT_A_FILE",
            f"path is a directory: {path}",
        )

    except OSError as exc:
        raise ToolError(
            "IO_ERROR",
            str(exc),
        )

    truncated = len(data) >= max_bytes

    return {
        "text": data.decode(
            "utf-8",
            errors="replace",
        ),
        "bytes_read": len(data),
        "truncated": truncated,
    }


def _validate_path(
    path: str,
):
    """
    Resolve a path without requiring it to exist.
    """

    if not path:
        raise ToolError(
            "INVALID_ARGUMENTS",
            "path cannot be empty",
        )

    try:
        return os.path.abspath(
            os.path.expanduser(path)
        )

    except Exception as exc:
        raise ToolError(
            "INVALID_PATH",
            str(exc),
        )


def _process_exists(pid: int):
    try:
        return psutil.Process(pid)

    except psutil.NoSuchProcess:
        raise ToolError(
            "PROCESS_NOT_FOUND",
            f"process {pid} does not exist",
        )

    except psutil.AccessDenied:
        raise ToolError(
            "PERMISSION_DENIED",
            f"access denied for process {pid}",
        )


def _command_basename(command: str):
    return os.path.basename(
        command.split()[0]
    )


# ============================================================
# GROUP A — FILESYSTEM
# ============================================================

@tool("filesystem")
def list_directory(
    path: Annotated[
        str,
        Field(description="Directory to inspect"),
    ] = ".",
    recursive: bool = False,
    max_entries: Annotated[
        int,
        Field(ge=1, le=500),
    ] = 100,
):
    """
    List files and directories with metadata.
    """

    path = _validate_path(path)

    if not os.path.isdir(path):
        raise ToolError(
            "NOT_A_DIRECTORY",
            f"not a directory: {path}",
        )

    results = []

    try:

        if recursive:

            for root, dirs, files in os.walk(path):

                for name in dirs + files:

                    if len(results) >= max_entries:
                        break

                    full = os.path.join(
                        root,
                        name,
                    )

                    try:
                        st = os.lstat(full)

                        results.append({
                            "path": full,
                            "name": name,
                            "type": (
                                "directory"
                                if stat.S_ISDIR(st.st_mode)
                                else "file"
                            ),
                            "size": st.st_size,
                            "mode": oct(
                                stat.S_IMODE(
                                    st.st_mode
                                )
                            ),
                        })

                    except OSError:
                        continue

                if len(results) >= max_entries:
                    break

        else:

            with os.scandir(path) as entries:

                for entry in entries:

                    if len(results) >= max_entries:
                        break

                    try:
                        st = entry.stat(
                            follow_symlinks=False
                        )

                        results.append({
                            "path": entry.path,
                            "name": entry.name,
                            "type": (
                                "directory"
                                if entry.is_dir(
                                    follow_symlinks=False
                                )
                                else "file"
                            ),
                            "size": st.st_size,
                            "mode": oct(
                                stat.S_IMODE(
                                    st.st_mode
                                )
                            ),
                        })

                    except OSError:
                        continue

    except PermissionError:
        raise ToolError(
            "PERMISSION_DENIED",
            f"permission denied: {path}",
        )

    return {
        "path": path,
        "recursive": recursive,
        "entries": results,
        "count": len(results),
    }


@tool("filesystem")
def find_files(
    root: str,
    pattern: str = "*",
    file_type: Literal[
        "any",
        "file",
        "directory",
        "symlink",
    ] = "any",
    max_results: Annotated[
        int,
        Field(ge=1, le=MAX_FIND_RESULTS),
    ] = 100,
):
    """
    Find files or directories below a root using a filename glob.
    """

    root = _validate_path(root)

    if not os.path.exists(root):
        raise ToolError(
            "NOT_FOUND",
            f"root does not exist: {root}",
        )

    results = []

    try:

        for current_root, dirs, files in os.walk(
            root,
            followlinks=False,
        ):

            candidates = []

            if file_type in (
                "any",
                "directory",
            ):
                candidates.extend(
                    os.path.join(
                        current_root,
                        d,
                    )
                    for d in dirs
                )

            if file_type in (
                "any",
                "file",
            ):
                candidates.extend(
                    os.path.join(
                        current_root,
                        f,
                    )
                    for f in files
                )

            if file_type == "symlink":
                candidates = [
                    os.path.join(
                        current_root,
                        x,
                    )
                    for x in dirs + files
                    if os.path.islink(
                        os.path.join(
                            current_root,
                            x,
                        )
                    )
                ]

            for path in candidates:

                if not fnmatch_name(
                    os.path.basename(path),
                    pattern,
                ):
                    continue

                results.append(path)

                if len(results) >= max_results:
                    return {
                        "root": root,
                        "pattern": pattern,
                        "file_type": file_type,
                        "results": results,
                        "count": len(results),
                        "truncated": True,
                    }

    except PermissionError:
        pass

    return {
        "root": root,
        "pattern": pattern,
        "file_type": file_type,
        "results": results,
        "count": len(results),
        "truncated": False,
    }


def fnmatch_name(
    name: str,
    pattern: str,
):
    """
    Lightweight glob matcher.
    """

    import fnmatch

    return fnmatch.fnmatch(
        name,
        pattern,
    )


@tool("filesystem")
def file_info(
    path: str,
):
    """
    Return detailed filesystem metadata for a path.
    """

    path = _validate_path(path)

    try:
        st = os.lstat(path)

    except FileNotFoundError:
        raise ToolError(
            "NOT_FOUND",
            f"path does not exist: {path}",
        )

    except PermissionError:
        raise ToolError(
            "PERMISSION_DENIED",
            f"permission denied: {path}",
        )

    mode = st.st_mode

    if stat.S_ISREG(mode):
        file_type = "file"
    elif stat.S_ISDIR(mode):
        file_type = "directory"
    elif stat.S_ISLNK(mode):
        file_type = "symlink"
    elif stat.S_ISSOCK(mode):
        file_type = "socket"
    elif stat.S_ISCHR(mode):
        file_type = "character_device"
    elif stat.S_ISBLK(mode):
        file_type = "block_device"
    elif stat.S_ISFIFO(mode):
        file_type = "fifo"
    else:
        file_type = "other"

    result = {
        "path": path,
        "type": file_type,
        "size": st.st_size,
        "mode": oct(
            stat.S_IMODE(mode)
        ),
        "uid": st.st_uid,
        "gid": st.st_gid,
        "inode": st.st_ino,
        "device": st.st_dev,
        "links": st.st_nlink,
        "mtime": st.st_mtime,
        "ctime": st.st_ctime,
        "atime": st.st_atime,
    }

    if stat.S_ISLNK(mode):
        try:
            result["target"] = os.readlink(
                path
            )
        except OSError:
            pass

    return result


@tool("filesystem")
def read_file(
    path: str,
    max_bytes: Annotated[
        int,
        Field(ge=1, le=MAX_READ_BYTES),
    ] = MAX_READ_BYTES,
):
    """
    Read a bounded amount of text from a file.
    """

    path = _validate_path(path)

    return {
        "path": path,
        **_read_text(
            path,
            max_bytes,
        ),
    }


@tool("filesystem")
def tail_file(
    path: str,
    lines: Annotated[
        int,
        Field(ge=1, le=1000),
    ] = 100,
):
    """
    Read the last lines of a text file.
    """

    path = _validate_path(path)

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
            errors="replace",
        ) as f:

            content = f.readlines()

    except FileNotFoundError:
        raise ToolError(
            "NOT_FOUND",
            f"file does not exist: {path}",
        )

    except PermissionError:
        raise ToolError(
            "PERMISSION_DENIED",
            f"permission denied: {path}",
        )

    return {
        "path": path,
        "lines_requested": lines,
        "content": "".join(
            content[-lines:]
        ),
    }


@tool("filesystem")
def search_text(
    pattern: str,
    root: str,
    max_results: Annotated[
        int,
        Field(ge=1, le=MAX_SEARCH_RESULTS),
    ] = 100,
    case_sensitive: bool = False,
):
    """
    Search text files recursively below a directory.
    """

    root = _validate_path(root)

    if not os.path.isdir(root):
        raise ToolError(
            "NOT_A_DIRECTORY",
            f"not a directory: {root}",
        )

    try:
        regex = re.compile(
            pattern,
            0 if case_sensitive else re.IGNORECASE,
        )

    except re.error as exc:
        raise ToolError(
            "INVALID_REGEX",
            str(exc),
        )

    results = []

    for current_root, dirs, files in os.walk(
        root,
        followlinks=False,
    ):

        for filename in files:

            if len(results) >= max_results:
                break

            path = os.path.join(
                current_root,
                filename,
            )

            try:

                if os.path.getsize(path) > MAX_READ_BYTES:
                    continue

                with open(
                    path,
                    "r",
                    encoding="utf-8",
                    errors="replace",
                ) as f:

                    for line_number, line in enumerate(
                        f,
                        start=1,
                    ):

                        if regex.search(line):

                            results.append({
                                "path": path,
                                "line": line_number,
                                "text": line.rstrip(
                                    "\n"
                                ),
                            })

                            if len(results) >= max_results:
                                break

            except (
                OSError,
                UnicodeError,
            ):
                continue

        if len(results) >= max_results:
            break

    return {
        "pattern": pattern,
        "root": root,
        "results": results,
        "count": len(results),
        "truncated": len(results) >= max_results,
    }


@tool("filesystem")
def directory_size(
    path: str,
):
    """
    Calculate the total size of files below a directory.
    """

    path = _validate_path(path)

    if not os.path.isdir(path):
        raise ToolError(
            "NOT_A_DIRECTORY",
            f"not a directory: {path}",
        )

    total = 0
    files = 0

    for root, dirs, filenames in os.walk(
        path,
        followlinks=False,
    ):

        for filename in filenames:

            full = os.path.join(
                root,
                filename,
            )

            try:
                total += os.path.getsize(
                    full
                )
                files += 1
            except OSError:
                continue

    return {
        "path": path,
        "bytes": total,
        "mb": round(
            total / (1024 * 1024),
            2,
        ),
        "files": files,
    }


@tool("filesystem")
def filesystem_usage():
    """
    Show disk usage for mounted filesystems.
    """

    results = []

    for partition in psutil.disk_partitions(
        all=False
    ):

        try:

            usage = psutil.disk_usage(
                partition.mountpoint
            )

            results.append({
                "device": partition.device,
                "mountpoint": partition.mountpoint,
                "filesystem": partition.fstype,
                "total_gb": round(
                    usage.total / (1024 ** 3),
                    2,
                ),
                "used_gb": round(
                    usage.used / (1024 ** 3),
                    2,
                ),
                "free_gb": round(
                    usage.free / (1024 ** 3),
                    2,
                ),
                "percent": usage.percent,
            })

        except (
            PermissionError,
            OSError,
        ):
            continue

    return results


# ============================================================
# GROUP B — PROCESS FORENSICS
# ============================================================

@tool("process")
def process_list(
    limit: Annotated[
        int,
        Field(ge=1, le=30),
    ] = 15,
):
    """
    List running processes with PID, user, CPU, memory and a
    TRUNCATED command line (200 chars max) to keep result size
    small and stay within token budgets.
    """

    processes = []

    for proc in psutil.process_iter(
        [
            "pid",
            "name",
            "username",
            "status",
            "cpu_percent",
            "memory_percent",
            "cmdline",
        ]
    ):

        try:

            info = proc.info

            raw_cmdline = info.get("cmdline") or []
            cmdline_str = " ".join(raw_cmdline)[:200]

            processes.append({
                "pid": info["pid"],
                "name": info["name"],
                "user": info["username"],
                "status": info["status"],
                "cpu_percent": info["cpu_percent"],
                "memory_percent": round(
                    info["memory_percent"] or 0,
                    2,
                ),
                "cmdline": cmdline_str,
            })

        except (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
        ):
            continue

    processes.sort(
        key=lambda x: (
            x["memory_percent"] or 0
        ),
        reverse=True,
    )

    return processes[:limit]


@tool("process")
def process_info(
    pid: int,
):
    """
    Detailed information about one running process.
    """

    proc = _process_exists(pid)

    try:

        with proc.oneshot():

            result = {
                "pid": proc.pid,
                "name": proc.name(),
                "exe": None,
                "cwd": None,
                "username": proc.username(),
                "status": proc.status(),
                "create_time": proc.create_time(),
                "ppid": proc.ppid(),
                "cmdline": proc.cmdline(),
                "num_threads": proc.num_threads(),
                "cpu_percent": proc.cpu_percent(
                    interval=0.1
                ),
                "memory": {
                    "rss": proc.memory_info().rss,
                    "vms": proc.memory_info().vms,
                },
            }

            try:
                result["exe"] = proc.exe()
            except (
                psutil.AccessDenied,
                psutil.NoSuchProcess,
            ):
                pass

            try:
                result["cwd"] = proc.cwd()
            except (
                psutil.AccessDenied,
                psutil.NoSuchProcess,
            ):
                pass

            return result

    except psutil.AccessDenied:
        raise ToolError(
            "PERMISSION_DENIED",
            f"permission denied for PID {pid}",
        )


@tool("process")
def process_tree(
    pid: int,
):
    """
    Return the parent chain and child processes for a PID.
    """

    proc = _process_exists(pid)

    parents = []

    try:
        current = proc

        while True:

            ppid = current.ppid()

            if ppid <= 0:
                break

            try:
                parent = psutil.Process(
                    ppid
                )

                parents.append({
                    "pid": parent.pid,
                    "name": parent.name(),
                })

                current = parent

            except (
                psutil.NoSuchProcess,
                psutil.AccessDenied,
            ):
                break

    except (
        psutil.NoSuchProcess,
        psutil.AccessDenied,
    ):
        pass

    children = []

    try:

        for child in proc.children(
            recursive=True
        ):

            try:
                children.append({
                    "pid": child.pid,
                    "name": child.name(),
                    "status": child.status(),
                    "cmdline": child.cmdline(),
                })

            except (
                psutil.NoSuchProcess,
                psutil.AccessDenied,
            ):
                continue

    except psutil.AccessDenied:
        raise ToolError(
            "PERMISSION_DENIED",
            f"cannot inspect children of PID {pid}",
        )

    return {
        "pid": pid,
        "parents": parents,
        "children": children,
    }


@tool("process")
def process_open_files(
    pid: int,
):
    """
    Show files and sockets opened by a process.
    """

    proc = _process_exists(pid)

    try:

        files = proc.open_files()

        return [
            {
                "path": item.path,
                "fd": item.fd,
            }
            for item in files
        ]

    except psutil.AccessDenied:
        raise ToolError(
            "PERMISSION_DENIED",
            f"cannot inspect open files for PID {pid}",
        )


@tool("process")
def process_connections(
    pid: int,
):
    """
    Show network connections associated with a process.
    """

    proc = _process_exists(pid)

    try:

        connections = proc.net_connections(
            kind="inet"
        )

    except psutil.AccessDenied:
        raise ToolError(
            "PERMISSION_DENIED",
            f"cannot inspect connections for PID {pid}",
        )

    results = []

    for conn in connections:

        results.append({
            "fd": conn.fd,
            "family": str(conn.family),
            "type": str(conn.type),
            "status": conn.status,
            "local": (
                list(conn.laddr)
                if conn.laddr
                else None
            ),
            "remote": (
                list(conn.raddr)
                if conn.raddr
                else None
            ),
        })

    return results


@tool("process")
def process_environment(
    pid: int,
):
    """
    Inspect environment variables exposed by a process.
    """

    proc = _process_exists(pid)

    path = f"/proc/{pid}/environ"

    try:

        with open(
            path,
            "rb",
        ) as f:

            data = f.read(
                MAX_READ_BYTES
            )

    except PermissionError:
        raise ToolError(
            "PERMISSION_DENIED",
            f"cannot read environment of PID {pid}",
        )

    except FileNotFoundError:
        raise ToolError(
            "PROCESS_NOT_FOUND",
            f"/proc entry disappeared for PID {pid}",
        )

    raw = data.decode(
        "utf-8",
        errors="replace",
    )

    environment = {}

    for item in raw.split("\x00"):

        if "=" not in item:
            continue

        key, value = item.split(
            "=",
            1,
        )

        environment[key] = value

    return {
        "pid": pid,
        "variables": environment,
    }


@tool("process")
def process_memory_maps(
    pid: int,
):
    """
    Inspect a process's memory mappings from /proc.
    """

    _process_exists(pid)

    path = f"/proc/{pid}/maps"

    try:

        content = _read_text(
            path,
            MAX_READ_BYTES,
        )

    except ToolError:
        raise

    mappings = []

    for line in content["text"].splitlines():

        parts = line.split(
            None,
            5,
        )

        if len(parts) < 5:
            continue

        mappings.append({
            "range": parts[0],
            "permissions": parts[1],
            "offset": parts[2],
            "device": parts[3],
            "inode": parts[4],
            "path": (
                parts[5]
                if len(parts) >= 6
                else None
            ),
        })

    return {
        "pid": pid,
        "mappings": mappings,
        "truncated": content["truncated"],
    }


@tool("process")
def process_fds(
    pid: int,
    max_entries: Annotated[
        int,
        Field(ge=1, le=500),
    ] = 200,
):
    """
    Inspect file descriptors exposed through /proc/PID/fd.
    """

    _process_exists(pid)

    directory = f"/proc/{pid}/fd"

    try:
        entries = os.listdir(
            directory
        )
    except PermissionError:
        raise ToolError(
            "PERMISSION_DENIED",
            f"cannot inspect file descriptors for PID {pid}",
        )
    except FileNotFoundError:
        raise ToolError(
            "PROCESS_NOT_FOUND",
            f"process {pid} no longer exists",
        )

    results = []

    for fd in entries[:max_entries]:

        path = os.path.join(
            directory,
            fd,
        )

        try:
            target = os.readlink(
                path
            )
        except OSError:
            target = None

        results.append({
            "fd": fd,
            "target": target,
        })

    return {
        "pid": pid,
        "fds": results,
        "count": len(results),
    }


# ============================================================
# GROUP C — NETWORK
# ============================================================

@tool("network")
def network_interfaces():
    """
    Show network interfaces, addresses and link state.
    """

    results = []

    addresses = psutil.net_if_addrs()
    stats = psutil.net_if_stats()

    for name, addr_list in addresses.items():

        info = {
            "interface": name,
            "up": (
                stats[name].isup
                if name in stats
                else None
            ),
            "speed_mbps": (
                stats[name].speed
                if name in stats
                else None
            ),
            "addresses": [],
        }

        for addr in addr_list:

            info["addresses"].append({
                "family": str(
                    addr.family
                ),
                "address": addr.address,
                "netmask": addr.netmask,
                "broadcast": addr.broadcast,
            })

        results.append(info)

    return results


@tool("network")
def network_routes():
    """
    Show the system routing table.
    """

    result = _run(
        "ip",
        ["route"],
    )

    return {
        "returncode": result["returncode"],
        "routes": result["stdout"],
        "stderr": result["stderr"],
    }


@tool("network")
def network_connections(
    kind: Literal[
        "inet",
        "inet4",
        "inet6",
        "tcp",
        "udp",
        "all",
    ] = "inet",
):
    """
    Show active network sockets and remote endpoints.
    """

    kind_map = {
        "inet": "inet",
        "inet4": "inet4",
        "inet6": "inet6",
        "tcp": "tcp",
        "udp": "udp",
        "all": "all",
    }

    try:
        connections = psutil.net_connections(
            kind=kind_map[kind]
        )
    except psutil.AccessDenied:
        raise ToolError(
            "PERMISSION_DENIED",
            "network connection inspection requires additional permission",
        )

    results = []

    for conn in connections:

        results.append({
            "fd": conn.fd,
            "family": str(conn.family),
            "type": str(conn.type),
            "status": conn.status,
            "local": (
                list(conn.laddr)
                if conn.laddr
                else None
            ),
            "remote": (
                list(conn.raddr)
                if conn.raddr
                else None
            ),
            "pid": conn.pid,
        })

    return results


@tool("network")
def listening_ports():
    """
    Show TCP and UDP sockets currently listening.
    """

    try:

        connections = psutil.net_connections(
            kind="inet"
        )

    except psutil.AccessDenied:
        raise ToolError(
            "PERMISSION_DENIED",
            "listening-port inspection requires additional permission",
        )

    results = []

    for conn in connections:

        if conn.status not in (
            psutil.CONN_LISTEN,
            psutil.CONN_NONE,
        ):
            continue

        results.append({
            "protocol": str(
                conn.type
            ),
            "local": (
                list(conn.laddr)
                if conn.laddr
                else None
            ),
            "pid": conn.pid,
        })

    return results


@tool("network")
def dns_lookup(
    host: str,
):
    """
    Resolve a hostname to IP addresses.
    """

    if not host.strip():
        raise ToolError(
            "INVALID_ARGUMENTS",
            "host cannot be empty",
        )

    try:

        results = socket.getaddrinfo(
            host,
            None,
        )

    except socket.gaierror as exc:
        raise ToolError(
            "DNS_ERROR",
            str(exc),
        )

    addresses = sorted({
        item[4][0]
        for item in results
        if item[4]
    })

    return {
        "host": host,
        "addresses": addresses,
    }


@tool("network")
def ping_host(
    host: str,
    count: Annotated[
        int,
        Field(ge=1, le=10),
    ] = 3,
):
    """
    Ping a host and return latency/loss information.
    """

    result = _run(
        "ping",
        [
            "-c",
            count,
            "-W",
            "2",
            host,
        ],
        timeout=max(
            10,
            count * 3,
        ),
    )

    return {
        "host": host,
        "returncode": result["returncode"],
        "output": _truncate(
            result["stdout"]
        ),
        "stderr": _truncate(
            result["stderr"]
        ),
    }


# ============================================================
# GROUP D — SYSTEM
# ============================================================

@tool("system")
def system_info():
    """
    Return CPU, memory, boot, platform and kernel information.
    """

    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()

    return {
        "platform": os.uname().sysname,
        "hostname": socket.gethostname(),
        "kernel": os.uname().release,
        "machine": os.uname().machine,
        "cpu_count_logical": psutil.cpu_count(
            logical=True
        ),
        "cpu_count_physical": psutil.cpu_count(
            logical=False
        ),
        "cpu_percent": psutil.cpu_percent(
            interval=0.2
        ),
        "memory": {
            "total": vm.total,
            "available": vm.available,
            "used": vm.used,
            "percent": vm.percent,
        },
        "swap": {
            "total": swap.total,
            "used": swap.used,
            "free": swap.free,
            "percent": swap.percent,
        },
        "boot_time": psutil.boot_time(),
        "uptime_seconds": max(
            0,
            time.time()
            - psutil.boot_time(),
        ),
    }


@tool("system")
def kernel_info():
    """
    Return Linux kernel and architecture information.
    """

    uname = os.uname()

    return {
        "sysname": uname.sysname,
        "nodename": uname.nodename,
        "release": uname.release,
        "version": uname.version,
        "machine": uname.machine,
    }


@tool("system")
def kernel_logs(
    lines: Annotated[
        int,
        Field(ge=1, le=500),
    ] = 100,
):
    """
    Read recent kernel messages without modifying the system.
    """

    result = _run(
        "dmesg",
        [
            "--color=never",
            "--time-format=iso",
            "-T",
            "-l",
            "emerg,alert,crit,err,warn,notice",
            "-n",
            str(lines),
        ],
    )

    return {
        "returncode": result["returncode"],
        "output": _truncate(
            result["stdout"]
        ),
        "stderr": _truncate(
            result["stderr"]
        ),
    }


@tool("system")
def system_logs(
    minutes: Annotated[
        int,
        Field(ge=1, le=1440),
    ] = 60,
    lines: Annotated[
        int,
        Field(ge=1, le=500),
    ] = 100,
):
    """
    Read recent systemd journal entries.
    """

    result = _run(
        "journalctl",
        [
            "--no-pager",
            "--output=short-iso",
            "--since",
            f"{minutes} minutes ago",
            "-n",
            str(lines),
        ],
    )

    return {
        "minutes": minutes,
        "returncode": result["returncode"],
        "output": _truncate(
            result["stdout"]
        ),
        "stderr": _truncate(
            result["stderr"]
        ),
    }


@tool("system")
def service_status(
    unit: str,
):
    """
    Inspect the current state of a systemd service/unit.
    """

    result = _run(
        "systemctl",
        [
            "show",
            unit,
            "--no-pager",
            "--property=Id,Description,LoadState,ActiveState,SubState,UnitFileState,MainPID,ExecMainStartTimestamp",
        ],
    )

    return {
        "unit": unit,
        "returncode": result["returncode"],
        "output": result["stdout"],
        "stderr": result["stderr"],
    }


@tool("system")
def service_logs(
    unit: str,
    minutes: Annotated[
        int,
        Field(ge=1, le=240),
    ] = 30,
    lines: Annotated[
        int,
        Field(ge=1, le=500),
    ] = 100,
):
    """
    Read recent journal entries for a systemd service.
    """

    result = _run(
        "journalctl",
        [
            "--no-pager",
            "--output=short-iso",
            "-u",
            unit,
            "--since",
            f"{minutes} minutes ago",
            "-n",
            str(lines),
        ],
    )

    return {
        "unit": unit,
        "minutes": minutes,
        "returncode": result["returncode"],
        "output": _truncate(
            result["stdout"]
        ),
        "stderr": _truncate(
            result["stderr"]
        ),
    }


@tool("system")
def systemd_timers():
    """
    List systemd timers and their current schedules.
    """

    result = _run(
        "systemctl",
        [
            "list-timers",
            "--all",
            "--no-pager",
        ],
    )

    return {
        "returncode": result["returncode"],
        "output": _truncate(
            result["stdout"]
        ),
        "stderr": _truncate(
            result["stderr"]
        ),
    }


# ============================================================
# GROUP E — USERS / SECURITY
# ============================================================

@tool("security")
def current_user():
    """
    Return the current user and identity information.
    """

    result = _run(
        "id"
    )

    return {
        "uid": os.getuid(),
        "gid": os.getgid(),
        "username": (
            os.getenv("USER")
            or os.getenv("USERNAME")
        ),
        "id_output": result["stdout"].strip(),
    }


@tool("security")
def user_info(
    username: str,
):
    """
    Inspect a local user's UID, groups and account metadata.
    """

    try:
        import pwd

        entry = pwd.getpwnam(
            username
        )

    except KeyError:
        raise ToolError(
            "USER_NOT_FOUND",
            f"user not found: {username}",
        )

    groups = []

    try:

        import grp

        for group in grp.getgrall():

            if username in group.gr_mem:

                groups.append(
                    group.gr_name
                )

    except Exception:
        pass

    return {
        "username": entry.pw_name,
        "uid": entry.pw_uid,
        "gid": entry.pw_gid,
        "home": entry.pw_dir,
        "shell": entry.pw_shell,
        "groups": sorted(
            set(groups)
        ),
    }


@tool("security")
def logged_in_users():
    """
    Show users currently logged into the system.
    """

    results = []

    for user in psutil.users():

        results.append({
            "name": user.name,
            "terminal": user.terminal,
            "host": user.host,
            "started": user.started,
            "pid": user.pid,
        })

    return results


@tool("security")
def sudo_configuration():
    """
    Inspect sudo configuration files without changing them.
    """

    paths = [
        "/etc/sudoers",
        "/etc/sudoers.d",
    ]

    results = []

    for path in paths:

        if not os.path.exists(path):
            continue

        if os.path.isfile(path):

            try:
                data = _read_text(
                    path,
                    MAX_READ_BYTES,
                )

                results.append({
                    "path": path,
                    "content": data["text"],
                })

            except ToolError as exc:

                results.append({
                    "path": path,
                    "error": exc.message,
                })

        elif os.path.isdir(path):

            try:

                for entry in os.listdir(
                    path
                ):

                    full = os.path.join(
                        path,
                        entry,
                    )

                    if not os.path.isfile(
                        full
                    ):
                        continue

                    try:

                        data = _read_text(
                            full,
                            MAX_READ_BYTES,
                        )

                        results.append({
                            "path": full,
                            "content": data["text"],
                        })

                    except ToolError as exc:

                        results.append({
                            "path": full,
                            "error": exc.message,
                        })

            except PermissionError:

                results.append({
                    "path": path,
                    "error": "permission denied",
                })

    return results


@tool("security")
def file_permissions(
    path: str,
):
    """
    Inspect ownership, permissions, ACL-related metadata and capabilities.
    """

    path = _validate_path(path)

    info = file_info(
        path
    )

    result = {
        "file": info,
    }

    getfacl = shutil.which(
        "getfacl"
    )

    if getfacl:

        acl = _run(
            getfacl,
            [
                "--absolute-names",
                path,
            ],
        )

        result["acl"] = {
            "returncode": acl["returncode"],
            "output": _truncate(
                acl["stdout"]
            ),
            "stderr": _truncate(
                acl["stderr"]
            ),
        }

    getcap = shutil.which(
        "getcap"
    )

    if getcap:

        cap = _run(
            getcap,
            [
                path,
            ],
        )

        result["capabilities"] = {
            "returncode": cap["returncode"],
            "output": cap["stdout"].strip(),
            "stderr": cap["stderr"].strip(),
        }

    return result


@tool("security")
def find_suid_sgid(
    root: str = "/",
    max_results: Annotated[
        int,
        Field(ge=1, le=500),
    ] = 200,
):
    """
    Find SUID/SGID files below a directory.
    """

    root = _validate_path(root)

    if not os.path.exists(root):
        raise ToolError(
            "NOT_FOUND",
            f"root does not exist: {root}",
        )

    results = []

    for current_root, dirs, files in os.walk(
        root,
        followlinks=False,
    ):

        for filename in files:

            if len(results) >= max_results:
                break

            path = os.path.join(
                current_root,
                filename,
            )

            try:

                mode = os.lstat(
                    path
                ).st_mode

                flags = []

                if mode & stat.S_ISUID:
                    flags.append(
                        "SUID"
                    )

                if mode & stat.S_ISGID:
                    flags.append(
                        "SGID"
                    )

                if flags:

                    results.append({
                        "path": path,
                        "flags": flags,
                        "mode": oct(
                            stat.S_IMODE(
                                mode
                            )
                        ),
                    })

            except OSError:
                continue

        if len(results) >= max_results:
            break

    return {
        "root": root,
        "results": results,
        "count": len(results),
        "truncated": len(results) >= max_results,
    }


# ============================================================
# GROUP F — PACKAGES / SOFTWARE
# ============================================================

@tool("software")
def installed_packages(
    manager: Literal[
        "dpkg",
        "snap",
        "flatpak",
        "pip",
        "npm",
    ] = "dpkg",
    limit: Annotated[
        int,
        Field(ge=1, le=500),
    ] = 100,
):
    """
    List installed software using a selected package manager.
    """

    if manager == "dpkg":

        result = _run(
            "dpkg-query",
            [
                "-W",
                "-f=${binary:Package}\t${Version}\n",
            ],
        )

    elif manager == "snap":

        result = _run(
            "snap",
            [
                "list",
            ],
        )

    elif manager == "flatpak":

        result = _run(
            "flatpak",
            [
                "list",
            ],
        )

    elif manager == "pip":

        result = _run(
            "python3",
            [
                "-m",
                "pip",
                "list",
                "--format=json",
            ],
        )

    elif manager == "npm":

        result = _run(
            "npm",
            [
                "list",
                "-g",
                "--depth=0",
                "--json",
            ],
        )

    else:
        raise ToolError(
            "INVALID_ARGUMENTS",
            f"unsupported package manager: {manager}",
        )

    output = result["stdout"]

    if manager in (
        "pip",
        "npm",
    ):

        try:
            parsed = json.loads(
                output
            )

            if isinstance(parsed, list):
                parsed = parsed[:limit]

            return {
                "manager": manager,
                "returncode": result["returncode"],
                "packages": parsed,
                "stderr": result["stderr"],
            }

        except json.JSONDecodeError:
            pass

    lines = [
        line
        for line in output.splitlines()
        if line.strip()
    ]

    return {
        "manager": manager,
        "returncode": result["returncode"],
        "packages": lines[:limit],
        "count": min(
            len(lines),
            limit,
        ),
        "truncated": len(lines) > limit,
        "stderr": result["stderr"],
    }


@tool("software")
def package_info(
    package: str,
):
    """
    Inspect package metadata using dpkg.
    """

    result = _run(
        "dpkg-query",
        [
            "-s",
            package,
        ],
    )

    return {
        "package": package,
        "returncode": result["returncode"],
        "metadata": _truncate(
            result["stdout"]
        ),
        "stderr": _truncate(
            result["stderr"]
        ),
    }


@tool("software")
def executable_location(
    name: str,
):
    """
    Locate an executable using PATH and common system lookup tools.
    """

    path = shutil.which(
        name
    )

    result = {
        "name": name,
        "path": path,
    }

    if path:

        try:
            result["realpath"] = os.path.realpath(
                path
            )

            result["file_info"] = file_info(
                path
            )

        except ToolError:
            pass

    return result


# ============================================================
# GROUP G — STORAGE / HARDWARE
# ============================================================

@tool("storage")
def block_devices():
    """
    List block devices and partitions.
    """

    result = _run(
        "lsblk",
        [
            "-J",
            "-o",
            "NAME,KNAME,PATH,TYPE,SIZE,FSTYPE,LABEL,UUID,MOUNTPOINTS,RO,RM",
        ],
    )

    try:
        return json.loads(
            result["stdout"]
        )

    except json.JSONDecodeError:

        return {
            "returncode": result["returncode"],
            "output": result["stdout"],
            "stderr": result["stderr"],
        }


@tool("storage")
def mounts():
    """
    List currently mounted filesystems.
    """

    result = _run(
        "findmnt",
        [
            "-J",
        ],
    )

    try:
        return json.loads(
            result["stdout"]
        )

    except json.JSONDecodeError:

        return {
            "returncode": result["returncode"],
            "output": result["stdout"],
            "stderr": result["stderr"],
        }


@tool("hardware")
def hardware_info():
    """
    Return basic CPU, memory, PCI, USB and sensor information.
    """

    result = {
        "cpu": {
            "logical": psutil.cpu_count(
                logical=True
            ),
            "physical": psutil.cpu_count(
                logical=False
            ),
            "percent": psutil.cpu_percent(
                interval=0.2
            ),
        },
        "memory": {
            "total": psutil.virtual_memory().total,
            "available": psutil.virtual_memory().available,
            "percent": psutil.virtual_memory().percent,
        },
    }

    for command, args, key in [
        (
            "lscpu",
            [],
            "lscpu",
        ),
        (
            "lsusb",
            [],
            "lsusb",
        ),
        (
            "lspci",
            [],
            "lspci",
        ),
    ]:

        if shutil.which(command):

            output = _run(
                command,
                args,
            )

            result[key] = _truncate(
                output["stdout"]
            )

    return result


# ============================================================
# GROUP H — BINARY / RE
# ============================================================

@tool("binary")
def binary_info(
    path: str,
):
    """
    Inspect ELF/file metadata using the file and readelf tools.
    """

    path = _validate_path(path)

    if not os.path.isfile(path):
        raise ToolError(
            "NOT_A_FILE",
            f"not a file: {path}",
        )

    result = {
        "path": path,
    }

    file_result = _run(
        "file",
        [
            "-L",
            path,
        ],
    )

    result["file"] = {
        "returncode": file_result["returncode"],
        "output": file_result["stdout"].strip(),
        "stderr": file_result["stderr"].strip(),
    }

    if shutil.which(
        "readelf"
    ):

        elf_result = _run(
            "readelf",
            [
                "-h",
                path,
            ],
        )

        result["elf_header"] = {
            "returncode": elf_result["returncode"],
            "output": _truncate(
                elf_result["stdout"]
            ),
            "stderr": _truncate(
                elf_result["stderr"]
            ),
        }

    return result


@tool("binary")
def binary_strings(
    path: str,
    max_lines: Annotated[
        int,
        Field(ge=1, le=1000),
    ] = 200,
):
    """
    Extract printable strings from a binary or file.
    """

    path = _validate_path(path)

    if not os.path.isfile(path):
        raise ToolError(
            "NOT_A_FILE",
            f"not a file: {path}",
        )

    result = _run(
        "strings",
        [
            "-n",
            "4",
            path,
        ],
        timeout=30,
    )

    lines = result["stdout"].splitlines()

    return {
        "path": path,
        "returncode": result["returncode"],
        "strings": lines[:max_lines],
        "count": min(
            len(lines),
            max_lines,
        ),
        "truncated": len(lines) > max_lines,
        "stderr": result["stderr"],
    }


@tool("binary")
def binary_dependencies(
    path: str,
):
    """
    Inspect dynamic library dependencies of an ELF binary.
    """

    path = _validate_path(path)

    if not os.path.isfile(path):
        raise ToolError(
            "NOT_A_FILE",
            f"not a file: {path}",
        )

    result = _run(
        "ldd",
        [
            path,
        ],
    )

    return {
        "path": path,
        "returncode": result["returncode"],
        "dependencies": _truncate(
            result["stdout"]
        ),
        "stderr": _truncate(
            result["stderr"]
        ),
    }


@tool("binary")
def binary_symbols(
    path: str,
    dynamic_only: bool = False,
):
    """
    Inspect ELF symbols using nm.
    """

    path = _validate_path(path)

    if not os.path.isfile(path):
        raise ToolError(
            "NOT_A_FILE",
            f"not a file: {path}",
        )

    args = [
        "-C",
    ]

    if dynamic_only:
        args.append(
            "-D"
        )

    args.append(
        path
    )

    result = _run(
        "nm",
        args,
        timeout=30,
    )

    return {
        "path": path,
        "returncode": result["returncode"],
        "symbols": _truncate(
            result["stdout"]
        ),
        "stderr": _truncate(
            result["stderr"]
        ),
    }


# ============================================================
# GROUP I — GIT
# ============================================================

@tool("git")
def git_status(
    path: str = ".",
):
    """
    Inspect Git repository status without modifying it.
    """

    path = _validate_path(path)

    result = _run(
        "git",
        [
            "-C",
            path,
            "status",
            "--short",
            "--branch",
        ],
    )

    return {
        "path": path,
        "returncode": result["returncode"],
        "output": result["stdout"],
        "stderr": result["stderr"],
    }


@tool("git")
def git_log(
    path: str = ".",
    limit: Annotated[
        int,
        Field(ge=1, le=100),
    ] = 20,
):
    """
    Inspect recent Git commit history.
    """

    path = _validate_path(path)

    result = _run(
        "git",
        [
            "-C",
            path,
            "log",
            f"-{limit}",
            "--date=iso",
            "--pretty=format:%H%x09%an%x09%ad%x09%s",
        ],
    )

    return {
        "path": path,
        "returncode": result["returncode"],
        "commits": result["stdout"],
        "stderr": result["stderr"],
    }


@tool("git")
def git_diff(
    path: str = ".",
):
    """
    Inspect the current Git diff without modifying the repository.
    """

    path = _validate_path(path)

    result = _run(
        "git",
        [
            "-C",
            path,
            "diff",
            "--",
        ],
    )

    return {
        "path": path,
        "returncode": result["returncode"],
        "diff": _truncate(
            result["stdout"],
            MAX_COMMAND_OUTPUT,
        ),
        "stderr": result["stderr"],
    }


# ============================================================
# GROUP J — PERSISTENCE / CONFIGURATION
# ============================================================

@tool("persistence")
def startup_entries():
    """
    Inspect common user and system startup locations.
    """

    locations = [
        "/etc/rc.local",
        "/etc/profile",
        "/etc/profile.d",
        os.path.expanduser(
            "~/.profile"
        ),
        os.path.expanduser(
            "~/.bashrc"
        ),
        os.path.expanduser(
            "~/.zshrc"
        ),
        os.path.expanduser(
            "~/.config/autostart"
        ),
    ]

    results = []

    for path in locations:

        if not os.path.exists(path):
            continue

        if os.path.isfile(path):

            try:

                info = _read_text(
                    path,
                    MAX_READ_BYTES,
                )

                results.append({
                    "path": path,
                    "type": "file",
                    "content": info["text"],
                })

            except ToolError as exc:

                results.append({
                    "path": path,
                    "error": exc.message,
                })

        elif os.path.isdir(path):

            try:

                entries = os.listdir(
                    path
                )

                results.append({
                    "path": path,
                    "type": "directory",
                    "entries": entries,
                })

            except PermissionError:

                results.append({
                    "path": path,
                    "error": "permission denied",
                })

    return results


@tool("persistence")
def cron_entries():
    """
    Inspect cron configuration and user crontab without changing it.
    """

    paths = [
        "/etc/crontab",
        "/etc/cron.d",
        "/etc/cron.daily",
        "/etc/cron.hourly",
        "/etc/cron.weekly",
        "/etc/cron.monthly",
    ]

    results = []

    for path in paths:

        if not os.path.exists(path):
            continue

        if os.path.isfile(path):

            try:

                data = _read_text(
                    path,
                    MAX_READ_BYTES,
                )

                results.append({
                    "path": path,
                    "content": data["text"],
                })

            except ToolError as exc:

                results.append({
                    "path": path,
                    "error": exc.message,
                })

        elif os.path.isdir(path):

            try:

                results.append({
                    "path": path,
                    "entries": os.listdir(
                        path
                    ),
                })

            except PermissionError:

                results.append({
                    "path": path,
                    "error": "permission denied",
                })

    # User crontab.
    result = _run(
        "crontab",
        [
            "-l",
        ],
    )

    results.append({
        "path": "user-crontab",
        "returncode": result["returncode"],
        "content": result["stdout"],
        "stderr": result["stderr"],
    })

    return results


@tool("configuration")
def environment_info():
    """
    Inspect the current process environment and executable PATH.
    """

    environment = dict(
        os.environ
    )

    # Do not expose likely secret values by default.
    redacted = {}

    secret_pattern = re.compile(
        r"(TOKEN|PASSWORD|PASSWD|SECRET|PRIVATE_KEY|API_KEY|CREDENTIAL)",
        re.IGNORECASE,
    )

    for key, value in environment.items():

        if secret_pattern.search(
            key
        ):

            redacted[key] = (
                "<redacted>"
            )

        else:

            redacted[key] = value

    return {
        "cwd": os.getcwd(),
        "user": (
            os.getenv("USER")
        ),
        "shell": (
            os.getenv("SHELL")
        ),
        "home": (
            os.path.expanduser("~")
        ),
        "path": os.getenv(
            "PATH",
            "",
        ).split(
            os.pathsep
        ),
        "environment": redacted,
    }


# ============================================================
# GROUP K — CONTROLLED COMMAND EXECUTION
# ============================================================

def _classify_command(
    command: str,
):
    """
    Classify a shell-like command before execution.

    Investigator is allowed only a restricted single command.

    Anything potentially mutating, privileged, chained,
    redirected or shell-executed is rejected and should
    eventually be handed to Verifier.
    """

    command = command.strip()

    if not command:
        return {
            "allowed": False,
            "reason": "empty command",
        }

    # --------------------------------------------------------
    # Shell metacharacters / chaining.
    # --------------------------------------------------------

    forbidden_syntax = [
        ";",
        "&&",
        "||",
        ">",
        "<",
        "`",
        "$(",
        "${",
        "|",
        "\n",
        "\r",
    ]

    for token in forbidden_syntax:

        if token in command:

            return {
                "allowed": False,
                "reason": (
                    f"shell syntax '{token}' "
                    "is not allowed"
                ),
            }

    try:
        argv = shlex.split(
            command
        )

    except ValueError as exc:

        return {
            "allowed": False,
            "reason": f"invalid command syntax: {exc}",
        }

    if not argv:

        return {
            "allowed": False,
            "reason": "empty command",
        }

    executable = os.path.basename(
        argv[0]
    )

    # --------------------------------------------------------
    # Explicitly dangerous commands.
    # --------------------------------------------------------

    if executable in DANGEROUS_COMMANDS:

        return {
            "allowed": False,
            "reason": (
                f"command '{executable}' "
                "requires Verifier"
            ),
            "requires_verifier": True,
        }

    # --------------------------------------------------------
    # Unknown commands are not executed.
    # --------------------------------------------------------

    if executable not in READONLY_COMMANDS:

        return {
            "allowed": False,
            "reason": (
                f"command '{executable}' "
                "is not in Investigator read-only allowlist"
            ),
        }

    # --------------------------------------------------------
    # Dangerous arguments / modes.
    # --------------------------------------------------------

    dangerous_arguments = {
        "--delete",
        "--remove",
        "--remove-files",
        "--exec",
        "-exec",
        "-delete",
        "--write",
        "--output",
        "-o",
        "--output-document",
        "-O",
        "--upload-file",
        "-T",
        "--data",
        "--data-raw",
        "--data-binary",
        "--form",
        "--form-string",
    }

    for arg in argv[1:]:

        if arg in dangerous_arguments:

            return {
                "allowed": False,
                "reason": (
                    f"argument '{arg}' "
                    "can perform a mutating operation"
                ),
                "requires_verifier": True,
            }

    # --------------------------------------------------------
    # Special handling for apt.
    # --------------------------------------------------------

    if executable == "apt":

        if not any(
            arg in {
                "list",
                "show",
                "policy",
                "search",
                "depends",
                "rdepends",
                "madison",
            }
            for arg in argv[1:]
        ):

            return {
                "allowed": False,
                "reason": (
                    "apt command is not "
                    "an explicitly read-only operation"
                ),
                "requires_verifier": True,
            }

    # --------------------------------------------------------
    # Special handling for systemctl.
    # --------------------------------------------------------

    if executable == "systemctl":

        allowed_subcommands = {
            "status",
            "show",
            "cat",
            "list-units",
            "list-unit-files",
            "list-timers",
            "is-active",
            "is-enabled",
            "is-failed",
            "is-system-running",
            "get-default",
            "list-dependencies",
        }

        subcommands = {
            arg
            for arg in argv[1:]
            if not arg.startswith("-")
        }

        if not (
            subcommands
            & allowed_subcommands
        ):

            return {
                "allowed": False,
                "reason": (
                    "systemctl operation "
                    "requires Verifier"
                ),
                "requires_verifier": True,
            }

    # --------------------------------------------------------
    # Special handling for git.
    # --------------------------------------------------------

    if executable == "git":

        readonly_subcommands = {
            "status",
            "log",
            "diff",
            "show",
            "branch",
            "remote",
            "ls-files",
            "rev-parse",
            "tag",
            "describe",
            "config",
        }

        subcommands = {
            arg
            for arg in argv[1:]
            if not arg.startswith("-")
            and arg != "-C"
        }

        if not (
            subcommands
            & readonly_subcommands
        ):

            return {
                "allowed": False,
                "reason": (
                    "git operation "
                    "requires Verifier"
                ),
                "requires_verifier": True,
            }

    return {
        "allowed": True,
        "argv": argv,
    }


@tool("command")
def execute_readonly_command(
    command: Annotated[
        str,
        Field(
            description=(
                "One read-only Linux command. "
                "No shell chaining, pipes, redirection "
                "or mutation is allowed."
            )
        ),
    ],
    timeout: Annotated[
        int,
        Field(ge=1, le=60),
    ] = 15,
):
    """
    Execute one explicitly allowed read-only Linux command.

    Mutating, privileged or unsafe commands return
    REQUIRES_VERIFIER instead of executing.
    """

    classification = _classify_command(
        command
    )

    if not classification["allowed"]:

        if classification.get(
            "requires_verifier",
            False,
        ):

            raise ToolError(
                "REQUIRES_VERIFIER",
                classification["reason"],
            )

        raise ToolError(
            "COMMAND_NOT_ALLOWED",
            classification["reason"],
        )

    argv = classification["argv"]

    executable = argv[0]

    # Resolve executable so the result tells the agent
    # what actually ran.
    executable_path = shutil.which(
        executable
    )

    if not executable_path:

        raise ToolError(
            "COMMAND_NOT_FOUND",
            f"command not found: {executable}",
        )

    try:

        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )

    except subprocess.TimeoutExpired:

        raise ToolError(
            "TIMEOUT",
            f"command timed out after {timeout}s",
            retryable=True,
        )

    except PermissionError:

        raise ToolError(
            "PERMISSION_DENIED",
            f"permission denied: {executable}",
        )

    return {
        "command": command,
        "executable": executable_path,
        "returncode": proc.returncode,
        "stdout": _truncate(
            proc.stdout,
            MAX_COMMAND_OUTPUT,
        ),
        "stderr": _truncate(
            proc.stderr,
            MAX_COMMAND_OUTPUT,
        ),
    }

@tool("WEB")
def web_search(
    query: str,
    max_results: int = 5,
) -> list:
    """
    Search the public web using DuckDuckGo.

    Returns search result titles, URLs, and snippets.
    This tool is read-only.
    """

    if not query.strip():
        raise ToolError(
            "INVALID_ARGUMENTS",
            "query is required",
        )

    if max_results < 1 or max_results > 10:
        raise ToolError(
            "INVALID_ARGUMENTS",
            "max_results must be between 1 and 10",
        )

    try:
        results = []

        with DDGS() as ddgs:
            for item in ddgs.text(
                query,
                max_results=max_results,
            ):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("href", ""),
                    "snippet": item.get("body", ""),
                })

        return results

    except Exception as exc:
        raise ToolError(
            "WEB_SEARCH_FAILED",
            str(exc)[:300],
            True,
        )

# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    print(
        f"{len(TOOLS)} read-only Investigator tools registered"
    )

    print()

    for name in sorted(TOOLS):

        meta = TOOLS[name]

        print(
            f"- {name} "
            f"[group={meta['group']}, "
            f"mutating={meta['mutating']}]"
        )