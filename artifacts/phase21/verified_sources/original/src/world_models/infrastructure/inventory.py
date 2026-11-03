"""Read-only host inventory without importing PyTorch or launching a simulator."""

from __future__ import annotations

import csv
import ctypes
from datetime import datetime, timezone
from importlib import metadata
import io
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any


REQUESTED_RESOURCES = {"gpu_count": 4, "gpu_model": "H200", "cpu_cores": 4}


def _physical_memory_bytes() -> int | None:
    try:
        if sys.platform == "win32":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_uint32),
                    ("load", ctypes.c_uint32),
                    ("total_physical", ctypes.c_uint64),
                    ("available_physical", ctypes.c_uint64),
                    ("total_pagefile", ctypes.c_uint64),
                    ("available_pagefile", ctypes.c_uint64),
                    ("total_virtual", ctypes.c_uint64),
                    ("available_virtual", ctypes.c_uint64),
                    ("available_extended_virtual", ctypes.c_uint64),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.total_physical)
            return None
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if pages > 0 and page_size > 0:
                return int(pages * page_size)
    except (AttributeError, OSError, ValueError):
        pass
    return None


def _cpu_inventory() -> dict[str, Any]:
    logical_count = os.cpu_count()
    affinity_count: int | None = None
    quota_cores: float | None = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity_count = len(os.sched_getaffinity(0))
        except OSError:
            pass
    if sys.platform.startswith("linux"):
        try:
            quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
            if quota != "max" and int(period) > 0:
                quota_cores = int(quota) / int(period)
        except (OSError, ValueError):
            pass
    limits = [float(value) for value in (logical_count, affinity_count, quota_cores) if value is not None]
    return {
        "logical_cpu_count": logical_count,
        "affinity_cpu_count": affinity_count,
        "cgroup_v2_quota_cores": quota_cores,
        "estimated_available_logical_cores": min(limits) if limits else None,
        "processor": platform.processor() or None,
        "note": "Logical CPUs/quotas are observations of this host, not proof of the planned GPU host allocation.",
    }


def _gpu_inventory() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    result: dict[str, Any] = {
        "probe": "nvidia-smi",
        "executable": executable,
        "status": "not_found",
        "gpus": [],
        "cuda_runtime": "not_probed_without_importing_training_stack",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if executable is None:
        result["note"] = "No nvidia-smi found on PATH; this does not establish that no GPU exists."
        return result
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if completed.returncode != 0:
            result.update(status="error", error=(completed.stderr or completed.stdout).strip()[:1000])
            return result
        gpus = []
        for row in csv.reader(io.StringIO(completed.stdout)):
            if not row or not any(field.strip() for field in row):
                continue
            if len(row) != 4:
                raise ValueError("Unexpected nvidia-smi GPU query format")
            index, name, memory, driver = (field.strip() for field in row)
            gpus.append(
                {
                    "index": int(index),
                    "name": name,
                    "memory_total_mib": float(memory),
                    "driver_version": driver,
                }
            )
        result.update(status="ok", gpus=gpus)
    except subprocess.TimeoutExpired:
        result.update(status="timeout", error="nvidia-smi exceeded the 10-second inventory timeout")
    except (OSError, ValueError, csv.Error) as error:
        result.update(status="error", error=str(error)[:1000])
    return result


def _package_versions() -> dict[str, Any]:
    result = {}
    for name, distribution in {
        "torch": "torch",
        "vizdoom": "vizdoom",
        "yaml": "PyYAML",
        "numpy": "numpy",
    }.items():
        try:
            result[name] = {"distribution": distribution, "version": metadata.version(distribution), "status": "installed"}
        except metadata.PackageNotFoundError:
            result[name] = {"distribution": distribution, "version": None, "status": "not_installed"}
    return result


def collect_inventory(workspace: str | Path | None = None) -> dict[str, Any]:
    """Return JSON-compatible, measured host facts and unverified target resources.

    GPU visibility and available memory for actual learning remain Phase 1
    validation tasks.  This function never imports torch, numpy, or vizdoom.
    """
    workspace_path = Path(workspace or Path.cwd()).resolve()
    try:
        usage = shutil.disk_usage(workspace_path)
        disk: dict[str, Any] = {
            "status": "ok",
            "workspace": str(workspace_path),
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
        }
    except OSError as error:
        disk = {"status": "unavailable", "workspace": str(workspace_path), "error": str(error)}
    cpu = _cpu_inventory()
    gpu = _gpu_inventory()
    available_cpu = cpu["estimated_available_logical_cores"]
    observed_gpu_models = [entry["name"] for entry in gpu["gpus"]]
    return {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Observed local host; planned remote/training hardware has not been verified by this inventory.",
        "requested_resources": dict(REQUESTED_RESOURCES),
        "python": {"version": platform.python_version(), "executable": sys.executable, "implementation": platform.python_implementation()},
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(), "description": platform.platform()},
        "cpu": cpu,
        "memory": {"physical_total_bytes": _physical_memory_bytes()},
        "disk": disk,
        "gpu": gpu,
        "packages": _package_versions(),
        "scenario_tools": {name: shutil.which(name) for name in ("acc", "gdcc-acc", "slade")},
        "requested_resource_comparison": {
            "cpu_estimate_at_least_four": available_cpu >= 4 if available_cpu is not None else None,
            "gpu_probe_succeeded": gpu["status"] == "ok",
            "observed_h200_count": sum("H200" in name.upper() for name in observed_gpu_models) if gpu["status"] == "ok" else None,
            "note": "Informational comparison only; simulator, CUDA, and multi-GPU training have not been smoke-tested.",
        },
    }
