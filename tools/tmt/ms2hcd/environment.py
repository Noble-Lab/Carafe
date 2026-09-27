#!/usr/bin/env python
"""Record the hardware and software a model was trained with.

Prints (and with --out writes) the OS, CPU, RAM, GPU, NVIDIA driver, CUDA and cuDNN versions, and
the versions of Python and the packages training depends on. Deliberately collects nothing that
identifies a person or machine: no host name, user name, serial numbers or file paths.

    python environment.py --out environment.txt
"""
from __future__ import annotations

import argparse
import importlib
import os
import platform
import subprocess
import sys

PACKAGES = ["torch", "peptdeep", "alphabase", "numpy", "pandas", "scipy", "pyarrow", "transformers"]


def ram_gb():
    try:
        import psutil
        return f"{psutil.virtual_memory().total / 2**30:.0f} GiB"
    except ImportError:
        pass
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return f"{status.ullTotalPhys / 2**30:.0f} GiB"
    return "unknown"


def cpu_name():
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif os.path.exists("/proc/cpuinfo"):  # Linux: platform.processor() gives only the architecture
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def nvidia_driver():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def cudnn_version(code):
    """torch.backends.cudnn.version() as major.minor.patch. The encoding changed with cuDNN 9:
    9.1.0 is 90100 (major*10000 + minor*100 + patch), 8.9.2 was 8902 (major*1000 + minor*100 + patch)."""
    if not code:
        return "none"
    if code >= 10000:
        return f"{code // 10000}.{code // 100 % 100}.{code % 100}"
    return f"{code // 1000}.{code // 100 % 10}.{code % 100}"


def os_name():
    """platform.release() reports Windows 11 as "10"; builds 22000 and later are Windows 11."""
    if sys.platform == "win32":
        build = platform.version().split(".")[-1]
        major = "11" if build.isdigit() and int(build) >= 22000 else platform.release()
        return f"Windows {major} (build {build})"
    return f"{platform.system()} {platform.release()}"


def collect():
    rows = [
        ("OS", os_name()),
        ("CPU", f"{cpu_name()}, {__import__('os').cpu_count()} logical cores"),
        ("RAM", ram_gb()),
    ]
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            rows.append(("GPU", f"{torch.cuda.get_device_name(0)}, {props.total_memory / 2**30:.0f} GiB"))
        else:
            rows.append(("GPU", "none (CUDA not available)"))
        rows.append(("NVIDIA driver", nvidia_driver()))
        rows.append(("CUDA (PyTorch build)", str(torch.version.cuda)))
        cudnn = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        rows.append(("cuDNN", cudnn_version(cudnn)))
    except ImportError:
        rows.append(("GPU", "unknown (PyTorch not installed)"))
    rows.append(("Python", platform.python_version()))
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            rows.append((name, getattr(module, "__version__", "installed")))
        except ImportError:
            pass
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="also write the table to this file")
    args = ap.parse_args(argv)
    text = "\n".join(f"{k:<22}{v}" for k, v in collect())
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
