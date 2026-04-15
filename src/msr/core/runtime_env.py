from __future__ import annotations

import os
import sys


def runtime_context() -> dict[str, str]:
    return {
        "python": sys.executable,
        "prefix": sys.prefix,
        "venv": os.getenv("VIRTUAL_ENV", ""),
        "cwd": os.getcwd(),
    }


def format_runtime_context() -> str:
    context = runtime_context()
    parts = [
        f"python={context['python']}",
        f"prefix={context['prefix']}",
    ]
    if context["venv"]:
        parts.append(f"venv={context['venv']}")
    parts.append(f"cwd={context['cwd']}")
    return ", ".join(parts)
