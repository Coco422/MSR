from __future__ import annotations

from typing import Any

import psutil


class ResourceMonitor:
    def __init__(self):
        self._gpu_available = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "cpu": {
                "percent": psutil.cpu_percent(interval=None),
                "count": psutil.cpu_count(),
            },
            "memory": {
                "percent": psutil.virtual_memory().percent,
                "used_mb": round(psutil.virtual_memory().used / 1024 / 1024, 2),
                "total_mb": round(psutil.virtual_memory().total / 1024 / 1024, 2),
            },
            "gpu": self._gpu_snapshot(),
        }

    def _gpu_snapshot(self) -> list[dict[str, Any]]:
        try:
            import pynvml

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            items = []
            for index in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="ignore")
                items.append(
                    {
                        "index": index,
                        "name": name,
                        "utilization_percent": util.gpu,
                        "memory_used_mb": round(memory.used / 1024 / 1024, 2),
                        "memory_total_mb": round(memory.total / 1024 / 1024, 2),
                    }
                )
            return items
        except Exception:
            return []
