from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from msr.core.config import load_settings


def main() -> int:
    settings = load_settings(project_root=ROOT)
    print("MSR doctor")
    print(f"Project root: {settings.project_root}")
    print(f"Strict offline: {settings.app.strict_offline}")
    print(f"Configured models: {len(settings.models)}")

    has_error = False
    for model in settings.models:
        local_path = model.local_path
        is_remote_like = "://" in str(local_path)
        exists = local_path.exists()
        status = "OK" if exists and not is_remote_like else "ERROR"
        print(
            f"[{status}] kind={model.kind} id={model.id} backend={model.backend} "
            f"path={local_path}"
        )
        if is_remote_like or not exists:
            has_error = True

    if has_error:
        print("Doctor found configuration issues.")
        return 1

    print("Doctor checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
