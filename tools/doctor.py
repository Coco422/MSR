from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from msr.core.config import load_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate local model directories for MSR.")
    parser.add_argument(
        "--include-alternates",
        action="store_true",
        help="Also validate non-default configured models.",
    )
    parser.add_argument(
        "--include-qwen",
        action="store_true",
        help="Also validate optional Qwen3-ASR and Qwen3-ForcedAligner directories.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings(project_root=ROOT)
    checked_models = [
        model
        for model in settings.models
        if model.default
        or (args.include_alternates and model.backend in {"funasr", "faster_whisper", "pyannote"})
        or (args.include_qwen and model.backend == "qwen_asr")
    ]
    print("MSR doctor")
    print(f"Project root: {settings.project_root}")
    print(f"Strict offline: {settings.app.strict_offline}")
    print(f"Configured models: {len(settings.models)}")
    print(f"Checking models: {len(checked_models)}")
    print(f"Include alternates: {args.include_alternates}")
    print(f"Include qwen: {args.include_qwen}")

    has_error = False
    for model in settings.models:
        if model not in checked_models:
            print(
                f"[SKIP] kind={model.kind} id={model.id} backend={model.backend} "
                "reason=alternate"
            )
            continue
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
            continue

        if model.backend == "qwen_asr":
            forced_aligner_path = model.options.get("forced_aligner_path")
            if not isinstance(forced_aligner_path, Path):
                print(f"[ERROR] kind={model.kind} id={model.id} backend={model.backend} forced_aligner=missing")
                has_error = True
                continue
            aligner_exists = forced_aligner_path.exists()
            aligner_remote_like = "://" in str(forced_aligner_path)
            aligner_status = "OK" if aligner_exists and not aligner_remote_like else "ERROR"
            print(
                f"[{aligner_status}] kind={model.kind} id={model.id} backend={model.backend} "
                f"forced_aligner_path={forced_aligner_path}"
            )
            if aligner_remote_like or not aligner_exists:
                has_error = True

    if has_error:
        print("Doctor found configuration issues.")
        return 1

    print("Doctor checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
