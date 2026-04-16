from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from msr.core.config import load_settings


FUNASR_DEFAULTS = {
    "funasr-paraformer-zh": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
}

THREE_D_SPEAKER_DEFAULTS = [
    ("iic/speech_campplus_sv_zh_en_16k-common_advanced", "v1.0.0"),
    ("iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", "v2.0.4"),
]

FASTER_WHISPER_DEFAULTS = {
    "faster-whisper-large-v3": "Systran/faster-whisper-large-v3",
}

PYANNOTE_DEFAULTS = {
    "pyannote-community-1": "pyannote/speaker-diarization-community-1",
}

QWEN_DEFAULTS = {
    "qwen3-asr-0.6b": "Qwen/Qwen3-ASR-0.6B",
    "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
}

QWEN_FORCED_ALIGNER_DEFAULT = "Qwen/Qwen3-ForcedAligner-0.6B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap local model directories for MSR.")
    parser.add_argument(
        "--include-alternates",
        action="store_true",
        help="Also download optional faster-whisper and pyannote models when possible.",
    )
    parser.add_argument(
        "--hf-token",
        default="",
        help="Hugging Face token for pyannote downloads. Required only for pyannote bootstrap.",
    )
    parser.add_argument(
        "--include-qwen",
        action="store_true",
        help="Also download optional Qwen3-ASR and Qwen3-ForcedAligner local directories.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings(project_root=ROOT)
    print("MSR model bootstrap")
    print(f"Project root: {settings.project_root}")
    bootstrapped_aligners: set[Path] = set()

    for model in settings.models:
        model.local_path.parent.mkdir(parents=True, exist_ok=True)

        if model.backend == "funasr":
            bootstrap_funasr(model.id, model.local_path)
        elif model.backend == "3d_speaker":
            bootstrap_three_d_speaker(model.local_path)
        elif args.include_alternates and model.backend == "faster_whisper":
            bootstrap_faster_whisper(model.id, model.local_path)
        elif args.include_alternates and model.backend == "pyannote":
            bootstrap_pyannote(model.id, model.local_path, args.hf_token)
        elif args.include_qwen and model.backend == "qwen_asr":
            bootstrap_qwen_asr(model.id, model.local_path)
            forced_aligner_path = model.options.get("forced_aligner_path")
            if isinstance(forced_aligner_path, Path) and forced_aligner_path not in bootstrapped_aligners:
                bootstrap_qwen_forced_aligner(forced_aligner_path)
                bootstrapped_aligners.add(forced_aligner_path)
        else:
            print(f"[SKIP] {model.id} ({model.backend})")

    print("Bootstrap completed.")
    return 0


def bootstrap_funasr(model_id: str, target_dir: Path) -> None:
    from modelscope.hub.snapshot_download import snapshot_download

    source = FUNASR_DEFAULTS.get(model_id)
    if not source:
        print(f"[SKIP] No bootstrap mapping for FunASR model: {model_id}")
        return

    print(f"[DL] FunASR {model_id} -> {target_dir}")
    snapshot_download(source, local_dir=str(target_dir))


def bootstrap_three_d_speaker(target_dir: Path) -> None:
    from modelscope.hub.snapshot_download import snapshot_download

    print(f"[DL] 3D-Speaker cache -> {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    for model_id, revision in THREE_D_SPEAKER_DEFAULTS:
        snapshot_download(model_id, revision=revision, cache_dir=str(target_dir))


def bootstrap_faster_whisper(model_id: str, target_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    source = FASTER_WHISPER_DEFAULTS.get(model_id)
    if not source:
        print(f"[SKIP] No bootstrap mapping for faster-whisper model: {model_id}")
        return

    print(f"[DL] faster-whisper {model_id} -> {target_dir}")
    snapshot_download(repo_id=source, local_dir=str(target_dir))


def bootstrap_pyannote(model_id: str, target_dir: Path, hf_token: str) -> None:
    if not hf_token:
        print(f"[SKIP] Missing --hf-token for pyannote model: {model_id}")
        return

    from huggingface_hub import snapshot_download

    source = PYANNOTE_DEFAULTS.get(model_id)
    if not source:
        print(f"[SKIP] No bootstrap mapping for pyannote model: {model_id}")
        return

    print(f"[DL] pyannote {model_id} -> {target_dir}")
    snapshot_download(repo_id=source, local_dir=str(target_dir), token=hf_token)


def bootstrap_qwen_asr(model_id: str, target_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    source = QWEN_DEFAULTS.get(model_id)
    if not source:
        print(f"[SKIP] No bootstrap mapping for Qwen3-ASR model: {model_id}")
        return

    print(f"[DL] Qwen3-ASR {model_id} -> {target_dir}")
    snapshot_download(repo_id=source, local_dir=str(target_dir))


def bootstrap_qwen_forced_aligner(target_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    print(f"[DL] Qwen3-ForcedAligner -> {target_dir}")
    snapshot_download(repo_id=QWEN_FORCED_ALIGNER_DEFAULT, local_dir=str(target_dir))


if __name__ == "__main__":
    raise SystemExit(main())
