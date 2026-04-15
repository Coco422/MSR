#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${MSR_PYTHON_BIN:-python3.12}"

usage() {
  cat <<'EOF'
MSR 运行环境切换脚本

用法:
  bash tools/runtime_env.sh setup [default|pyannote|all]
  bash tools/runtime_env.sh run [default|pyannote]
  bash tools/runtime_env.sh exec [default|pyannote] <command...>
  bash tools/runtime_env.sh help

说明:
  default   : FunASR + 3D-Speaker + WebRTC VAD，偏兼容与当前默认链
  pyannote  : faster-whisper + pyannote，偏准确率优先
  注意      : 需要切换到 pyannote 链路时，不要直接执行 uv run msr-api

示例:
  bash tools/runtime_env.sh setup default
  bash tools/runtime_env.sh setup pyannote
  bash tools/runtime_env.sh run pyannote
  bash tools/runtime_env.sh exec pyannote python tools/doctor.py --include-alternates
EOF
}

ensure_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    echo "未找到 uv，请先安装 uv。" >&2
    exit 1
  fi
}

profile_dir() {
  case "${1}" in
    default) echo "${ROOT_DIR}/.venv.default" ;;
    pyannote) echo "${ROOT_DIR}/.venv.pyannote" ;;
    *)
      echo "未知 profile: ${1}" >&2
      exit 1
      ;;
  esac
}

python_path() {
  echo "$(profile_dir "${1}")/bin/python"
}

ensure_profile_exists() {
  local profile="${1}"
  local dir
  dir="$(profile_dir "${profile}")"
  if [[ ! -x "${dir}/bin/python" ]]; then
    echo "运行环境 ${profile} 尚未创建，请先执行: bash tools/runtime_env.sh setup ${profile}" >&2
    exit 1
  fi
}

create_default_env() {
  local dir="${ROOT_DIR}/.venv.default"
  local python="${dir}/bin/python"
  echo "[default] 创建虚拟环境: ${dir}"
  uv venv --python "${PYTHON_BIN}" "${dir}"
  echo "[default] 安装项目基础依赖: dev"
  uv pip install --python "${python}" -e ".[dev]"
  echo "[default] 安装 torch / FunASR / 3D-Speaker 运行栈"
  uv pip install --python "${python}" \
    "torch==2.10.*" \
    "torchaudio==2.10.*" \
    "torchvision==0.25.*" \
    "funasr>=1.2.0" \
    "speakerlab" \
    "addict>=2.4.0,<3.0.0" \
    "simplejson>=3.20.0,<4.0.0" \
    "sortedcontainers>=2.4.0,<3.0.0"
}

create_pyannote_env() {
  local dir="${ROOT_DIR}/.venv.pyannote"
  local python="${dir}/bin/python"
  echo "[pyannote] 创建虚拟环境: ${dir}"
  uv venv --python "${PYTHON_BIN}" "${dir}"
  echo "[pyannote] 安装项目基础依赖: dev"
  uv pip install --python "${python}" -e ".[dev]"
  echo "[pyannote] 安装 torch / faster-whisper / pyannote 运行栈"
  uv pip install --python "${python}" \
    "torch==2.10.*" \
    "torchaudio==2.10.*" \
    "torchvision==0.25.*" \
    "faster-whisper>=1.1.0,<2.0.0" \
    "pyannote.audio>=4.0.0,<5.0.0"
}

setup_profiles() {
  local target="${1:-all}"
  ensure_uv
  cd "${ROOT_DIR}"
  case "${target}" in
    default)
      create_default_env
      ;;
    pyannote)
      create_pyannote_env
      ;;
    all)
      create_default_env
      create_pyannote_env
      ;;
    *)
      echo "不支持的 setup 目标: ${target}" >&2
      usage
      exit 1
      ;;
  esac

  cat <<'EOF'

创建完成。常用命令:
  bash tools/runtime_env.sh run default
  bash tools/runtime_env.sh run pyannote
  bash tools/runtime_env.sh exec pyannote python tools/doctor.py --include-alternates
EOF
}

run_profile() {
  local profile="${1}"
  ensure_profile_exists "${profile}"
  local dir
  local pythonpath="${ROOT_DIR}/src"
  dir="$(profile_dir "${profile}")"
  cd "${ROOT_DIR}"
  echo "[${profile}] 启动 MSR 服务"
  echo "[${profile}] Python: ${dir}/bin/python"
  if [[ -n "${PYTHONPATH:-}" ]]; then
    pythonpath="${pythonpath}:${PYTHONPATH}"
  fi
  exec env VIRTUAL_ENV="${dir}" PATH="${dir}/bin:${PATH}" PYTHONPATH="${pythonpath}" "${dir}/bin/python" -m msr.app.main
}

exec_profile() {
  local profile="${1}"
  shift
  ensure_profile_exists "${profile}"
  if [[ "${#}" -eq 0 ]]; then
    echo "exec 模式至少需要一个命令。" >&2
    usage
    exit 1
  fi
  local dir
  local pythonpath="${ROOT_DIR}/src"
  dir="$(profile_dir "${profile}")"
  cd "${ROOT_DIR}"
  echo "[${profile}] Python: ${dir}/bin/python"
  if [[ -n "${PYTHONPATH:-}" ]]; then
    pythonpath="${pythonpath}:${PYTHONPATH}"
  fi
  exec env VIRTUAL_ENV="${dir}" PATH="${dir}/bin:${PATH}" PYTHONPATH="${pythonpath}" "$@"
}

main() {
  local action="${1:-help}"
  case "${action}" in
    setup)
      setup_profiles "${2:-all}"
      ;;
    run)
      if [[ $# -lt 2 ]]; then
        echo "run 模式需要指定 profile。" >&2
        usage
        exit 1
      fi
      run_profile "${2}"
      ;;
    exec)
      if [[ $# -lt 3 ]]; then
        echo "exec 模式需要指定 profile 和命令。" >&2
        usage
        exit 1
      fi
      shift
      exec_profile "$@"
      ;;
    help|-h|--help)
      usage
      ;;
    *)
      echo "未知操作: ${action}" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
