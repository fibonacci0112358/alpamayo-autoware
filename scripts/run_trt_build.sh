#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR=""
MODEL_NAME_OR_PATH="nvidia/Alpamayo-1.5-10B"
CLIP_ID="030c760c-ae38-49aa-9ad8-f5650a545d26"
T0_US="5100000"
MAX_GENERATION_LENGTH="64"
NUM_CALIBRATION_SAMPLES="8"
CALIBRATION_METHOD="entropy"
SMOOTHQUANT_ALPHA="0.6"
SEED="0"
SKIP_VALIDATION="false"
INSTALL_DEPS="false"
TMUX_SESSION="alpamayo-trt-build"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") --output-dir <dir> [options]

Required:
  --output-dir <dir>                 Output directory for ONNX/TRT artifacts

Options:
  --model-name-or-path <value>       HF model ID or local model directory path
  --clip-id <value>                  Clip ID for calibration input
  --t0-us <value>                    Timestamp offset (microseconds)
  --max-generation-length <value>    Generation length for calibration run
  --num-calibration-samples <value>  Number of calibration samples
  --calibration-method <value>       entropy | percentile | minmax
  --smoothquant-alpha <value>        SmoothQuant alpha
  --seed <value>                     Random seed
  --skip-validation                  Skip native-vs-TRT validation pass
  --python-bin <path>                Python executable to use (default: python3)
  --install-deps                     Run: uv pip install -r scripts/requirements-trt-build.txt
  --tmux-session <name>              tmux session name (default: alpamayo-trt-build)
  -h, --help                         Show this help

Examples:
  # Use default HF model
  $(basename "$0") --output-dir /tmp/alpamayo-engines

  # Use local SFT model directory
  $(basename "$0") \\
    --output-dir /tmp/alpamayo-engines \\
    --model-name-or-path /data/models/alpamayo-sft

  # Start build in tmux and attach
  tmux attach -t alpamayo-trt-build
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --model-name-or-path)
      MODEL_NAME_OR_PATH="$2"
      shift 2
      ;;
    --clip-id)
      CLIP_ID="$2"
      shift 2
      ;;
    --t0-us)
      T0_US="$2"
      shift 2
      ;;
    --max-generation-length)
      MAX_GENERATION_LENGTH="$2"
      shift 2
      ;;
    --num-calibration-samples)
      NUM_CALIBRATION_SAMPLES="$2"
      shift 2
      ;;
    --calibration-method)
      CALIBRATION_METHOD="$2"
      shift 2
      ;;
    --smoothquant-alpha)
      SMOOTHQUANT_ALPHA="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --skip-validation)
      SKIP_VALIDATION="true"
      shift
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --install-deps)
      INSTALL_DEPS="true"
      shift
      ;;
    --tmux-session)
      TMUX_SESSION="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${OUTPUT_DIR}" ]]; then
  echo "Error: --output-dir is required." >&2
  usage
  exit 1
fi

cd "${REPO_ROOT}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "Error: tmux is not installed or not in PATH." >&2
  exit 1
fi

cmd=(
  "${PYTHON_BIN}" scripts/build_trt_expert_engine.py
  --output-dir "${OUTPUT_DIR}"
  --model-name-or-path "${MODEL_NAME_OR_PATH}"
  --clip-id "${CLIP_ID}"
  --t0-us "${T0_US}"
  --max-generation-length "${MAX_GENERATION_LENGTH}"
  --num-calibration-samples "${NUM_CALIBRATION_SAMPLES}"
  --calibration-method "${CALIBRATION_METHOD}"
  --smoothquant-alpha "${SMOOTHQUANT_ALPHA}"
  --seed "${SEED}"
)

if [[ "${SKIP_VALIDATION}" == "true" ]]; then
  cmd+=(--skip-validation)
fi

if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  echo "Error: tmux session '${TMUX_SESSION}' already exists." >&2
  echo "Use a different name with --tmux-session or attach with: tmux attach -t ${TMUX_SESSION}" >&2
  exit 1
fi

printf -v build_cmd '%q ' "${cmd[@]}"
run_cmd="set -euo pipefail; cd $(printf '%q' "${REPO_ROOT}");"
if [[ "${INSTALL_DEPS}" == "true" ]]; then
  run_cmd+=" uv pip install -r scripts/requirements-trt-build.txt;"
fi
run_cmd+=" ${build_cmd}"

tmux new-session -d -s "${TMUX_SESSION}" "bash -lc $(printf '%q' "${run_cmd}")"

echo "Started TensorRT build in tmux session: ${TMUX_SESSION}"
echo "Attach: tmux attach -t ${TMUX_SESSION}"
echo "Detach: Ctrl+b then d"
