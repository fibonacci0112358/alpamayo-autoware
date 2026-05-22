#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${PYTHON_BIN:-}" && -x "${REPO_ROOT}/.venv-trt/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/.venv-trt/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR="${HOME}/test_repo/engines"
MODEL_NAME_OR_PATH="${HOME}/test_repo/stage2_h100_8gpu_full_folders"
PYTHON_PATH=""
VLM_NAME_OR_PATH=""
CLIP_ID="030c760c-ae38-49aa-9ad8-f5650a545d26"
T0_US="5100000"
MAX_GENERATION_LENGTH="64"
NUM_CALIBRATION_SAMPLES="8"
CALIBRATION_METHOD="entropy"
SMOOTHQUANT_ALPHA="0.6"
SEED="0"
SKIP_VALIDATION="false"
INSTALL_DEPS="false"

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
  --python-path <path>               Additional Python source path (prepended to PYTHONPATH)
  --install-deps                     Run: uv pip install -r scripts/requirements-trt-build.txt
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
    --python-path)
      PYTHON_PATH="$2"
      shift 2
      ;;
    --vlm-name-or-path)
      VLM_NAME_OR_PATH="$2"
      shift 2
      ;;
    --install-deps)
      INSTALL_DEPS="true"
      shift
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

# Set PYTHONPATH so local `src/` package modules (e.g. alpamayo1_5) are importable.
if [[ -n "${PYTHON_PATH}" ]]; then
  export PYTHONPATH="${PYTHON_PATH}:${PYTHONPATH:-}"
else
  export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/build:${PYTHONPATH:-}"
fi

cuda_lib_dirs=()
if [[ -d "${REPO_ROOT}/.venv-trt/lib/python3.12/site-packages/nvidia" ]]; then
  while IFS= read -r -d '' lib_dir; do
    cuda_lib_dirs+=("${lib_dir}")
  done < <(find "${REPO_ROOT}/.venv-trt/lib/python3.12/site-packages/nvidia" -type d -name lib -print0)
fi
if [[ -d "${REPO_ROOT}/.venv-trt/lib/python3.12/site-packages/torch/lib" ]]; then
  cuda_lib_dirs+=("${REPO_ROOT}/.venv-trt/lib/python3.12/site-packages/torch/lib")
fi
if [[ -d "/usr/local/cuda/targets/x86_64-linux/lib" ]]; then
  cuda_lib_dirs+=("/usr/local/cuda/targets/x86_64-linux/lib")
fi
if [[ ${#cuda_lib_dirs[@]} -gt 0 ]]; then
  cuda_ld_path=$(IFS=:; echo "${cuda_lib_dirs[*]}")
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="${cuda_ld_path}:${LD_LIBRARY_PATH}"
  else
    export LD_LIBRARY_PATH="${cuda_ld_path}"
  fi
fi
echo "PYTHONPATH=${PYTHONPATH}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
if [[ -n "${VLM_NAME_OR_PATH}" ]]; then
  export ALPAMAYO_VLM_NAME_OR_PATH="${VLM_NAME_OR_PATH}"
  echo "ALPAMAYO_VLM_NAME_OR_PATH=${ALPAMAYO_VLM_NAME_OR_PATH}"
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

printf -v build_cmd '%q ' "${cmd[@]}"
run_cmd="set -euo pipefail; cd $(printf '%q' "${REPO_ROOT}");"
if [[ "${INSTALL_DEPS}" == "true" ]]; then
  run_cmd+=" uv pip install -r scripts/requirements-trt-build.txt;"
fi
run_cmd+=" ${build_cmd}"

echo "Starting TensorRT build (foreground)."
if ! ${PYTHON_BIN} -c "import torchvision" >/dev/null 2>&1; then
  cat <<MSG
Missing Python dependency: torchvision

The build requires 'torchvision' for some image/video processors. Install a
version matching your installed 'torch' (GPU vs CPU/CUDA) before continuing.
Example:
  python -m pip install torchvision

Refer to: https://pytorch.org/get-started/locally/ for platform-specific wheels.

You can also let this script install deps by re-running with '--install-deps'.
MSG
  exit 1
fi
bash -lc "${run_cmd}"
