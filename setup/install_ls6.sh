#!/bin/bash
# Lonestar6 GPU node only (idev). Separate venv from the MATH OPDL install.
set -euo pipefail

HOPD_HOME="${HOPD_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck disable=SC1091
source "$HOPD_HOME/setup/env_vars.ls6.sh"

if [[ -n "${HOPD_ROOT:-}" ]]; then
    ROOT="$HOPD_ROOT"
elif [[ -n "${SCRATCH:-}" ]]; then
    ROOT="$SCRATCH/hopd"
else
    ROOT="$(cd "$HOPD_HOME/.." && pwd)"
fi
VENV="${HOPD_VENV:-$ROOT/.venv}"
VERL_HOME="${HOPD_VERL_HOME:-$HOPD_HOME/verl}"

echo "HOPD_HOME=$HOPD_HOME"
echo "VENV=$VENV"
echo "VERL_HOME=$VERL_HOME"
echo "CUDA_HOME=${CUDA_HOME:-unset}"
command -v nvcc && nvcc --version | head -3 || echo "WARN: nvcc not on PATH"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

mkdir -p "$ROOT"
uv python install 3.12
if [[ -x "$VENV/bin/python" ]]; then
    echo "Reusing existing venv $VENV"
else
    uv venv --python 3.12 "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

uv pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install "vllm==0.12.0"

uv pip install "transformers[hf_xet]==4.57.6" accelerate datasets peft hf-transfer \
    huggingface_hub "numpy<2.3" "pyarrow>=15.0.0" pandas \
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
    "ray>=2.40" codetiming hydra-core pylatexenc wandb dill pybind11 liger-kernel \
    mathruler math-verify qwen-vl-utils \
    "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" \
    "pydantic>=2.9" "grpcio>=1.62.1" "trl==0.14.0"

uv pip uninstall flash_attn -y 2>/dev/null || true
uv pip install flashinfer-python==0.3.1 || echo "WARN: flashinfer missing — continuing"

if [[ ! -d "$VERL_HOME/.git" ]]; then
    bash "$HOPD_HOME/setup/bootstrap_verl.sh" "$VERL_HOME"
fi
uv pip install -e "$VERL_HOME"

# veRL's setup.py still declares numpy<2.0.0 and drags numpy down to 1.26, but
# vLLM's worker imports numba (needs numpy<=2.2) and scipy needs numpy>=2.0.
# Only 2.2.x satisfies all three, so pin it AFTER the editable veRL install.
uv pip install "numpy==2.2.6"

python - <<'PY'
import numpy, torch, vllm
print("numpy", numpy.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count())
print("vllm", vllm.__version__)
import numba
print("numba", numba.__version__)
from vllm.v1.spec_decode.ngram_proposer import NgramProposer  # numba/numpy tripwire
print("vllm worker imports OK")
PY

echo "OK. source $VENV/bin/activate"
echo "    export HOPD_VENV=$VENV HOPD_VERL_HOME=$VERL_HOME HOPD_VERL_BRANCH=${HOPD_VERL_BRANCH:-hopd}"
echo "    export HOPD_ENV_SCRIPT=$HOPD_HOME/setup/env_vars.ls6.sh"
