# Lonestar6: x86_64, 3× A100-40GB, CUDA 12.8.

if command -v module >/dev/null 2>&1; then
    module reset 2>/dev/null || true
    module load gcc/13.2.0 2>/dev/null || module load gcc 2>/dev/null || true
    module load cuda/12.8 2>/dev/null || module load cuda/12.2 2>/dev/null || true
fi

if [[ -z "${CUDA_HOME:-}" ]]; then
    for _c in "${TACC_CUDA_DIR:-}" /opt/apps/cuda/12.8 /usr/local/cuda; do
        [[ -n "$_c" && -d "$_c" ]] && { export CUDA_HOME="$_c"; break; }
    done
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
    export PATH="$CUDA_HOME/bin:${PATH:-}"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

_cache_root="${HOPD_SCRATCH:-${SCRATCH:-$HOME}}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$_cache_root/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$_cache_root/.cache/uv-python}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export HF_HOME="${HF_HOME:-$_cache_root/.cache/huggingface}"
export TIKTOKEN_ENCODINGS_BASE="${TIKTOKEN_ENCODINGS_BASE:-$_cache_root/data/embeddings}"
export TMPDIR="${TMPDIR:-$_cache_root/tmp}"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TMPDIR"

export MAX_JOBS="${MAX_JOBS:-32}"
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
unset CFLAGS CXXFLAGS LDFLAGS
export USE_MEGATRON=0
export USE_SGLANG=0
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export VERL_USE_UV=0
