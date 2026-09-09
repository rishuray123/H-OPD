#!/bin/bash
set -euo pipefail

HOPD_HOME="${HOPD_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck disable=SC1091
source "$HOPD_HOME/config.sh"

PASS=0; FAIL=0; WARN=0
ok()   { echo "  [ ok ] $*"; PASS=$((PASS+1)); }
bad()  { echo "  [FAIL] $*"; FAIL=$((FAIL+1)); }
warn() { echo "  [warn] $*"; WARN=$((WARN+1)); }

echo "HOPD_HOME      $HOPD_HOME"
echo "HOPD_VERL_HOME $HOPD_VERL_HOME"
echo "HOPD_VENV      ${HOPD_VENV:-<python on PATH>}"
echo "HOPD_DATA_ROOT $HOPD_DATA_ROOT"

if hopd_activate_env; then ok "env"; else bad "hopd_activate_env"; fi

if [[ -d "$HOPD_VERL_HOME/.git" ]]; then
    head="$(git -C "$HOPD_VERL_HOME" rev-parse HEAD)"
    br="$(git -C "$HOPD_VERL_HOME" rev-parse --abbrev-ref HEAD)"
    ok "veRL $br $head"
    if [[ "$br" != "$HOPD_VERL_BRANCH" ]]; then
        warn "expected branch $HOPD_VERL_BRANCH (have $br)"
    fi
    if ! git -C "$HOPD_VERL_HOME" merge-base --is-ancestor "$HOPD_VERL_COMMIT" HEAD; then
        bad "HEAD does not contain ls6 pin $HOPD_VERL_COMMIT"
    fi
else
    bad "no git checkout at $HOPD_VERL_HOME"
fi

for f in verl/trainer/distillation/losses.py \
         verl/trainer/config/distillation/distillation.yaml; do
    [[ -f "$HOPD_VERL_HOME/$f" ]] && ok "$f" || bad "missing $f"
done

python3 - <<'PY'
import importlib, sys
need = ["torch", "transformers", "vllm", "ray", "hydra", "pandas", "verl"]
fail = 0
for m in need:
    try:
        mod = importlib.import_module(m)
        print(f"  [ ok ] import {m} {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"  [FAIL] import {m}: {e}")
        fail = 1
sys.exit(fail)
PY
[[ $? -eq 0 ]] && ok "python imports" || bad "python imports"

n=$(find "$HOPD_DATA_ROOT" -name '*.parquet' 2>/dev/null | wc -l | tr -d ' ')
if [[ "$n" -gt 0 ]]; then
    ok "$n parquet files in $HOPD_DATA_ROOT"
else
    warn "no parquet yet — python download_data.py --out_dir \$HOPD_DATA_ROOT"
fi

echo "pass=$PASS fail=$FAIL warn=$WARN"
[[ "$FAIL" -eq 0 ]]
