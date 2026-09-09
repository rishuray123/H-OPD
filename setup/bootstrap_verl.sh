#!/bin/bash
# Clone our veRL fork and check out the H-OPD experiment branch.
set -euo pipefail

HOPD_HOME="${HOPD_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck disable=SC1091
source "$HOPD_HOME/config.sh"

TARGET="${1:-$HOPD_VERL_HOME}"

echo "veRL repo   : $HOPD_VERL_REPO"
echo "veRL branch : $HOPD_VERL_BRANCH"
echo "veRL floor  : $HOPD_VERL_COMMIT  (ls6 stack; hopd must contain this)"
echo "target      : $TARGET"

if [[ -d "$TARGET/.git" ]]; then
    git -C "$TARGET" fetch --quiet origin "$HOPD_VERL_BRANCH" \
        || git -C "$TARGET" fetch --quiet origin
else
    git clone "$HOPD_VERL_REPO" "$TARGET"
    git -C "$TARGET" remote add upstream https://github.com/verl-project/verl.git 2>/dev/null || true
    git -C "$TARGET" fetch --quiet origin "$HOPD_VERL_BRANCH"
fi

if git -C "$TARGET" show-ref --verify --quiet "refs/remotes/origin/$HOPD_VERL_BRANCH"; then
    git -C "$TARGET" checkout -B "$HOPD_VERL_BRANCH" "origin/$HOPD_VERL_BRANCH"
else
    echo "ERROR: origin/$HOPD_VERL_BRANCH not found" >&2
    exit 1
fi

have="$(git -C "$TARGET" rev-parse HEAD)"
if ! git -C "$TARGET" merge-base --is-ancestor "$HOPD_VERL_COMMIT" HEAD; then
    echo "ERROR: $HOPD_VERL_BRANCH ($have) does not contain ls6 pin $HOPD_VERL_COMMIT" >&2
    echo "       refusing to install fork main / CUDA 13 stack" >&2
    exit 1
fi

for f in verl/trainer/distillation/losses.py \
         verl/trainer/config/distillation/distillation.yaml \
         verl/experimental/teacher_loop/teacher_manager.py; do
    if [[ ! -f "$TARGET/$f" ]]; then
        echo "ERROR: missing $f — this checkout has no OPD trainer" >&2
        exit 1
    fi
done

echo "OK. veRL at $TARGET  branch=$HOPD_VERL_BRANCH  HEAD=${have:0:8}"
