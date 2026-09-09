#!/usr/bin/env python3
"""Download H-OPD parquet from Hugging Face onto scratch."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPO = "qixiangbupt/H-OPD-data"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=os.environ.get("HOPD_DATA_ROOT", "scratch/data"))
    p.add_argument("--repo", default=DEFAULT_REPO)
    args = p.parse_args()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=args.repo, repo_type="dataset", local_dir=str(out))
    parquets = sorted(out.rglob("*.parquet"))
    print(f"Wrote {len(parquets)} parquet files under {out}")
    for f in parquets:
        print(f"  {f.relative_to(out)}")
    if parquets:
        import pandas as pd

        sample = parquets[0]
        df = pd.read_parquet(sample)
        print(f"\nSchema ({sample.name}): {list(df.columns)}  n={len(df)}")


if __name__ == "__main__":
    main()
