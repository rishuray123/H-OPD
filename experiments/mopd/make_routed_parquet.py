#!/usr/bin/env python3
"""Split H-OPD parquet for veRL routed MOPD (one teacher per row, not H-OPD mix)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    src = Path(args.src)
    out = Path(args.out)
    df = pd.read_parquet(src)
    if df.empty:
        raise SystemExit(f"empty parquet: {src}")
    df = df.copy()
    n = len(df)
    df["data_source"] = ["hopd_vl" if i % 2 == 0 else "hopd_text" for i in range(n)]
    if "images" in df.columns:
        df["images"] = [
            None if ds == "hopd_text" else im for ds, im in zip(df["data_source"], df["images"])
        ]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Wrote {out} n={n} vl={(df['data_source']=='hopd_vl').sum()} text={(df['data_source']=='hopd_text').sum()}")


if __name__ == "__main__":
    main()
