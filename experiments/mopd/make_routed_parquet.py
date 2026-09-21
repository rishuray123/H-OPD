#!/usr/bin/env python3
"""Split H-OPD parquet for veRL routed MOPD (one teacher per row, not H-OPD mix)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _clean_image(img):
    """Drop null struct fields so verl's process_image accepts the dict.

    The parquet stores images as a struct that carries both `bytes` and `image`
    (one of them null). `process_image` asserts they are never both present, and
    a null field still counts as present, so every VL row gets dropped.
    """
    if not isinstance(img, dict):
        return img
    out = {k: v for k, v in img.items() if v is not None}
    if "bytes" in out and "image" in out:
        out.pop("image")
    return out


def _clean_images(images):
    if images is None:
        return None
    try:
        return [_clean_image(i) for i in images]
    except TypeError:
        return images


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--max_rows",
        type=int,
        default=0,
        help="If >0, take the first N rows of the real parquet (smoke). 0 = all rows.",
    )
    args = p.parse_args()
    src = Path(args.src)
    out = Path(args.out)
    df = pd.read_parquet(src)
    if df.empty:
        raise SystemExit(f"empty parquet: {src}")
    if args.max_rows and args.max_rows > 0:
        df = df.iloc[: args.max_rows]
    df = df.copy()
    n = len(df)
    df["data_source"] = ["hopd_vl" if i % 2 == 0 else "hopd_text" for i in range(n)]
    if "images" in df.columns:
        first = next((im for im in df["images"] if im is not None and len(im) > 0), None)
        if first is not None:
            sample = first[0]
            if isinstance(sample, dict):
                print(f"image struct fields: {sorted(sample)}")
        df["images"] = [
            None if ds == "hopd_text" else _clean_images(im)
            for ds, im in zip(df["data_source"], df["images"])
        ]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    n_vl = (df["data_source"] == "hopd_vl").sum()
    n_text = (df["data_source"] == "hopd_text").sum()
    print(f"Wrote {out} n={n} vl={n_vl} text={n_text}")
    if "images" in df.columns:
        kept = sum(1 for im in df["images"] if im is not None and len(im) > 0)
        print(f"rows carrying images: {kept} (expect {n_vl})")


if __name__ == "__main__":
    main()
