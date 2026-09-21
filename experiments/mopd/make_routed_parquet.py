#!/usr/bin/env python3
"""Split H-OPD parquet for veRL routed MOPD (one teacher per row, not H-OPD mix)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _clean_image(img, max_pixels: int, min_pixels: int):
    """Normalize an image struct for verl's process_image / qwen_vl_utils.

    Two problems with the raw parquet:
      * the struct carries both `bytes` and `image` (one null) and
        `process_image` asserts they are never both present — a null field still
        counts as present, which silently drops every VL row;
      * at full resolution one image expands to more tokens than
        `max_prompt_length`, so the rows that survive get length-filtered.
        `max_pixels`/`min_pixels` are read per image by `fetch_image`.
    """
    if not isinstance(img, dict):
        return img
    out = {k: v for k, v in img.items() if v is not None}
    if "bytes" in out and "image" in out:
        out.pop("image")
    out.setdefault("max_pixels", max_pixels)
    out.setdefault("min_pixels", min_pixels)
    return out


def _clean_images(images, max_pixels: int, min_pixels: int):
    if images is None:
        return None
    try:
        return [_clean_image(i, max_pixels, min_pixels) for i in images]
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
    # One merged Qwen3-VL token covers a 32x32 patch, so this caps an image at
    # about 256 tokens and keeps VL prompts inside max_prompt_length.
    p.add_argument("--image_max_pixels", type=int, default=256 * 32 * 32)
    p.add_argument("--image_min_pixels", type=int, default=4 * 32 * 32)
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
        if isinstance(first, (list, tuple)) and isinstance(first[0], dict):
            print(f"image struct fields in: {sorted(first[0])}")
        df["images"] = [
            None if ds == "hopd_text" else _clean_images(im, args.image_max_pixels, args.image_min_pixels)
            for ds, im in zip(df["data_source"], df["images"])
        ]
        first = next((im for im in df["images"] if im is not None and len(im) > 0), None)
        if isinstance(first, (list, tuple)) and isinstance(first[0], dict):
            print(f"image struct fields out: {sorted(first[0])} max_pixels={args.image_max_pixels}")
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
