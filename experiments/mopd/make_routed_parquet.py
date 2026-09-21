#!/usr/bin/env python3
"""Split H-OPD parquet for veRL routed MOPD (one teacher per row, not H-OPD mix)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# HuggingFace `datasets` materializes every Arrow struct field, including
# nulls. If this schema kept the source `image` field, process_image would
# still see both `bytes` and `image` even after we dropped nulls in pandas.
_IMAGE_STRUCT = pa.struct(
    [
        pa.field("bytes", pa.binary()),
        pa.field("max_pixels", pa.int64()),
        pa.field("min_pixels", pa.int64()),
    ]
)
_IMAGES_TYPE = pa.list_(_IMAGE_STRUCT)


def _as_bytes(raw) -> bytes:
    if isinstance(raw, (bytes, bytearray, memoryview)):
        return bytes(raw)
    if hasattr(raw, "tobytes"):
        return raw.tobytes()
    return bytes(raw)


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
    if "bytes" not in out:
        return None
    return {
        "bytes": _as_bytes(out["bytes"]),
        "max_pixels": int(out.get("max_pixels", max_pixels)),
        "min_pixels": int(out.get("min_pixels", min_pixels)),
    }


def _clean_images(images, max_pixels: int, min_pixels: int):
    if images is None:
        return None
    try:
        return [c for c in (_clean_image(i, max_pixels, min_pixels) for i in images) if c is not None]
    except TypeError:
        return images


def _write_parquet(df: pd.DataFrame, out: Path) -> None:
    """Write with an images schema that cannot resurrect a null `image` field."""
    if "images" in df.columns:
        other = df.drop(columns=["images"])
        table = pa.Table.from_pandas(other, preserve_index=False)
        table = table.append_column(
            "images",
            pa.array([None if im is None else im for im in df["images"]], type=_IMAGES_TYPE),
        )
    else:
        table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out, row_group_size=min(1024, max(len(df), 1)))
    if "images" in table.column_names:
        print(f"arrow images: {table.schema.field('images').type}")


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
    p.add_argument(
        "--keep_all_images",
        action="store_true",
        help="Keep images on text-routed rows too (paper H-OPD mix). Default: drop images on odd/text rows.",
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
        if isinstance(first, (list, tuple)) and isinstance(first[0], dict):
            print(f"image struct fields in: {sorted(first[0])}")
        cleaned = [
            None
            if (ds == "hopd_text" and not args.keep_all_images)
            else _clean_images(im, args.image_max_pixels, args.image_min_pixels)
            for ds, im in zip(df["data_source"], df["images"])
        ]
        df["images"] = cleaned
        first = next((im for im in cleaned if im is not None and len(im) > 0), None)
        if isinstance(first, (list, tuple)) and isinstance(first[0], dict):
            print(f"image struct fields out: {sorted(first[0])} max_pixels={args.image_max_pixels}")
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_parquet(df, out)
    n_vl = (df["data_source"] == "hopd_vl").sum()
    n_text = (df["data_source"] == "hopd_text").sum()
    print(f"Wrote {out} n={n} vl={n_vl} text={n_text}")
    if "images" in df.columns:
        kept = sum(1 for im in df["images"] if im is not None and len(im) > 0)
        print(f"rows carrying images: {kept} (expect {n if args.keep_all_images else n_vl})")


if __name__ == "__main__":
    main()
