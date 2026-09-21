#!/usr/bin/env python3
"""No-GPU check: even/odd routing + text rows drop images."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def route(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    df["data_source"] = ["hopd_vl" if i % 2 == 0 else "hopd_text" for i in range(n)]
    if "images" in df.columns:
        df["images"] = [
            None if ds == "hopd_text" else im for ds, im in zip(df["data_source"], df["images"])
        ]
    return df


def test_tiny() -> None:
    # Mirror the real parquet: a list of structs carrying both bytes and a null
    # image field, which process_image rejects until make_routed_parquet cleans it.
    raw = [{"bytes": b"\x89PNG-fake", "image": None}]
    df = pd.DataFrame({"prompt": ["a", "b", "c", "d"], "images": [list(raw) for _ in range(4)]})
    out = route(df)
    assert list(out["data_source"]) == ["hopd_vl", "hopd_text", "hopd_vl", "hopd_text"]
    assert out.loc[0, "images"][0]["bytes"] == raw[0]["bytes"]
    assert out.loc[1, "images"] is None
    assert out.loc[3, "images"] is None
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src.parquet"
        dst = Path(td) / "out.parquet"
        df.to_parquet(src, index=False)
        # call the CLI
        import subprocess

        r = subprocess.run(
            [
                sys.executable,
                str(ROOT / "experiments/mopd/make_routed_parquet.py"),
                "--src",
                str(src),
                "--out",
                str(dst),
                "--max_rows",
                "2",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        got = pd.read_parquet(dst)
        assert len(got) == 2
        assert list(got["data_source"]) == ["hopd_vl", "hopd_text"]
        vl_img = got["images"].iloc[0][0]
        assert set(vl_img) == {"bytes", "max_pixels", "min_pixels"}, sorted(vl_img)
        assert vl_img["max_pixels"] == 256 * 32 * 32
        text_img = got["images"].iloc[1]
        assert text_img is None or len(text_img) == 0
        print(r.stdout.strip())
    print("[ ok ] mopd parquet routing (VL keeps a capped image, text drops it)")


if __name__ == "__main__":
    test_tiny()
