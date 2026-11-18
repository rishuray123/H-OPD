#!/usr/bin/env python3
"""Export validation rows: image + prompt + gold + student answer.

veRL val JSONL (skip_special_tokens) drops image pads and does not store pixels.
This joins each dump row to the eval parquet(s) by ground-truth + question text,
then writes an HTML gallery (and optional per-row folders).

  # on login (parquets live on $SCRATCH):
  python3 experiments/mopd/export_val_gallery.py \\
    --val-jsonl logs/mopd_3592945/val/80.jsonl \\
    --parquet $SCRATCH/hopd/data/mopd/mathvista_200_test_eval.parquet \\
    --parquet $SCRATCH/hopd/data/mopd/mathverse_200_test_eval.parquet \\
    --parquet $SCRATCH/hopd/data/mopd/mathvision_test_eval.parquet \\
    --out experiments/mopd/runs/mopd_3592945/val_gallery \\
    --n 12 --wrong 6 --right 6

  # laptop after scp'ing gallery/ (or run locally if parquets copied):
  open experiments/mopd/runs/mopd_3592945/val_gallery/index.html
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import re
import sys
from pathlib import Path


def _question_from_dump_input(s: str) -> str:
    s = (s or "").strip()
    # decoded chat: "user\n...\nassistant\n"
    m = re.search(r"(?is)^user\s*\n(.*?)(?:\nassistant\b|$)", s)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    return re.sub(r"\s+", " ", s).strip()


def _question_from_parquet_prompt(prompt) -> str:
    """Best-effort extract user text from veRL chat prompt column."""
    if prompt is None:
        return ""
    # list of chat messages
    if isinstance(prompt, list):
        parts = []
        for msg in prompt:
            if not isinstance(msg, dict):
                continue
            if str(msg.get("role", "")).lower() not in ("user", "human"):
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") in (None, "text") and c.get("text"):
                        parts.append(str(c["text"]))
                    elif isinstance(c, str):
                        parts.append(c)
            elif isinstance(content, str):
                parts.append(content)
        return re.sub(r"\s+", " ", " ".join(parts)).strip()
    if isinstance(prompt, str):
        return re.sub(r"\s+", " ", prompt).strip()
    return re.sub(r"\s+", " ", str(prompt)).strip()


def _gt(row) -> str:
    rm = row.get("reward_model")
    if isinstance(rm, dict) and rm.get("ground_truth") is not None:
        return str(rm["ground_truth"]).strip()
    if row.get("ground_truth") is not None:
        return str(row["ground_truth"]).strip()
    return ""


def _image_bytes(images) -> bytes | None:
    if images is None:
        return None
    items = images if isinstance(images, (list, tuple)) else [images]
    for im in items:
        if im is None:
            continue
        if isinstance(im, dict):
            b = im.get("bytes")
            if b is None:
                continue
            if isinstance(b, memoryview):
                b = b.tobytes()
            if isinstance(b, (bytes, bytearray)) and len(b) > 0:
                return bytes(b)
        elif isinstance(im, (bytes, bytearray)) and len(im) > 0:
            return bytes(im)
    return None


def _load_parquet_index(paths: list[Path]):
    import pandas as pd

    rows = []
    for path in paths:
        df = pd.read_parquet(path)
        for i in range(len(df)):
            r = df.iloc[i].to_dict()
            q = _question_from_parquet_prompt(r.get("prompt"))
            g = _gt(r)
            rows.append(
                {
                    "source": path.name,
                    "row": i,
                    "question": q,
                    "gt": g,
                    "key": (g, q.lower()),
                    "images": r.get("images"),
                    "data_source": r.get("data_source"),
                }
            )
    by_key: dict[tuple[str, str], list] = {}
    by_gt: dict[str, list] = {}
    for r in rows:
        by_key.setdefault(r["key"], []).append(r)
        if r["gt"]:
            by_gt.setdefault(r["gt"], []).append(r)
    return rows, by_key, by_gt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--val-jsonl", required=True, type=Path)
    p.add_argument("--parquet", action="append", required=True, type=Path, help="eval parquet (repeatable)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n", type=int, default=12, help="max samples in gallery")
    p.add_argument("--wrong", type=int, default=6, help="prefer this many acc=0")
    p.add_argument("--right", type=int, default=6, help="prefer this many acc=1")
    p.add_argument("--save-files", action="store_true", help="also write per-row png/txt under out/rows/")
    args = p.parse_args()

    dumps = [json.loads(l) for l in args.val_jsonl.read_text(errors="replace").splitlines() if l.strip()]
    _, by_key, by_gt = _load_parquet_index(args.parquet)

    wrong = [r for r in dumps if r.get("acc") == 0]
    right = [r for r in dumps if r.get("acc") == 1]
    chosen = wrong[: args.wrong] + right[: args.right]
    if len(chosen) < args.n:
        rest = [r for r in dumps if r not in chosen]
        chosen.extend(rest[: args.n - len(chosen)])
    chosen = chosen[: args.n]

    args.out.mkdir(parents=True, exist_ok=True)
    rows_dir = args.out / "rows"
    if args.save_files:
        rows_dir.mkdir(exist_ok=True)

    cards = []
    matched = 0
    for idx, rec in enumerate(chosen):
        q = _question_from_dump_input(str(rec.get("input", "")))
        g = "" if rec.get("gts") is None else str(rec["gts"]).strip()
        hit = None
        cands = by_key.get((g, q.lower()), [])
        if len(cands) == 1:
            hit = cands[0]
        elif len(cands) > 1:
            hit = cands[0]
        else:
            # fallback: unique gt
            gt_hits = by_gt.get(g, [])
            if len(gt_hits) == 1:
                hit = gt_hits[0]
            elif gt_hits:
                # pick longest question overlap
                ql = q.lower()
                hit = max(gt_hits, key=lambda r: len(_overlap(ql, r["question"].lower())))

        img_b64 = None
        meta = "unmatched parquet"
        if hit is not None:
            matched += 1
            meta = f"{hit['source']}#{hit['row']} ds={hit.get('data_source')}"
            raw = _image_bytes(hit.get("images"))
            if raw:
                # sniff jpeg/png
                mime = "image/jpeg"
                if raw[:8] == b"\x89PNG\r\n\x1a\n":
                    mime = "image/png"
                img_b64 = f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")
                if args.save_files:
                    ext = "png" if "png" in mime else "jpg"
                    (rows_dir / f"{idx:02d}.{ext}").write_bytes(raw)

        if args.save_files:
            (rows_dir / f"{idx:02d}.txt").write_text(
                f"acc={rec.get('acc')}\ngt={g}\nmeta={meta}\n\nPROMPT:\n{q}\n\nSTUDENT:\n{rec.get('output','')}\n",
                encoding="utf-8",
            )

        img_html = (
            f'<img src="{img_b64}" alt="sample" style="max-width:100%;max-height:360px;"/>'
            if img_b64
            else '<div class="missing">(no image matched)</div>'
        )
        acc = rec.get("acc")
        cards.append(
            f"""
<section class="card {'ok' if acc==1 else 'bad'}">
  <header>#{idx}  acc={html.escape(str(acc))}  gt={html.escape(g)}  <small>{html.escape(meta)}</small></header>
  {img_html}
  <h3>Prompt</h3>
  <pre>{html.escape(q)}</pre>
  <h3>Student</h3>
  <pre>{html.escape(str(rec.get('output','')))}</pre>
</section>"""
        )

    page = f"""<!doctype html>
<html><head><meta charset="utf-8"/>
<title>val gallery</title>
<style>
body{{font-family:ui-sans-serif,system-ui,sans-serif;margin:24px;background:#f6f6f4;color:#111}}
.card{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:16px;margin:0 0 20px}}
.card.ok{{border-left:6px solid #2a7}}
.card.bad{{border-left:6px solid #c33}}
pre{{white-space:pre-wrap;background:#fafafa;padding:10px;border-radius:6px;font-size:13px}}
.missing{{padding:40px;background:#eee;color:#666;text-align:center}}
header{{margin-bottom:10px;font-weight:600}}
small{{font-weight:400;color:#666}}
</style></head><body>
<h1>Validation gallery</h1>
<p>source <code>{html.escape(str(args.val_jsonl))}</code> · matched {matched}/{len(chosen)} · wrong≈{args.wrong} right≈{args.right}</p>
{''.join(cards)}
</body></html>"""
    index = args.out / "index.html"
    index.write_text(page, encoding="utf-8")
    print(f"wrote {index} ({matched}/{len(chosen)} images matched)")


def _overlap(a: str, b: str) -> str:
    # longest common substring light proxy: shared tokens
    ta, tb = set(a.split()), set(b.split())
    return " ".join(ta & tb)


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        sys.exit(f"need pandas/pyarrow in the env: {e}")
