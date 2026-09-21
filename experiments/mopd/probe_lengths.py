#!/usr/bin/env python3
"""CPU probe: how long are the processed prompts, and would they survive the filter?

Mirrors veRL's `doc2len` (same message build, same image expansion) but reports
the length per data_source and re-raises the real error instead of veRL's
"Error processing one of the samples, skipping..." — which is what made a whole
class of rows vanish silently. Run this after changing max_prompt_length or the
image pixel cap, rather than learning the answer from `filter dataset len` half
a GPU-hour into a run.

  python3 experiments/mopd/probe_lengths.py --parquet scratch/data/mopd/train_routed_96.parquet
"""
from __future__ import annotations

import argparse
import collections
import os
import traceback


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet", required=True)
    p.add_argument("--model", default=os.environ.get("STUDENT_MODEL", "Qwen/Qwen3-VL-2B-Instruct"))
    p.add_argument("--max_prompt", type=int, default=int(os.environ.get("MAX_PROMPT", 2048)))
    p.add_argument("--per_source", type=int, default=8, help="rows to measure per data_source")
    args = p.parse_args()

    from omegaconf import OmegaConf

    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.dataset.rl_dataset import RLHFDataset
    from verl.utils.dataset.vision_utils import process_image

    tokenizer = hf_tokenizer(args.model)
    processor = hf_processor(args.model, use_fast=True)
    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "image_key": "images",
            "image_patch_size": 16,
            "max_prompt_length": args.max_prompt,
            "truncation": "right",
            "filter_overlong_prompts": False,
        }
    )
    ds = RLHFDataset(data_files=args.parquet, tokenizer=tokenizer, processor=processor, config=config)
    df = ds.dataframe

    def measure(doc) -> int:
        messages = ds._build_messages(doc, key=ds.prompt_key)
        raw_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        images = doc.get(ds.image_key) or None
        if images:
            images = [process_image(im, image_patch_size=ds.image_patch_size) for im in images]
            return len(processor(text=[raw_prompt], images=images)["input_ids"][0])
        return len(processor.tokenizer(text=raw_prompt, add_special_tokens=False)["input_ids"])

    sources = list(df["data_source"]) if "data_source" in df.column_names else ["all"] * len(df)
    picked: dict[str, list[int]] = collections.defaultdict(list)
    for i, src in enumerate(sources):
        if len(picked[src]) < args.per_source:
            picked[src].append(i)

    lengths: dict[str, list[int]] = collections.defaultdict(list)
    failures: dict[str, int] = collections.Counter()
    for src, idxs in picked.items():
        for i in idxs:
            try:
                lengths[src].append(measure(df[i]))
            except Exception as e:  # noqa: BLE001 — the point is to surface it
                failures[src] += 1
                if failures[src] == 1:
                    print(f"\n--- {src} row {i} failed to process: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    print()

    print(f"\nparquet {args.parquet}  rows {len(df)}")
    print(f"model   {args.model}")
    print(f"budget  max_prompt_length={args.max_prompt}\n")
    for src in sorted(picked):
        vals = sorted(lengths[src])
        if not vals:
            print(f"{src:10s} all {failures[src]} sampled rows errored — see traceback above")
            continue
        over = sum(1 for v in vals if v > args.max_prompt)
        print(
            f"{src:10s} n={len(vals):3d}  min={vals[0]:6d}  median={vals[len(vals) // 2]:6d}  "
            f"max={vals[-1]:6d}  over budget: {over}/{len(vals)}  errors: {failures[src]}"
        )
    worst = max((v for vals in lengths.values() for v in vals), default=0)
    if worst:
        print(f"\nmeasured max {worst} tokens -> max_prompt_length >= {int(worst * 1.15) + 64} is safe")


if __name__ == "__main__":
    main()
