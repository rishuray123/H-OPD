#!/usr/bin/env python3
"""Warm veRL's overlong-prompt index cache on CPU before Ray grabs the GPUs.

Later trainer launches reuse the cache and skip the tokenize pass. Rebuild with
HOPD_REFILTER=1 or VERL_DISABLE_FILTER_CACHE=1.
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", required=True, action="append")
    p.add_argument("--model", default=os.environ.get("STUDENT_MODEL", "Qwen/Qwen3-VL-2B-Instruct"))
    p.add_argument("--max_prompt", type=int, default=int(os.environ.get("MAX_PROMPT", 2048)))
    p.add_argument("--image_patch_size", type=int, default=16)
    p.add_argument("--workers", type=int, default=int(os.environ.get("FILTER_WORKERS", 4)))
    args = p.parse_args()

    import pyarrow.parquet as pq
    from omegaconf import OmegaConf

    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.dataset.rl_dataset import RLHFDataset, overlong_filter_cache_ready

    tokenizer = hf_tokenizer(args.model)
    processor = hf_processor(args.model, use_fast=True)
    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "image_key": "images",
            "image_patch_size": args.image_patch_size,
            "max_prompt_length": args.max_prompt,
            "truncation": "error",
            "filter_overlong_prompts": True,
            "filter_overlong_prompts_workers": max(1, args.workers),
            "shuffle": True,
            "seed": None,
            "max_samples": -1,
        }
    )
    for path in args.parquet:
        n_in = pq.ParquetFile(path).metadata.num_rows
        print(f"overlong filter: {path}  rows={n_in}  max_prompt={args.max_prompt}")
        if overlong_filter_cache_ready(
            [path],
            tokenizer=tokenizer,
            processor=processor,
            max_prompt_length=args.max_prompt,
            image_patch_size=args.image_patch_size,
            n_in=n_in,
            shuffle=True,
            seed=None,
        ):
            print(f"overlong filter cache ready — skip tokenize ({n_in} rows)")
            continue
        ds = RLHFDataset(data_files=path, tokenizer=tokenizer, processor=processor, config=config)
        kept = getattr(ds, "_filter_kept_indices", None)
        extra = f" kept={len(kept)}" if kept is not None else ""
        print(f"overlong filter done: {len(ds)} rows{extra}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("overlong filter failed", file=sys.stderr)
        raise
