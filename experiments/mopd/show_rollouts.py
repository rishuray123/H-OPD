#!/usr/bin/env python3
"""Print student rollouts saved by a MOPD run.

veRL dumps one JSONL per step into `rollouts/` (training) and `val/`
(validation). This renders them readably: image pad runs are collapsed and the
prompt is trimmed to its tail, so you can see the question and the full answer.

  python3 experiments/mopd/show_rollouts.py                 # newest step, 3 samples
  python3 experiments/mopd/show_rollouts.py --step 10 --n 5
  python3 experiments/mopd/show_rollouts.py --val --wrong   # failed val samples
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import textwrap

PAD_RE = re.compile(r"(<\|image_pad\|>|<\|video_pad\|>)+")


def find_run(explicit: str | None) -> str:
    if explicit:
        return os.path.abspath(explicit)
    root = os.environ.get("HOPD_LOG_DIR") or os.path.join(
        os.environ.get("HOPD_HOME", os.getcwd()), "logs"
    )
    latest = os.path.join(root, "mopd_latest")
    if os.path.isdir(latest):
        return os.path.realpath(latest)
    runs = sorted(glob.glob(os.path.join(root, "mopd_*")), key=os.path.getmtime)
    if not runs:
        sys.exit(f"no MOPD run directories under {root} (pass --run DIR)")
    return runs[-1]


def collapse(text: str) -> str:
    return PAD_RE.sub(lambda m: f"<{m.group(1)[2:-2]} x{len(m.group(0)) // len(m.group(1))}>", text)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", help="run log dir (default: newest logs/mopd_*)")
    p.add_argument("--step", help="step number, or 'latest' / 'all'", default="latest")
    p.add_argument("--n", type=int, default=3, help="samples per step")
    p.add_argument("--val", action="store_true", help="read val/ instead of rollouts/")
    p.add_argument("--wrong", action="store_true", help="only samples with acc == 0")
    p.add_argument("--right", action="store_true", help="only samples with acc == 1")
    p.add_argument("--grep", help="only samples whose prompt or response contains this")
    p.add_argument("--prompt_chars", type=int, default=600, help="prompt tail length (0 = all)")
    p.add_argument("--stats", action="store_true", help="per-step accuracy table only")
    args = p.parse_args()

    run = find_run(args.run)
    sub = "val" if args.val else "rollouts"
    files = sorted(
        glob.glob(os.path.join(run, sub, "*.jsonl")),
        key=lambda f: int(os.path.splitext(os.path.basename(f))[0])
        if os.path.splitext(os.path.basename(f))[0].isdigit()
        else 0,
    )
    if not files:
        sys.exit(f"no dumps in {os.path.join(run, sub)} — is trainer.{'validation' if args.val else 'rollout'}_data_dir set?")

    if args.step == "latest":
        files = files[-1:]
    elif args.step != "all":
        want = os.path.join(run, sub, f"{args.step}.jsonl")
        if not os.path.exists(want):
            sys.exit(f"no dump for step {args.step}; have {[os.path.basename(f) for f in files]}")
        files = [want]

    print(f"run {run}  ({sub})\n")
    for path in files:
        step = os.path.splitext(os.path.basename(path))[0]
        with open(path, errors="replace") as f:
            recs = [json.loads(line) for line in f if line.strip()]

        accs = [r["acc"] for r in recs if isinstance(r.get("acc"), (int, float))]
        header = f"=== step {step}: {len(recs)} samples"
        if accs:
            header += f", acc {sum(accs) / len(accs):.3f}"
        vl = [r["is_vl"] for r in recs if isinstance(r.get("is_vl"), (int, float))]
        if vl:
            header += f", vl {int(sum(vl))}/{len(vl)}"
        print(header + " ===")
        if args.stats:
            continue

        shown = 0
        for rec in recs:
            acc = rec.get("acc")
            if args.wrong and acc != 0:
                continue
            if args.right and acc != 1:
                continue
            blob = f"{rec.get('input', '')}\n{rec.get('output', '')}"
            if args.grep and args.grep.lower() not in blob.lower():
                continue
            if shown >= args.n:
                break
            shown += 1

            prompt = collapse(str(rec.get("input", "")))
            if args.prompt_chars and len(prompt) > args.prompt_chars:
                prompt = "...(trimmed)... " + prompt[-args.prompt_chars :]
            tags = [f"acc={acc}" if acc is not None else "", f"score={rec.get('score')}"]
            tags += ["vl" if rec.get("is_vl") else "text"] if "is_vl" in rec else []
            print(f"\n--- sample {shown}  {'  '.join(t for t in tags if t)}")
            print("PROMPT:")
            print(textwrap.indent(prompt.strip(), "  "))
            print("STUDENT:")
            print(textwrap.indent(str(rec.get("output", "")).strip(), "  "))
            gt = rec.get("gts")
            if gt not in (None, ""):
                print(f"GROUND TRUTH: {gt}")
        if shown == 0 and not args.stats:
            print("(no samples matched the filters)")
        print()


if __name__ == "__main__":
    main()
