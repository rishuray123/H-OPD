#!/usr/bin/env python3
"""Live metrics view for a MOPD run.

Reads the structured metrics written by veRL's `file` logger (falling back to
parsing the console log) and the per-step rollout dumps, then prints one row per
training step. Runs from any python3 — pandas/matplotlib are optional.

  python3 experiments/mopd/watch.py --follow
  python3 experiments/mopd/watch.py --csv metrics.csv --plot plots/
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

# (column header, substrings tried in order) — first key present in the run wins.
DEFAULT_VIEWS = [
    ("distill_loss", ["distillation/loss", "distillation/abs_loss"]),
    ("teacher_mass", ["distillation/teacher_mass"]),
    ("student_mass", ["distillation/student_mass"]),
    ("entropy", ["actor/entropy"]),
    ("grad_norm", ["actor/grad_norm"]),
    ("resp_len", ["response_length/mean"]),
    ("tok/s", ["perf/throughput"]),
    ("step_s", ["timing_s/step"]),
]
CONSOLE_RE = re.compile(r"step:(\d+)\s+-\s+(.*)")
PAIR_RE = re.compile(r"([\w./\-]+):(?:np\.float\d+\(|np\.int\d+\()?(-?[\d.eE+]+|nan|inf|-inf)\)?")


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


def load_metrics(run: str) -> dict[int, dict[str, float]]:
    steps: dict[int, dict[str, float]] = {}
    jsonl = os.path.join(run, "metrics.jsonl")
    if os.path.exists(jsonl):
        with open(jsonl, errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                data = rec.get("data", {})
                row = steps.setdefault(int(rec.get("step", 0)), {})
                for k, v in data.items():
                    if isinstance(v, (int, float)):
                        row[k] = float(v)
        if steps:
            return steps

    log = os.path.join(run, "training.log")
    if not os.path.exists(log):
        return steps
    with open(log, errors="replace") as f:
        for line in f:
            m = CONSOLE_RE.search(line)
            if not m:
                continue
            row = steps.setdefault(int(m.group(1)), {})
            for key, val in PAIR_RE.findall(m.group(2)):
                try:
                    row[key] = float(val)
                except ValueError:
                    continue
    return steps


def load_rollout_stats(run: str, subdir: str) -> dict[int, dict[str, float]]:
    stats: dict[int, dict[str, float]] = {}
    for path in glob.glob(os.path.join(run, subdir, "*.jsonl")):
        try:
            step = int(os.path.splitext(os.path.basename(path))[0])
        except ValueError:
            continue
        acc: list[float] = []
        vl: list[float] = []
        alpha: list[float] = []
        with open(path, errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec.get("acc"), (int, float)):
                    acc.append(float(rec["acc"]))
                if isinstance(rec.get("is_vl"), (int, float)):
                    vl.append(float(rec["is_vl"]))
                if isinstance(rec.get("alpha_vl"), (int, float)):
                    alpha.append(float(rec["alpha_vl"]))
        if acc or alpha:
            stats[step] = {"n": float(max(len(acc), len(alpha)))}
            if acc:
                stats[step]["acc"] = sum(acc) / len(acc)
            if vl:
                stats[step]["vl_frac"] = sum(vl) / len(vl)
            if alpha:
                stats[step]["alpha_vl"] = sum(alpha) / len(alpha)
    return stats


def pick_columns(steps: dict[int, dict[str, float]], extra: list[str], show_all: bool):
    present = {k for row in steps.values() for k in row}
    if show_all:
        return [(k, k) for k in sorted(present)]
    cols = []
    for header, candidates in DEFAULT_VIEWS:
        for cand in candidates:
            if cand in present:
                cols.append((header, cand))
                break
    for key in extra:
        if key in present:
            cols.append((key.split("/")[-1], key))
        else:
            matches = sorted(k for k in present if key in k)
            cols.extend((m, m) for m in matches)
    return cols


def fmt(v: float | None) -> str:
    if v is None:
        return "-"
    if v != v:
        return "nan"
    a = abs(v)
    if a >= 1000:
        return f"{v:.0f}"
    if a >= 1:
        return f"{v:.2f}"
    if a == 0:
        return "0"
    return f"{v:.3g}"


def render(run: str, args) -> str:
    steps = load_metrics(run)
    train_roll = load_rollout_stats(run, "rollouts")
    val_roll = load_rollout_stats(run, "val")
    cols = pick_columns(steps, args.keys.split(",") if args.keys else [], args.all)

    order = sorted(steps)
    if not order:
        return f"run {run}\nno steps logged yet — training is still starting up."

    show_alpha = any("alpha_vl" in v for v in train_roll.values())
    headers = ["step"] + [h for h, _ in cols]
    if train_roll:
        headers += ["acc", "n"]
        if show_alpha:
            headers += ["alpha_vl"]
    if val_roll:
        headers += ["val_acc"]

    rows = []
    for step in order[-args.last :]:
        row = [str(step)] + [fmt(steps[step].get(k)) for _, k in cols]
        if train_roll:
            r = train_roll.get(step, {})
            acc = r.get("acc")
            row += [f"{acc:.3f}" if acc is not None else "-", str(int(r["n"])) if "n" in r else "-"]
            if show_alpha:
                a = r.get("alpha_vl")
                row += [f"{a:.3f}" if a is not None else "-"]
        if val_roll:
            acc = val_roll.get(step, {}).get("acc")
            row += [f"{acc:.3f}" if acc is not None else "-"]
        rows.append(row)

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    lines = [
        f"run {run}   steps {order[0]}..{order[-1]}",
        "  ".join(h.rjust(w) for h, w in zip(headers, widths)),
        "  ".join("-" * w for w in widths),
    ]
    lines += ["  ".join(c.rjust(w) for c, w in zip(r, widths)) for r in rows]

    trends = []
    for header, key in cols:
        series = [steps[s][key] for s in order if key in steps[s]]
        if len(series) >= 2:
            delta = series[-1] - series[0]
            arrow = "up" if delta > 0 else ("down" if delta < 0 else "flat")
            trends.append(f"{header} {fmt(series[0])}->{fmt(series[-1])} ({arrow})")
    if trends:
        lines += ["", "first -> last:  " + " | ".join(trends)]
    if train_roll:
        accs = [train_roll[s]["acc"] for s in sorted(train_roll)]
        lines.append(f"rollout acc:    {fmt(accs[0])} -> {fmt(accs[-1])} over {len(accs)} steps")
    return "\n".join(lines)


def write_csv(run: str, path: str, args) -> None:
    import csv

    steps = load_metrics(run)
    roll = load_rollout_stats(run, "rollouts")
    keys = sorted({k for row in steps.values() for k in row})
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step"] + keys + ["rollout_acc", "rollout_n"])
        for step in sorted(steps):
            r = roll.get(step, {})
            w.writerow(
                [step]
                + [steps[step].get(k, "") for k in keys]
                + [r.get("acc", ""), r.get("n", "")]
            )
    print(f"wrote {path} ({len(steps)} steps, {len(keys)} metrics)")


def write_plots(run: str, out_dir: str, args) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib not installed: uv pip install matplotlib (or use --csv)")

    steps = load_metrics(run)
    roll = load_rollout_stats(run, "rollouts")
    cols = pick_columns(steps, args.keys.split(",") if args.keys else [], args.all)
    if roll:
        cols = cols + [("rollout_acc", "__acc__")]
    if not cols:
        sys.exit("nothing to plot yet")

    os.makedirs(out_dir, exist_ok=True)
    ncol = min(3, len(cols))
    nrow = (len(cols) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.8 * nrow), squeeze=False)
    for ax, (header, key) in zip(axes.flat, cols):
        if key == "__acc__":
            xs = sorted(roll)
            ys = [roll[s]["acc"] for s in xs]
        else:
            xs = [s for s in sorted(steps) if key in steps[s]]
            ys = [steps[s][key] for s in xs]
        ax.plot(xs, ys, marker="o", ms=3)
        ax.set_title(header, fontsize=9)
        ax.set_xlabel("step", fontsize=8)
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(cols) :]:
        ax.axis("off")
    fig.tight_layout()
    path = os.path.join(out_dir, "mopd_metrics.png")
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", help="run log dir (default: newest logs/mopd_*)")
    p.add_argument("--follow", action="store_true", help="refresh until interrupted")
    p.add_argument("--interval", type=float, default=15.0)
    p.add_argument("--last", type=int, default=15, help="rows of history to show")
    p.add_argument("--keys", default="", help="extra metric keys/substrings, comma separated")
    p.add_argument("--all", action="store_true", help="show every logged metric")
    p.add_argument("--csv", help="write a wide CSV of all metrics and exit")
    p.add_argument("--plot", help="write PNG curves into this dir and exit")
    args = p.parse_args()

    run = find_run(args.run)
    if args.csv:
        write_csv(run, args.csv, args)
        return
    if args.plot:
        write_plots(run, args.plot, args)
        return
    if not args.follow:
        print(render(run, args))
        return
    try:
        while True:
            os.system("clear" if os.name != "nt" else "cls")
            print(render(run, args))
            print(f"\nrefreshing every {args.interval:g}s — ctrl-c to stop")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
