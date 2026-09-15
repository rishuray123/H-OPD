#!/usr/bin/env python3
"""Plots from a token-prob dump (student vs teacher top-k)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PAD = -1


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - np.nanmax(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.nansum(e, axis=axis, keepdims=True)


def _valid_mask(length: np.ndarray, lmax: int) -> np.ndarray:
    t = np.arange(lmax)[None, :]
    return t < length[:, None]


def load_dump(dump_dir: Path) -> tuple[dict, np.lib.npyio.NpzFile]:
    meta = json.loads((dump_dir / "meta.json").read_text())
    tensors = np.load(dump_dir / "tensors.npz")
    return meta, tensors


def summarize(meta: dict, z) -> dict:
    length = z["length"]
    lmax = z["response_ids"].shape[1]
    mask = _valid_mask(length, lmax)
    stu_lp = z["student_token_lp"]
    tea_lp = z["teacher_vl_token_lp"]
    stu_ids = z["student_topk_ids"]
    tea_ids = z["teacher_vl_topk_ids"]
    stu_tk = z["student_topk_lp"]
    tea_tk = z["teacher_vl_topk_lp"]

    pos_mean_stu = []
    pos_mean_tea = []
    pos_ent_stu = []
    pos_ent_tea = []
    pos_jac = []
    categories = []
    max_pos = int(min(length.max(), 48))
    for t in range(max_pos):
        m = mask[:, t]
        if not np.any(m):
            break
        categories.append(str(t))
        pos_mean_stu.append(float(np.nanmean(stu_lp[m, t])))
        pos_mean_tea.append(float(np.nanmean(tea_lp[m, t])))
        ps = _softmax(stu_tk[m, t], axis=-1)
        pt = _softmax(tea_tk[m, t], axis=-1)
        pos_ent_stu.append(float(np.mean(-np.nansum(ps * np.log(np.clip(ps, 1e-12, 1)), axis=-1))))
        pos_ent_tea.append(float(np.mean(-np.nansum(pt * np.log(np.clip(pt, 1e-12, 1)), axis=-1))))
        jac = []
        for i in np.where(m)[0]:
            a = set(int(x) for x in stu_ids[i, t] if x != PAD)
            b = set(int(x) for x in tea_ids[i, t] if x != PAD)
            u = a | b
            jac.append(len(a & b) / len(u) if u else 0.0)
        pos_jac.append(float(np.mean(jac)))

    gap = (tea_lp - stu_lp)[mask]
    chosen_in_teacher_top1 = float(
        np.mean(z["teacher_vl_topk_ids"][:, :, 0][mask] == z["response_ids"][mask])
    )
    return {
        "backend": meta.get("backend"),
        "n_pairs": int(length.shape[0]),
        "n_tokens": int(mask.sum()),
        "mean_student_token_lp": float(np.nanmean(stu_lp[mask])),
        "mean_teacher_token_lp": float(np.nanmean(tea_lp[mask])),
        "mean_teacher_minus_student_lp": float(np.nanmean(gap)),
        "frac_student_token_is_teacher_top1": chosen_in_teacher_top1,
        "mean_topk_jaccard": float(np.mean(pos_jac)) if pos_jac else 0.0,
        "position": {
            "categories": categories,
            "student_token_lp": pos_mean_stu,
            "teacher_token_lp": pos_mean_tea,
            "student_topk_entropy": pos_ent_stu,
            "teacher_topk_entropy": pos_ent_tea,
            "topk_jaccard": pos_jac,
        },
        "meta": meta,
    }


def plot_pngs(summary: dict, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos = summary["position"]
    x = np.arange(len(pos["categories"]))

    if not pos["categories"]:
        print("WARN: no token positions to plot")
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    ax = axes[0, 0]
    ax.plot(x, pos["student_token_lp"], label="student")
    ax.plot(x, pos["teacher_token_lp"], label="teacher VL")
    ax.set_title("Mean logprob of student token vs position")
    ax.set_xlabel("Response token index")
    ax.set_ylabel("Mean log P(y_t)")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(x, pos["student_topk_entropy"], label="student")
    ax.plot(x, pos["teacher_topk_entropy"], label="teacher VL")
    ax.set_title("Mean entropy of renormalized top-k vs position")
    ax.set_xlabel("Response token index")
    ax.set_ylabel("Entropy (nats)")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(x, pos["topk_jaccard"])
    ax.set_title("Mean Jaccard overlap of student vs teacher top-k")
    ax.set_xlabel("Response token index")
    ax.set_ylabel("Jaccard")
    ax.set_ylim(0, 1)

    ax = axes[1, 1]
    ax.axis("off")
    lines = [
        f"backend: {summary['backend']}",
        f"pairs: {summary['n_pairs']}",
        f"tokens: {summary['n_tokens']}",
        f"mean student lp: {summary['mean_student_token_lp']:.3f}",
        f"mean teacher lp: {summary['mean_teacher_token_lp']:.3f}",
        f"mean Δ (T-S): {summary['mean_teacher_minus_student_lp']:.3f}",
        f"teacher top-1 = student token: {summary['frac_student_token_is_teacher_top1']:.3f}",
        f"mean top-k Jaccard: {summary['mean_topk_jaccard']:.3f}",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=9)
    fig.suptitle("Token-prob dump (offline; not H-OPD training)")
    fig.tight_layout()
    fig.savefig(out_dir / "token_prob_overview.png", dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dump_dir", default="")
    args = p.parse_args()
    if args.dump_dir:
        dump_dir = Path(args.dump_dir)
    else:
        root = os_data_root() / "token_prob_dump"
        runs = sorted(root.glob("run_*"))
        if not runs:
            raise SystemExit(f"no dumps under {root}")
        dump_dir = runs[-1]

    meta, z = load_dump(dump_dir)
    summary = summarize(meta, z)
    (dump_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_pngs(summary, dump_dir)
    print(f"Wrote {dump_dir / 'summary.json'}")
    print(f"Wrote {dump_dir / 'token_prob_overview.png'}")


def os_data_root() -> Path:
    import os

    return Path(os.environ.get("HOPD_DATA_ROOT", "scratch/data"))


if __name__ == "__main__":
    main()
