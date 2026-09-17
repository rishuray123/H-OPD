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
    tea_vl = z["teacher_vl_token_lp"]
    has_text = "teacher_text_token_lp" in z.files
    tea_tx = z["teacher_text_token_lp"] if has_text else None
    stu_ids = z["student_topk_ids"]
    tea_ids = z["teacher_vl_topk_ids"]
    stu_tk = z["student_topk_lp"]
    tea_tk = z["teacher_vl_topk_lp"]
    tea_tx_ids = z["teacher_text_topk_ids"] if has_text else None
    tea_tx_tk = z["teacher_text_topk_lp"] if has_text else None

    pos_mean_stu, pos_mean_vl, pos_mean_tx = [], [], []
    pos_ent_stu, pos_ent_vl, pos_ent_tx = [], [], []
    pos_jac_vl, pos_jac_tx, pos_jac_teachers = [], [], []
    categories = []
    max_pos = int(min(length.max(), 48))
    for t in range(max_pos):
        m = mask[:, t]
        if not np.any(m):
            break
        categories.append(str(t))
        pos_mean_stu.append(float(np.nanmean(stu_lp[m, t])))
        pos_mean_vl.append(float(np.nanmean(tea_vl[m, t])))
        ps = _softmax(stu_tk[m, t], axis=-1)
        pt = _softmax(tea_tk[m, t], axis=-1)
        pos_ent_stu.append(float(np.mean(-np.nansum(ps * np.log(np.clip(ps, 1e-12, 1)), axis=-1))))
        pos_ent_vl.append(float(np.mean(-np.nansum(pt * np.log(np.clip(pt, 1e-12, 1)), axis=-1))))
        if has_text:
            pos_mean_tx.append(float(np.nanmean(tea_tx[m, t])))
            ptx = _softmax(tea_tx_tk[m, t], axis=-1)
            pos_ent_tx.append(float(np.mean(-np.nansum(ptx * np.log(np.clip(ptx, 1e-12, 1)), axis=-1))))
        jac_vl, jac_tx, jac_tt = [], [], []
        for i in np.where(m)[0]:
            a = set(int(x) for x in stu_ids[i, t] if x != PAD)
            b = set(int(x) for x in tea_ids[i, t] if x != PAD)
            u = a | b
            jac_vl.append(len(a & b) / len(u) if u else 0.0)
            if has_text:
                c = set(int(x) for x in tea_tx_ids[i, t] if x != PAD)
                u2 = a | c
                jac_tx.append(len(a & c) / len(u2) if u2 else 0.0)
                u3 = b | c
                jac_tt.append(len(b & c) / len(u3) if u3 else 0.0)
        pos_jac_vl.append(float(np.mean(jac_vl)))
        if has_text:
            pos_jac_tx.append(float(np.mean(jac_tx)))
            pos_jac_teachers.append(float(np.mean(jac_tt)))

    gap = (tea_vl - stu_lp)[mask]
    chosen_vl = float(np.mean(z["teacher_vl_topk_ids"][:, :, 0][mask] == z["response_ids"][mask]))
    out = {
        "backend": meta.get("backend"),
        "n_pairs": int(length.shape[0]),
        "n_tokens": int(mask.sum()),
        "mean_student_token_lp": float(np.nanmean(stu_lp[mask])),
        "mean_teacher_vl_token_lp": float(np.nanmean(tea_vl[mask])),
        "mean_teacher_minus_student_lp": float(np.nanmean(gap)),
        "frac_student_token_is_teacher_vl_top1": chosen_vl,
        "mean_topk_jaccard_student_vl": float(np.mean(pos_jac_vl)) if pos_jac_vl else 0.0,
        "has_text_teacher": has_text,
        "position": {
            "categories": categories,
            "student_token_lp": pos_mean_stu,
            "teacher_vl_token_lp": pos_mean_vl,
            "student_topk_entropy": pos_ent_stu,
            "teacher_vl_topk_entropy": pos_ent_vl,
            "topk_jaccard_student_vl": pos_jac_vl,
        },
        "meta": meta,
    }
    if has_text:
        out["mean_teacher_text_token_lp"] = float(np.nanmean(tea_tx[mask]))
        out["frac_student_token_is_teacher_text_top1"] = float(
            np.mean(z["teacher_text_topk_ids"][:, :, 0][mask] == z["response_ids"][mask])
        )
        out["mean_topk_jaccard_student_text"] = float(np.mean(pos_jac_tx)) if pos_jac_tx else 0.0
        out["mean_topk_jaccard_vl_text"] = float(np.mean(pos_jac_teachers)) if pos_jac_teachers else 0.0
        out["position"]["teacher_text_token_lp"] = pos_mean_tx
        out["position"]["teacher_text_topk_entropy"] = pos_ent_tx
        out["position"]["topk_jaccard_student_text"] = pos_jac_tx
        out["position"]["topk_jaccard_vl_text"] = pos_jac_teachers
    return out


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
    ax.plot(x, pos["teacher_vl_token_lp"], label="teacher VL")
    if "teacher_text_token_lp" in pos:
        ax.plot(x, pos["teacher_text_token_lp"], label="teacher text")
    ax.set_title("Mean logprob of student token vs position")
    ax.set_xlabel("Response token index")
    ax.set_ylabel("Mean log P(y_t)")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(x, pos["student_topk_entropy"], label="student")
    ax.plot(x, pos["teacher_vl_topk_entropy"], label="teacher VL")
    if "teacher_text_topk_entropy" in pos:
        ax.plot(x, pos["teacher_text_topk_entropy"], label="teacher text")
    ax.set_title("Mean entropy of renormalized top-k vs position")
    ax.set_xlabel("Response token index")
    ax.set_ylabel("Entropy (nats)")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(x, pos["topk_jaccard_student_vl"], label="student vs VL")
    if "topk_jaccard_student_text" in pos:
        ax.plot(x, pos["topk_jaccard_student_text"], label="student vs text")
        ax.plot(x, pos["topk_jaccard_vl_text"], label="VL vs text")
    ax.set_title("Mean Jaccard overlap of top-k")
    ax.set_xlabel("Response token index")
    ax.set_ylabel("Jaccard")
    ax.set_ylim(0, 1)
    ax.legend()

    ax = axes[1, 1]
    ax.axis("off")
    lines = [
        f"backend: {summary['backend']}",
        f"pairs: {summary['n_pairs']}",
        f"tokens: {summary['n_tokens']}",
        f"mean student lp: {summary['mean_student_token_lp']:.3f}",
        f"mean VL teacher lp: {summary['mean_teacher_vl_token_lp']:.3f}",
    ]
    if summary.get("has_text_teacher"):
        lines.append(f"mean text teacher lp: {summary['mean_teacher_text_token_lp']:.3f}")
        lines.append(f"Jaccard VL vs text: {summary['mean_topk_jaccard_vl_text']:.3f}")
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
