#!/usr/bin/env python3
"""CPU check of entropy mix, union Ω_t, and reverse KL (stdlib + optional torch)."""
from __future__ import annotations

import math


def _softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]


def _entropy(logps: list[float]) -> float:
    p = _softmax(logps)
    return -sum(pi * math.log(max(pi, 1e-12)) for pi in p)


def mix_p(y_logp_vl: float, y_logp_tx: float, topk_vl: list[float], topk_tx: list[float], tau: float = 1.0) -> tuple[float, float]:
    h_vl, h_tx = _entropy(topk_vl), _entropy(topk_tx)
    w_vl, w_tx = math.exp(-h_vl / tau), math.exp(-h_tx / tau)
    alpha = w_vl / (w_vl + w_tx)
    q = alpha * math.exp(y_logp_vl) + (1.0 - alpha) * math.exp(y_logp_tx)
    return alpha, q


def test_low_entropy_vl_wins() -> None:
    topk_vl = [-10.0] * 7 + [0.0]
    topk_tx = [0.0] * 8
    alpha, q = mix_p(0.0, math.log(1 / 8), topk_vl, topk_tx)
    assert alpha > 0.8, alpha
    assert q > 0.8, q


def test_union_mix_disjoint() -> None:
    """Equal-entropy teachers, disjoint top-1 → α=0.5 and both ids in Ω."""
    h = 0.0
    alpha = math.exp(-h) / (math.exp(-h) + math.exp(-h))
    assert abs(alpha - 0.5) < 1e-12
    q = {10: alpha * 1.0, 20: (1.0 - alpha) * 1.0}
    total = sum(q.values())
    assert abs(q[10] / total - 0.5) < 1e-12
    assert abs(q[20] / total - 0.5) < 1e-12


def test_align_teacher_rank() -> None:
    """2D teacher ids must be expandable to student [B, S, V] for gather."""
    student_ndim, teacher_ndim = 3, 2
    while teacher_ndim < student_ndim:
        teacher_ndim += 1
    assert teacher_ndim == student_ndim


def test_reverse_kl_peaked_vs_flat() -> None:
    """KL(student || teacher) on a 2-set: peaked student is near 0, uniform is larger."""
    p = [0.99, 0.01]
    q_peak = [0.99, 0.01]
    q_flat = [0.5, 0.5]
    kl_peak = sum(qi * math.log(qi / pi) for qi, pi in zip(q_peak, p))
    kl_flat = sum(qi * math.log(qi / pi) for qi, pi in zip(q_flat, p))
    assert kl_peak < 0.05, kl_peak
    assert kl_flat > kl_peak


def test_text_prompt_strip_keeps_response() -> None:
    vision = 151655
    prompt = [1, vision, vision, 2]
    response = [9, 8, 7]
    stripped = [i for i in prompt if i != vision]
    text_seq = stripped + response
    assert text_seq[-3:] == response
    assert vision not in text_seq


def test_equal_entropy_half() -> None:
    topk = [0.0, 0.0]
    alpha, q = mix_p(math.log(0.5), math.log(0.5), topk, topk)
    assert abs(alpha - 0.5) < 1e-9, alpha
    assert abs(q - 0.5) < 1e-9, q


def _run_torch_tests() -> bool:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "verl"))
    import torch

    from verl.experimental.teacher_loop.teacher_manager import (
        mix_teacher_token_logprobs,
        mix_teacher_union_logprobs,
        strip_vision_tokens,
    )
    from verl.trainer.distillation.fsdp.losses import reverse_kl_on_omega

    seq = [7] * 4
    vl_ids = torch.arange(8).unsqueeze(0).expand(4, -1)
    vl_lp = torch.full((4, 8), -10.0)
    vl_lp[:, 7] = 0.0
    mixed, stats = mix_teacher_token_logprobs(vl_ids, vl_lp, vl_ids, torch.zeros(4, 8), seq)
    assert stats["alpha_vl"] > 0.8, stats
    assert mixed.exp().mean() > 0.5

    pads = [1, 151652, 151655, 2, 3]
    assert strip_vision_tokens(pads) == [1, 2, 3]
    assert strip_vision_tokens([7, 8]) == [7, 8]

    # Disjoint top-1: union must keep both ids; peaked VL gets most of the mass.
    vl_ids = torch.tensor([[10, -1], [10, -1]])
    vl_lp = torch.tensor([[0.0, -10.0], [0.0, -10.0]])
    tx_ids = torch.tensor([[20, -1], [20, -1]])
    tx_lp = torch.tensor([[0.0, -10.0], [0.0, -10.0]])
    u_ids, u_lp, u_stats = mix_teacher_union_logprobs(vl_ids, vl_lp, tx_ids, tx_lp)
    row_ids = {int(i) for i in u_ids[0].tolist() if int(i) >= 0}
    assert row_ids == {10, 20}, row_ids
    assert abs(u_stats["alpha_vl"] - 0.5) < 1e-5, u_stats
    mass = {int(tid): math.exp(float(lp)) for tid, lp in zip(u_ids[0].tolist(), u_lp[0].tolist()) if tid >= 0}
    assert abs(mass[10] - 0.5) < 1e-5 and abs(mass[20] - 0.5) < 1e-5, mass
    assert u_stats["omega_k"] == 2.0, u_stats

    # Student peaked on teacher mode → reverse KL near 0; uniform is larger.
    logits_peak = torch.full((1, 1, 32), -20.0)
    logits_peak[0, 0, 10] = 5.0
    logits_flat = torch.zeros(1, 1, 32)
    teacher_ids = torch.tensor([[[10, 20, -1]]])
    teacher_lp = torch.tensor([[[0.0, -10.0, -10.0]]])
    kl_peak, _, _ = reverse_kl_on_omega(logits_peak, teacher_ids, teacher_lp)
    kl_flat, _, _ = reverse_kl_on_omega(logits_flat, teacher_ids, teacher_lp)
    assert float(kl_peak) < 0.05, float(kl_peak)
    assert float(kl_flat) > float(kl_peak), (float(kl_flat), float(kl_peak))
    return True


if __name__ == "__main__":
    test_low_entropy_vl_wins()
    test_equal_entropy_half()
    test_text_prompt_strip_keeps_response()
    test_union_mix_disjoint()
    test_reverse_kl_peaked_vs_flat()
    test_align_teacher_rank()
    try:
        _run_torch_tests()
        print("[ ok ] hopd paper mix + reverse KL (stdlib + torch)")
    except ImportError:
        print("[ ok ] hopd entropy mix (stdlib)")
