#!/usr/bin/env python3
"""CPU check of entropy-weighted teacher mix (stdlib only)."""
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


def test_equal_entropy_half() -> None:
    topk = [0.0, 0.0]
    alpha, q = mix_p(math.log(0.5), math.log(0.5), topk, topk)
    assert abs(alpha - 0.5) < 1e-9, alpha
    assert abs(q - 0.5) < 1e-9, q


if __name__ == "__main__":
    test_low_entropy_vl_wins()
    test_equal_entropy_half()
    try:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "verl"))
        import torch

        from verl.experimental.teacher_loop.teacher_manager import mix_teacher_token_logprobs

        seq = [7] * 4
        vl_ids = torch.arange(8).unsqueeze(0).expand(4, -1)
        vl_lp = torch.full((4, 8), -10.0)
        vl_lp[:, 7] = 0.0
        mixed, stats = mix_teacher_token_logprobs(vl_ids, vl_lp, vl_ids, torch.zeros(4, 8), seq)
        assert stats["alpha_vl"] > 0.8, stats
        assert mixed.exp().mean() > 0.5
        print("[ ok ] hopd entropy mix (stdlib + torch)")
    except ImportError:
        print("[ ok ] hopd entropy mix (stdlib)")
