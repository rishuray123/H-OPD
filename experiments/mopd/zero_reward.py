#!/usr/bin/env python3
"""Zero task reward for routed MOPD.

veRL scores every rollout even when `distillation.use_task_rewards=False`, and
`default_compute_score` only knows public dataset names, so `hopd_vl` /
`hopd_text` raise NotImplementedError. The learning signal here is teacher
disagreement alone, so return a constant and keep the reward path inert.
"""
from __future__ import annotations

from typing import Any, Optional


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    **kwargs: Any,
) -> float:
    return 0.0
