#!/usr/bin/env python3
"""Accuracy probe that never feeds the MOPD loss.

veRL scores every rollout, and `default_compute_score` only knows public dataset
names, so custom `data_source` values raise NotImplementedError.

`score` stays 0.0 on purpose: routed MOPD learns from teacher disagreement
(`distillation.use_task_rewards=False`), and a non-zero score would change
advantages. `acc` rides along in the dumped rollouts and in validation metrics,
which is how we watch whether the student is actually improving.
"""
from __future__ import annotations

import os
from typing import Any, Optional


def _grade(solution_str: str, ground_truth: Any) -> float:
    if ground_truth is None:
        return 0.0
    gt = str(ground_truth).strip()
    if not gt:
        return 0.0
    try:
        from mathruler.grader import extract_boxed_content, grade_answer

        answer = extract_boxed_content(solution_str)
        if answer and answer != "None":
            return 1.0 if grade_answer(answer, gt) else 0.0
    except Exception:  # noqa: BLE001 — grading must never break training
        pass
    tail = solution_str[-200:].strip().lower()
    return 1.0 if gt.lower() and gt.lower() in tail else 0.0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    **kwargs: Any,
) -> dict:
    acc = _grade(solution_str, ground_truth)
    score = acc if os.environ.get("HOPD_TASK_REWARD", "0") == "1" else 0.0
    return {
        "score": score,
        "acc": acc,
        "is_vl": 1.0 if str(data_source) == "hopd_vl" else 0.0,
        "resp_chars": float(len(solution_str)),
    }
