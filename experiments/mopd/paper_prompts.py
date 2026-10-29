#!/usr/bin/env python3
"""Paper tilde-x lives in the H-OPD parquet as a top-level `text_prompt` column.

Student / VL teacher keep `prompt` (question + `<image>`). The text teacher should
see `text_prompt`, which is already
question text + [IMAGE DESCRIPTION] + d.
"""
from __future__ import annotations

from typing import Any, Optional

IMAGE_DESCRIPTION_MARK = "[IMAGE DESCRIPTION]"


def _as_list(raw: Any) -> Any:
    if raw is None:
        return None
    if hasattr(raw, "tolist") and not isinstance(raw, (str, bytes, dict, list, tuple)):
        try:
            raw = raw.tolist()
        except Exception:  # noqa: BLE001
            return raw
    return raw


def _text_from_content(content: Any) -> str:
    content = _as_list(content)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts: list[str] = []
        for blk in content:
            blk = _as_list(blk)
            if isinstance(blk, dict):
                if blk.get("type") in (None, "text"):
                    parts.append(str(blk.get("text", blk.get("content", ""))))
            elif isinstance(blk, str):
                parts.append(blk)
        return "".join(parts)
    return str(content)


def messages_from_text_prompt(raw: Any) -> Optional[list[dict[str, str]]]:
    """Normalize parquet `text_prompt` to tokenizer chat messages (no images)."""
    raw = _as_list(raw)
    if raw is None or (isinstance(raw, float) and raw != raw):
        return None
    if isinstance(raw, str):
        text = raw.replace("<image>", "").replace("<video>", "").strip()
        return [{"role": "user", "content": text}] if text else None
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, (list, tuple)) or not raw:
        return None
    messages: list[dict[str, str]] = []
    for item in raw:
        item = _as_list(item)
        if not isinstance(item, dict):
            continue
        content = _text_from_content(item.get("content", "")).replace("<image>", "").replace("<video>", "")
        if not content.strip():
            continue
        messages.append({"role": str(item.get("role", "user")), "content": content})
    return messages or None


def text_prompt_has_description(raw: Any) -> bool:
    messages = messages_from_text_prompt(raw)
    if not messages:
        return False
    blob = "\n".join(m.get("content", "") for m in messages)
    return IMAGE_DESCRIPTION_MARK in blob
