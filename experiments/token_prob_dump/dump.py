#!/usr/bin/env python3
"""Dump student rollouts + teacher-forced top-k logprobs (not H-OPD training)."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PAD = -1


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def write_dump(
    out_dir: Path,
    *,
    meta: dict,
    prompt_id: np.ndarray,
    sample_i: np.ndarray,
    length: np.ndarray,
    response_ids: np.ndarray,
    models: dict[str, dict[str, np.ndarray]],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "prompt_id": prompt_id.astype(np.int32),
        "sample_i": sample_i.astype(np.int32),
        "length": length.astype(np.int32),
        "response_ids": response_ids.astype(np.int32),
    }
    for name, tensors in models.items():
        for key, arr in tensors.items():
            payload[f"{name}_{key}"] = arr
    np.savez_compressed(out_dir / "tensors.npz", **payload)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out_dir


def dump_synthetic(args: argparse.Namespace) -> Path:
    rng = np.random.default_rng(args.seed)
    n_prompts = args.n_prompts
    k_samp = args.n_samples
    n = n_prompts * k_samp
    lmax = args.max_new_tokens
    topk = args.topk
    vocab = 32000

    prompt_id = np.repeat(np.arange(n_prompts), k_samp)
    sample_i = np.tile(np.arange(k_samp), n_prompts)
    length = rng.integers(max(8, lmax // 4), lmax + 1, size=n, dtype=np.int32)
    response_ids = np.full((n, lmax), PAD, dtype=np.int32)
    for i, L in enumerate(length):
        response_ids[i, :L] = rng.integers(0, vocab, size=L)

    def pack_model(bias: float) -> dict[str, np.ndarray]:
        token_lp = np.full((n, lmax), np.nan, dtype=np.float32)
        topk_ids = np.full((n, lmax, topk), PAD, dtype=np.int32)
        topk_lp = np.full((n, lmax, topk), np.nan, dtype=np.float32)
        for i, L in enumerate(length):
            # Chosen-token logprob declines slightly with position (harder later tokens).
            pos = np.arange(L, dtype=np.float32)
            token_lp[i, :L] = rng.normal(-1.2 + bias, 0.45, size=L) - 0.008 * pos
            for t in range(L):
                ids = rng.integers(0, vocab, size=topk)
                raw = rng.normal(-3.5, 1.1, size=topk).astype(np.float32)
                if rng.random() < 0.65:
                    ids[rng.integers(0, topk)] = int(response_ids[i, t])
                    raw[0] = token_lp[i, t]
                order = np.argsort(-raw)
                topk_ids[i, t] = ids[order]
                topk_lp[i, t] = raw[order]
        return {
            "token_lp": token_lp,
            "topk_ids": topk_ids,
            "topk_lp": topk_lp,
        }

        student = pack_model(0.0)
    teacher = pack_model(0.35)
    teacher_text = pack_model(0.15)

    out = Path(args.out_dir)
    meta = {
        "backend": "synthetic",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_prompts": n_prompts,
        "n_samples": k_samp,
        "n_pairs": n,
        "max_new_tokens": lmax,
        "topk": topk,
        "student_model": "synthetic-student",
        "teachers": ["synthetic-teacher-vl", "synthetic-teacher-text"],
        "note": "Placeholder tensors to exercise dump/plot. Not Qwen3-VL.",
    }
    return write_dump(
        out,
        meta=meta,
        prompt_id=prompt_id,
        sample_i=sample_i,
        length=length,
        response_ids=response_ids,
        models={"student": student, "teacher_vl": teacher, "teacher_text": teacher_text},
    )


def _row_text_and_images(row: dict):
    from io import BytesIO

    from PIL import Image

    prompt = row.get("prompt", row.get("question", row.get("problem", "")))
    images = []
    raw = row.get("images", row.get("image"))
    if raw is None:
        return prompt, images
    if not isinstance(raw, (list, tuple)):
        raw = [raw]
    for item in raw:
        if item is None:
            continue
        if hasattr(item, "convert"):
            images.append(item.convert("RGB"))
            continue
        if isinstance(item, dict):
            b = item.get("bytes") or item.get("image")
            p = item.get("path")
            if isinstance(b, (bytes, bytearray)):
                images.append(Image.open(BytesIO(b)).convert("RGB"))
            elif p:
                images.append(Image.open(p).convert("RGB"))
            continue
        if isinstance(item, (bytes, bytearray)):
            images.append(Image.open(BytesIO(item)).convert("RGB"))
    return prompt, images


def _messages(prompt, images):
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
        return prompt
    if not isinstance(prompt, str):
        try:
            import numpy as np

            if isinstance(prompt, np.ndarray):
                prompt = prompt.tolist()
        except Exception:
            pass
        if not isinstance(prompt, str):
            prompt = json.dumps(prompt, default=str)
    content = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _teacher_force_topk(model, inputs, prompt_len: int, response_ids, topk: int, device):
    import torch
    import torch.nn.functional as F

    resp = response_ids.to(device).reshape(1, -1)
    L = int(resp.shape[1])
    if L == 0:
        empty = np.zeros((0,), dtype=np.float32)
        empty_i = np.zeros((0, topk), dtype=np.int32)
        return empty, empty_i, empty
    input_ids = inputs["input_ids"].to(device)
    attn = inputs["attention_mask"].to(device)
    full_ids = torch.cat([input_ids, resp], dim=1)
    full_attn = torch.cat([attn, torch.ones_like(resp)], dim=1)
    kw = {}
    for k, v in inputs.items():
        if k in ("input_ids", "attention_mask"):
            continue
        kw[k] = v.to(device) if hasattr(v, "to") else v
    with torch.no_grad():
        out = model(input_ids=full_ids, attention_mask=full_attn, **kw)
        # logits[t] predicts token t+1; response token j is predicted at index prompt_len-1+j
        sl = out.logits[:, prompt_len - 1 : prompt_len - 1 + L, :]
        logp = F.log_softmax(sl.float(), dim=-1)
        token_lp = logp.gather(-1, resp.unsqueeze(-1)).squeeze(-1)
        topk_lp, topk_ids = logp.topk(topk, dim=-1)
    return token_lp[0].cpu().numpy(), topk_ids[0].cpu().numpy(), topk_lp[0].cpu().numpy()


def _move_batch(inputs, device):
    out = {}
    for k, v in inputs.items():
        out[k] = v.to(device) if hasattr(v, "to") else v
    return out


def _load_vl(model_id: str, dtype, device):
    from transformers import AutoModelForImageTextToText

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True
        )
    except Exception:
        from transformers import Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True
        )
    return model.to(device).eval()


def _plain_text(prompt) -> str:
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
        c = prompt[0].get("content", "")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            parts = []
            for blk in c:
                if isinstance(blk, dict) and blk.get("type") == "text":
                    parts.append(str(blk.get("text", "")))
                elif isinstance(blk, str):
                    parts.append(blk)
            return "\n".join(parts)
        return json.dumps(c, default=str)
    if isinstance(prompt, str):
        return prompt
    return json.dumps(prompt, default=str)


def _load_lm(model_id: str, dtype, device):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True)
    print(f"loaded {model_id} on CPU, moving to {device}…", flush=True)
    model = model.to(device).eval()
    print(f"{model_id} on {device}", flush=True)
    return model


def dump_hf(args: argparse.Namespace) -> Path:
    import pandas as pd
    import torch
    from transformers import AutoProcessor

    parquet = Path(args.train_file)
    if not parquet.is_file():
        raise SystemExit(f"parquet not found: {parquet}")
    df = pd.read_parquet(parquet).head(args.n_prompts)
    if df.empty:
        raise SystemExit(f"parquet has 0 rows: {parquet}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # T4/Colab (sm75): float16. A100/LS6 (sm80+): bfloat16.
        major = torch.cuda.get_device_capability()[0]
        dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        dtype = torch.float32
    print(f"device={device} dtype={dtype}", flush=True)

    n = len(df) * args.n_samples
    lmax = args.max_new_tokens
    topk = args.topk

    prompt_id = np.zeros(n, dtype=np.int32)
    sample_i = np.zeros(n, dtype=np.int32)
    length = np.zeros(n, dtype=np.int32)
    response_ids = np.full((n, lmax), PAD, dtype=np.int32)

    def empty_pack():
        return {
            "token_lp": np.full((n, lmax), np.nan, dtype=np.float32),
            "topk_ids": np.full((n, lmax, topk), PAD, dtype=np.int32),
            "topk_lp": np.full((n, lmax, topk), np.nan, dtype=np.float32),
        }

    student_pack = empty_pack()
    teacher_vl_pack = empty_pack()
    teacher_text_pack = empty_pack()

    if device != "cuda":
        print("WARN: no CUDA; HF dump will be extremely slow on CPU", flush=True)

    processor = AutoProcessor.from_pretrained(args.student, trust_remote_code=True)
    pad_id = getattr(processor.tokenizer, "pad_token_id", None) or processor.tokenizer.eos_token_id
    student = _load_vl(args.student, dtype, device)

    pair = 0
    cached_inputs = []
    for pi, row in enumerate(df.to_dict(orient="records")):
        prompt, images = _row_text_and_images(row)
        messages = _messages(prompt, images)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        proc_images = images or None
        inputs = processor(text=[text], images=proc_images, return_tensors="pt", padding=True)
        prompt_len = int(inputs["input_ids"].shape[1])
        gen_kw = _move_batch(inputs, device)
        for si in range(args.n_samples):
            with torch.no_grad():
                gen = student.generate(
                    **gen_kw,
                    max_new_tokens=lmax,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=0.95,
                    pad_token_id=pad_id,
                )
            resp = gen[0, prompt_len:]
            L = min(int(resp.shape[0]), lmax)
            prompt_id[pair] = pi
            sample_i[pair] = si
            length[pair] = L
            response_ids[pair, :L] = resp[:L].detach().cpu().numpy()
            tlp, tids, tlpk = _teacher_force_topk(
                student, inputs, prompt_len, resp[:L].cpu(), topk, device
            )
            student_pack["token_lp"][pair, :L] = tlp[:L]
            student_pack["topk_ids"][pair, :L] = tids[:L]
            student_pack["topk_lp"][pair, :L] = tlpk[:L]
            cached_inputs.append((pair, inputs, prompt_len, resp[:L].cpu(), prompt))
            pair += 1
            print(f"student generate prompt={pi} sample={si} L={L}", flush=True)

    del student
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("loading VL teacher…", flush=True)
    teacher_vl = _load_vl(args.teacher, dtype, device)
    print(f"VL teacher on device, {len(cached_inputs)} pairs to score", flush=True)

    for pair_i, inputs, prompt_len, resp, _prompt in cached_inputs:
        L = int(resp.shape[0])
        tlp, tids, tlpk = _teacher_force_topk(teacher_vl, inputs, prompt_len, resp, topk, device)
        teacher_vl_pack["token_lp"][pair_i, :L] = tlp[:L]
        teacher_vl_pack["topk_ids"][pair_i, :L] = tids[:L]
        teacher_vl_pack["topk_lp"][pair_i, :L] = tlpk[:L]
        print(f"teacher_vl force pair={pair_i} L={L}", flush=True)

    del teacher_vl
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    from transformers import AutoTokenizer

    print("loading text teacher (CPU then GPU)…", flush=True)
    text_tok = AutoTokenizer.from_pretrained(args.teacher_text, trust_remote_code=True)
    teacher_text = _load_lm(args.teacher_text, dtype, device)
    print(f"text teacher on device, {len(cached_inputs)} pairs to score", flush=True)
    for n_done, (pair_i, _inputs, _plen, resp, prompt) in enumerate(cached_inputs):
        L = int(resp.shape[0])
        print(f"teacher_text start pair={pair_i} ({n_done+1}/{len(cached_inputs)}) L={L}", flush=True)
        user = _plain_text(prompt)
        messages = [{"role": "user", "content": user}]
        text = text_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tin = text_tok(text, return_tensors="pt")
        text_len = int(tin["input_ids"].shape[1])
        # Re-encode Y with the text tokenizer (do not reuse VL token ids).
        resp_text = processor.tokenizer.decode(resp.tolist(), skip_special_tokens=False)
        resp_ids = text_tok.encode(resp_text, add_special_tokens=False)
        if not resp_ids:
            print(f"teacher_text skip pair={pair_i} empty re-encode", flush=True)
            continue
        resp_t = torch.tensor(resp_ids[:lmax], dtype=torch.long)
        Lt = int(resp_t.shape[0])
        tlp, tids, tlpk = _teacher_force_topk(teacher_text, tin, text_len, resp_t, topk, device)
        nfill = min(L, Lt, tlp.shape[0])
        teacher_text_pack["token_lp"][pair_i, :nfill] = tlp[:nfill]
        teacher_text_pack["topk_ids"][pair_i, :nfill] = tids[:nfill]
        teacher_text_pack["topk_lp"][pair_i, :nfill] = tlpk[:nfill]
        print(f"teacher_text force pair={pair_i} L={nfill}", flush=True)

    del teacher_text
    out = Path(args.out_dir)
    meta = {
        "backend": "hf",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_prompts": int(len(df)),
        "n_samples": args.n_samples,
        "n_pairs": pair,
        "max_new_tokens": lmax,
        "topk": topk,
        "student_model": args.student,
        "teachers": [args.teacher, args.teacher_text],
        "train_file": str(parquet),
        "temperature": args.temperature,
        "device": device,
    }
    return write_dump(
        out,
        meta=meta,
        prompt_id=prompt_id[:pair],
        sample_i=sample_i[:pair],
        length=length[:pair],
        response_ids=response_ids[:pair],
        models={
            "student": {k: v[:pair] for k, v in student_pack.items()},
            "teacher_vl": {k: v[:pair] for k, v in teacher_vl_pack.items()},
            "teacher_text": {k: v[:pair] for k, v in teacher_text_pack.items()},
        },
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["synthetic", "hf"], default="synthetic")
    p.add_argument("--out_dir", default="")
    p.add_argument("--train_file", default="")
    p.add_argument("--n_prompts", type=int, default=32)
    p.add_argument("--n_samples", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--student", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--teacher", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--teacher_text", default="Qwen/Qwen3-4B-Instruct-2507")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(os.environ.get("HOPD_DATA_ROOT", "scratch/data"))
    if not args.out_dir:
        args.out_dir = str(data_root / "token_prob_dump" / f"run_{_now()}")
    if args.backend == "hf" and not args.train_file:
        hits = list(data_root.rglob("mmfine_reason_sampled_55k_text_prompt.parquet"))
        args.train_file = str(hits[0]) if hits else ""
    if args.backend == "synthetic":
        out = dump_synthetic(args)
    else:
        out = dump_hf(args)
    print(f"Wrote {out}")
    print(f"  {out / 'tensors.npz'}")
    print(f"  {out / 'meta.json'}")


if __name__ == "__main__":
    main()
