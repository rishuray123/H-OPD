#!/usr/bin/env python3
"""Why can't veRL import transformers? Prints the torch-visibility chain."""
from __future__ import annotations

import importlib.metadata as md
import importlib.util
import os
import sys

BAD = []


def line(k: str, v: object) -> None:
    print(f"{k:28} {v}")


print("== interpreter ==")
line("python", sys.executable)

print("\n== env gates read by transformers ==")
for k in ("USE_TORCH", "USE_TF", "USE_JAX", "TRANSFORMERS_OFFLINE", "HF_HOME"):
    line(k, os.environ.get(k, "<unset>"))

print("\n== torch ==")
try:
    import torch

    line("torch.__version__", torch.__version__)
    line("torch.cuda.is_available", torch.cuda.is_available())
    line("torch.cuda.device_count", torch.cuda.device_count())
except Exception as e:  # noqa: BLE001
    line("torch import FAILED", f"{type(e).__name__}: {e}")
    BAD.append("torch does not import")

line("find_spec('torch')", importlib.util.find_spec("torch") is not None)
try:
    line("metadata version torch", md.version("torch"))
except Exception as e:  # noqa: BLE001
    line("metadata version torch", f"FAILED {type(e).__name__}: {e}")
    BAD.append("torch dist-info metadata missing (transformers hides torch classes)")

print("\n== transformers ==")
try:
    import transformers

    line("transformers.__version__", transformers.__version__)
    line("transformers.__file__", transformers.__file__)
    from transformers.utils import import_utils as iu

    line("_torch_available", getattr(iu, "_torch_available", "<absent>"))
    line("_torch_version", getattr(iu, "_torch_version", "<absent>"))
    from transformers.utils import is_torch_available

    ok = is_torch_available()
    line("is_torch_available()", ok)
    if not ok:
        BAD.append("transformers.is_torch_available() is False -> PreTrainedModel etc. are hidden")
    for name in ("PreTrainedModel", "PretrainedConfig", "AutoModel", "MistralForSequenceClassification"):
        try:
            getattr(transformers, name)
            line(f"  {name}", "ok")
        except Exception as e:  # noqa: BLE001
            line(f"  {name}", f"MISSING {type(e).__name__}")
            BAD.append(f"transformers.{name} missing")
except Exception as e:  # noqa: BLE001
    line("transformers import FAILED", f"{type(e).__name__}: {e}")
    BAD.append("transformers does not import")

print("\n== numpy / scipy ==")
for pkg in ("numpy", "scipy"):
    try:
        m = __import__(pkg)
        line(pkg, m.__version__)
    except Exception as e:  # noqa: BLE001
        line(pkg, f"FAILED {type(e).__name__}: {e}")

print("\n== vllm / verl ==")
for mod in ("vllm", "verl"):
    try:
        m = __import__(mod)
        line(mod, getattr(m, "__version__", m.__file__))
    except Exception as e:  # noqa: BLE001
        line(mod, f"FAILED {type(e).__name__}: {e}")
        BAD.append(f"{mod} does not import")

print("\n== trainer entrypoint ==")
try:
    importlib.util.find_spec("verl.trainer.main_ppo")
    import verl.trainer.main_ppo  # noqa: F401

    line("verl.trainer.main_ppo", "imports ok")
except Exception as e:  # noqa: BLE001
    line("verl.trainer.main_ppo", f"FAILED {type(e).__name__}: {e}")
    BAD.append("verl.trainer.main_ppo does not import")

print()
if BAD:
    print("PROBLEMS:")
    for b in BAD:
        print(f"  - {b}")
    sys.exit(1)
print("doctor OK — trainer imports are healthy")
