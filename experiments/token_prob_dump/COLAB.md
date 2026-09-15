# Colab smoke test

Lonestar job `3443317` can stay in queue. Colab only checks that dump/plot **code** runs. It is not the 3×A100 experiment.

## 1. Runtime

`Runtime → Change runtime type → T4 GPU` (or CPU for step 2 only).

## 2. Synthetic (no models, 30 seconds)

```python
!git clone --depth 1 https://github.com/rishuray123/H-OPD.git
%cd H-OPD
!pip -q install numpy matplotlib pandas pyarrow pillow

!python experiments/token_prob_dump/dump.py --backend synthetic \
  --out_dir /content/tokdump_syn --n_prompts 8 --n_samples 4 --max_new_tokens 32 --topk 8
!python experiments/token_prob_dump/plot.py --dump_dir /content/tokdump_syn

from IPython.display import Image
Image("/content/tokdump_syn/token_prob_overview.png")
```

If that PNG appears, dump + plot are fine.

## 3. Tiny HF path on T4 (optional)

Free Colab T4 is **16GB**. Do **not** load Qwen3-VL-4B. Use **2B as student and teacher** to test generate + teacher-force. One prompt, short response.

T4 has no bfloat16; current `dump.py` uses float16 there (A100 still uses bf16).

```python
!pip -q install -U "transformers>=4.57" accelerate torch torchvision torchaudio

import pandas as pd
from pathlib import Path
Path("/content/data").mkdir(exist_ok=True)
pd.DataFrame([{"prompt": "Write one sentence about a red apple.", "images": None}]).to_parquet("/content/data/tiny.parquet")

!python experiments/token_prob_dump/dump.py --backend hf \
  --train_file /content/data/tiny.parquet \
  --out_dir /content/tokdump_hf \
  --n_prompts 1 --n_samples 1 --max_new_tokens 32 --topk 8 \
  --student Qwen/Qwen3-VL-2B-Instruct \
  --teacher Qwen/Qwen3-VL-2B-Instruct

!python experiments/token_prob_dump/plot.py --dump_dir /content/tokdump_hf
```

First run downloads ~4GB of 2B weights. OOM → lower `--max_new_tokens` or use a Colab **A100** and then you can try `--teacher Qwen/Qwen3-VL-4B-Instruct`.

## What Colab does not prove

- Lonestar install (`uv`, vLLM 0.12, CUDA 12.8)
- Full 55k parquet / images
- 2B←4B on 40GB A100
- The queued Slurm job
