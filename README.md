# Room VLM

Fine-tune and deploy a vision-language model that classifies indoor rooms into exactly one of:

- bedroom
- living room
- bathroom
- kitchen

**Primary dataset:** NYU Depth V2 raw RGB video sequences (scene-level splits)  
**Primary model:** [`Qwen/Qwen3-VL-4B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)  
**Training method:** LoRA / QLoRA parameter-efficient supervised fine-tuning  

This repository does **not** train or download models when you open it. All steps are explicit commands.

Vercel is **not** used. A 4B VLM needs a long-lived GPU (or slow CPU) process; use Docker / local FastAPI + Gradio instead.

---

## Architecture

```
NYU Depth V2
     |
     v
Scene discovery
     |
     v
RGB frame extraction
     |
     v
Temporal sampling
     |
     v
Scene-level stratified split
     |
     +-------------------+
     |                   |
     v                   v
Zero-shot Qwen      QLoRA fine-tuning
baseline                  |
     |                    v
     |               validation
     |                    |
     +---------+----------+
               |
               v
         held-out test
               |
               v
         metrics/report
               |
               v
         FastAPI / Gradio
               |
        +------+------+
        |             |
      image          video
                      |
               frame sampling
                      |
               score aggregation
                      |
                      v
                 room label
```

---

## Why this model

Qwen3-VL-4B-Instruct is used because:

- it is a modern pretrained vision-language model
- 4B is practical for experimentation vs very large VLMs
- it supports image understanding and video/multiframe reasoning
- language output keeps this a genuine VLM task (not a CNN classifier)
- LoRA/QLoRA makes adaptation feasible without updating all parameters

A conventional image classifier would likely be cheaper for a four-class problem. This project intentionally studies **VLM adaptation**.

---

## Dataset workflow (leakage prevention)

NYU Depth V2 raw data is video. Example:

```
bedroom_0001/
  frame at t=0s
  frame at t=0.04s
  frame at t=0.08s
  ...
```

Adjacent frames are highly correlated. **Never randomly split individual frames.**

Instead:

1. Sample ~1 RGB frame per second (timestamp-based by default)
2. Assign each **physical scene** entirely to train, validation, or test
3. Example: all sampled frames from `bedroom_0001` stay in TRAIN; `bedroom_0014` may be TEST

This tests whether the model recognizes a bedroom it has **never seen**, not whether it memorizes nearby video frames.

Random frame splitting would produce inflated, invalid results.

---

## Model / training workflow

```
Pretrained Qwen3-VL
  +-- visual encoder (frozen by default)
  +-- multimodal components
  +-- language model
        |
        v
     add LoRA
        |
        v
Train a small fraction of parameters
        |
        v
visual evidence -> room category -> "bedroom"
```

Supervised target is **exactly** one label (no punctuation / explanation). Causal LM loss is masked so **only assistant answer tokens** contribute.

---

## Security (do not expose data or secrets)

Never commit:

- `.env` (use `.env.example` placeholders only)
- NYU raw/extracted/processed/metadata
- model weights / LoRA adapters (`.safetensors`, `.bin`, checkpoints)
- generated `outputs/` artifacts with room images

Run before every push:

```bash
make security-check
# or: python -m room_vlm.security_check
```

Private GitHub is not a substitute for `.gitignore`.

---

## Install

Python **3.11+**. CUDA is strongly recommended for QLoRA (`bitsandbytes`).

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

python -m pip install -U pip
pip install -e ".[dev]"
```

`transformers>=4.57` is required (`Qwen3VLForConditionalGeneration`).

### CUDA / QLoRA notes

- QLoRA (4-bit NF4) needs CUDA + bitsandbytes.
- If unsupported, training **will not silently switch**. Pass `--allow-bf16-lora` and/or set `quantization.enabled: false` for BF16 LoRA.
- CPU inference/training works but is very slow.

### Hardware guidance (estimated, not measured)

| GPU VRAM | Estimated guidance |
|----------|--------------------|
| 12 GB | QLoRA, batch 1, high grad accumulation, gradient checkpointing |
| 16 GB | QLoRA more comfortable; still batch 1–2 |
| 24 GB | Preferable for fewer OOMs / larger effective batch |
| 48 GB | Headroom for experiments (projector/vision toggles, larger batches) |

If OOM occurs, the trainer prints remediation suggestions (batch size, accumulation, checkpointing, 4-bit, image token limits).

---

## Command-line workflow

### 1. Download / place NYU (optional helper)

```bash
python scripts/download_nyu.py --dry-run
python scripts/download_nyu.py --dest data/raw/archives
```

Only bathrooms / bedrooms / kitchens / living rooms are targeted (not the full ~428 GB dump). Official `horatio.cs.nyu.edu` URLs only.

**Manual alternative:** download category zips from the [NYU Depth V2 page](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html), extract scene folders, then point `--source` at that tree.

### 2. Prepare dataset

```bash
python scripts/prepare_nyu.py --source /path/to/nyu/raw --config configs/base.yaml
```

Smoke test (fixtures, no NYU download):

```bash
python scripts/prepare_nyu.py --source tests/fixtures/nyu --config configs/base.yaml --output .tmp/room-vlm-test
```

### 3. Validate

```bash
python scripts/validate_dataset.py --config configs/base.yaml
```

### 4. Zero-shot baseline (unmodified base model)

```bash
python scripts/run_baseline.py --config configs/base.yaml
```

### 5. Fine-tune (LoRA/QLoRA)

```bash
python scripts/train.py --config configs/qlora.yaml
# If QLoRA unavailable:
python scripts/train.py --config configs/qlora.yaml --allow-bf16-lora
```

### 6. Evaluate held-out test

```bash
python scripts/evaluate.py --config configs/qlora.yaml --split test --checkpoint outputs/<run_name>/adapter
```

### 7. Predict

```bash
python scripts/predict_image.py path/to/image.jpg --checkpoint outputs/<run_name>/adapter
python scripts/predict_video.py room.mp4 --checkpoint outputs/<run_name>/adapter
```

### 8. API

```bash
uvicorn room_vlm.api.app:app --host 0.0.0.0 --port 8000
```

### 9. UI

```bash
python web/gradio_app.py
```

### Makefile

```bash
make install
make test
make security-check
make prepare-data SOURCE=tests/fixtures/nyu OUTPUT=.tmp/room-vlm-test
make validate-data
make baseline
make train
make evaluate CHECKPOINT=outputs/.../adapter
make api
make ui
```

`make test` does **not** download the model or NYU dataset.

---

## Inference strategies

1. **Constrained generation** — short greedy decode; normalize to a label or `invalid`.
2. **Candidate scoring (preferred)** — conditional log-likelihood of each label given image+prompt; length-normalize multi-token labels; softmax for relative scores.

These softmax scores are **normalized candidate likelihoods**, not necessarily calibrated real-world probabilities.

Video default: sample 1 fps (max 32 frames), aggregate with **mean log-probability**, then softmax.

Optional abstention (`unknown`) is disabled until you calibrate on **validation only**:

```bash
python scripts/calibrate_threshold.py --checkpoint outputs/<run_name>/adapter
```

---

## Deployment artifact

Deployed stack = base `Qwen/Qwen3-VL-4B-Instruct` + trained LoRA adapter (not full fine-tuned weights in-repo).

```bash
cp .env.example .env   # fill locally; never commit .env
docker compose up --build
```

---

## Prompt

Centralized in `src/room_vlm/prompts.py`:

> Classify the indoor room shown in the image.  
> Choose exactly one label from: bedroom / living room / bathroom / kitchen.  
> Respond with only the label.

---

## Reproducibility

`seed = 42` for Python / NumPy / PyTorch / scene splitting where possible. Full bit-for-bit GPU reproducibility is not always guaranteed.

---

## Limitations

1. NYU Depth V2 is an older Kinect indoor dataset.
2. Consecutive frames are highly correlated.
3. Scene-level splitting is mandatory.
4. A four-class model cannot identify arbitrary room types.
5. Candidate likelihood confidence may not be calibrated.
6. NYU imagery may differ from modern smartphone photos.
7. High NYU test scores do not imply strong real-world performance.
8. Deployments should eventually be evaluated on external modern data.
9. A conventional classifier may be more efficient for exactly four labels.
10. The purpose here is specifically to develop and evaluate a fine-tuned VLM.

---

## API surface

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness |
| GET | `/model-info` | Model id + adapter **name** (no host paths) |
| POST | `/predict/image` | Multipart image upload |
| POST | `/predict/video` | Multipart video upload |

---

## License

Apache-2.0. NYU Depth V2 and Qwen model weights retain their upstream licenses/terms — obtain and use them separately; they are not redistributed here.

## Citation

If you use NYU Depth V2, cite Silberman et al., ECCV 2012. If you use Qwen3-VL, cite the Qwen team’s technical reports.
