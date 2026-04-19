# Reproducibility

These commands assume you are working from the original workspace root, `/Users/admin/AI`.

## Verify Root Files

```sh
find . -maxdepth 1 -type f -print
```

The original folderless Python files are now also present directly in this project folder:

- `train.py`
- `train2.py`
- `train3.py`
- `chat.py`
- `load.py`
- `test.py`
- `clean.py`
- `model_final.pt`
- `model_english.pt`
- `model_qa.pt`
- `qa_dataset.txt`
- `quotes.txt`
- `shakespeare.txt`

This project folder copies the source scripts, lightweight text data, and the three root `.pt` model checkpoints under `models/`.

## Verify Dataset Size

```sh
wc -m data/english_basic/*.txt shakespeare.txt
```

Expected total for the root pretraining corpus:

```text
2,727,753 characters / char-level tokens
```

## Verify Checkpoint Parameter Counts

Use an environment with PyTorch installed:

```sh
python - <<'PY'
import torch

for path in ["model_final.pt", "model_english.pt", "model_qa.pt"]:
    ckpt = torch.load(path, map_location="cpu")
    params = sum(t.numel() for t in ckpt["model_state"].values())
    print(path, params, ckpt.get("vocab_size"), ckpt.get("stage"))
PY
```

Expected output shape:

```text
model_final.pt 38303744 207 None
model_english.pt 38320128 223 english_pretrain
model_qa.pt 38303744 207 None
```

## Re-run English Pretraining

```sh
python train.py
```

Default behavior:

- Reads `data/english_basic` and `shakespeare.txt`
- Builds a character vocabulary
- Trains a 512-dim, 16-head, 12-layer GPT-style model
- Saves periodic checkpoints under `models_english`
- Saves final checkpoint to `model_english.pt`

## Re-run Teacher-Guided Q&A Fine-Tuning

Start Ollama with compatible teacher models available, then run:

```sh
python train3.py
```

Default behavior:

- Reads `quotes.txt`
- Uses `deepseek-r1:latest` and `llama3:latest` through Ollama
- Builds or refreshes `qa_dataset.txt`
- Applies reward-guided answer loss
- Saves the best Q&A checkpoint to `model_qa.pt`

## Launch Chat

```sh
python chat.py
```

Then select a checkpoint such as `model_qa.pt`.
