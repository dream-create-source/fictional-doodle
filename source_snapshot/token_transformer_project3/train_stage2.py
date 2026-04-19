import os

import torch

from common import (
    MODEL_CONFIG,
    build_model,
    encode_text,
    ensure_dir,
    get_batch,
    get_device,
    load_corpus_from_paths,
    load_model_checkpoint,
    load_tokenizer_from_checkpoint,
    maybe_clear_mps_cache,
    parse_env_paths,
    save_checkpoint,
    split_tensor,
)


PROJECT_DIR = os.path.dirname(__file__)
PROJECT2_DIR = os.path.join(PROJECT_DIR, "..", "project2", "data")

BASE_MODEL_PATH = os.getenv("BASE_MODEL_PATH", os.path.join(PROJECT_DIR, "stage1_final.pt"))
TRAIN_SOURCE_PATHS = parse_env_paths(
    os.getenv(
        "TRAIN_SOURCE_PATHS",
        ",".join(
            [
                os.path.join(PROJECT2_DIR, "instruct", "databricks-dolly-15k.jsonl"),
                os.path.join(PROJECT2_DIR, "synthetic", "ollama_synthetic.jsonl"),
            ]
        ),
    )
)

OUTPUT_MODEL = os.getenv("OUTPUT_MODEL", os.path.join(PROJECT_DIR, "stage2_final.pt"))
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", os.path.join(PROJECT_DIR, "checkpoints", "stage2"))
TRAIN_STEPS = int(os.getenv("TRAIN_STEPS", "2500"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))
GRAD_ACCUM_STEPS = int(os.getenv("GRAD_ACCUM_STEPS", "8"))
SEQ_LEN = int(os.getenv("SEQ_LEN", str(MODEL_CONFIG["max_seq_len"])))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-4"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.01"))
EVAL_INTERVAL = int(os.getenv("EVAL_INTERVAL", "125"))
EVAL_BATCHES = int(os.getenv("EVAL_BATCHES", "6"))
MAX_CHARS_PER_FILE = int(os.getenv("MAX_CHARS_PER_FILE", "12000000"))
TOTAL_CHAR_CAP = int(os.getenv("TOTAL_CHAR_CAP", "20000000"))
JSON_MAX_RECORDS = int(os.getenv("JSON_MAX_RECORDS", "120000"))


if not os.path.exists(BASE_MODEL_PATH):
    raise FileNotFoundError(f"Base model not found at {BASE_MODEL_PATH}. Run train_stage1.py first.")

device = get_device()
print(f"Using device: {device}")
print("Architecture: Transformer LM")
print(f"Model config: {MODEL_CONFIG}")

checkpoint, model = load_model_checkpoint(BASE_MODEL_PATH, device)
tokenizer = load_tokenizer_from_checkpoint(checkpoint)

instruction_text, loaded_train_paths = load_corpus_from_paths(
    TRAIN_SOURCE_PATHS,
    json_max_records=JSON_MAX_RECORDS,
    max_chars_per_file=MAX_CHARS_PER_FILE,
    total_char_cap=TOTAL_CHAR_CAP,
    instruction_mode=True,
)

print("Stage 2 instruction sources:")
for path in loaded_train_paths:
    print(f"  - {path}")
print(f"Per-file char cap: {MAX_CHARS_PER_FILE:,}")
print(f"Total char cap: {TOTAL_CHAR_CAP:,}")

encoded, dropped = encode_text(instruction_text, tokenizer, add_bos=True, add_eos=True)
data = torch.tensor(encoded, dtype=torch.long)
train_data, val_data = split_tensor(data, ratio=0.92)

print(f"Instruction dataset size: {len(data):,} tokens")
print(f"Dropped tokens during encoding: {dropped}")
print(f"Loaded base model from '{BASE_MODEL_PATH}'")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
ensure_dir(CHECKPOINT_DIR)
best_val_loss = float("inf")

for step in range(TRAIN_STEPS):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    running_loss = 0.0
    for _ in range(GRAD_ACCUM_STEPS):
        x, y = get_batch(data, train_data, val_data, "train", BATCH_SIZE, SEQ_LEN, device)
        _, loss = model(x, y)
        (loss / GRAD_ACCUM_STEPS).backward()
        running_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % EVAL_INTERVAL == 0:
        model.eval()
        val_losses = []
        with torch.no_grad():
            for _ in range(EVAL_BATCHES):
                x_val, y_val = get_batch(data, train_data, val_data, "val", BATCH_SIZE, SEQ_LEN, device)
                _, val_loss = model(x_val, y_val)
                val_losses.append(val_loss.item())

        train_loss = running_loss / GRAD_ACCUM_STEPS
        val_loss_mean = sum(val_losses) / len(val_losses)
        print(f"Step {step:5d} | train: {train_loss:.4f} | val: {val_loss_mean:.4f}")

        save_checkpoint(
            os.path.join(CHECKPOINT_DIR, f"stage2_step_{step}.pt"),
            model,
            optimizer=None,
            tokenizer=tokenizer,
            extra={
                "step": step,
                "stage": "stage2_instruct",
                "base_model_path": BASE_MODEL_PATH,
                "train_source_paths": loaded_train_paths,
                "train_loss": train_loss,
                "val_loss": val_loss_mean,
            },
        )

        if val_loss_mean < best_val_loss:
            best_val_loss = val_loss_mean
            save_checkpoint(
                OUTPUT_MODEL,
                model,
                optimizer=None,
                tokenizer=tokenizer,
                extra={
                    "step": step,
                    "stage": "stage2_instruct",
                    "base_model_path": BASE_MODEL_PATH,
                    "train_source_paths": loaded_train_paths,
                    "train_loss": train_loss,
                    "val_loss": val_loss_mean,
                },
            )
            print(f"  New best checkpoint -> {OUTPUT_MODEL}")

        maybe_clear_mps_cache()

print(f"\nStage 2 complete -> {OUTPUT_MODEL}")
