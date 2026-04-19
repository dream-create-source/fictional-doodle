import os

import torch

from common import (
    MODEL_CONFIG,
    build_model,
    encode_text,
    ensure_dir,
    expand_source_paths,
    get_batch,
    get_device,
    iter_source_chunks,
    load_corpus_from_paths,
    load_tokenizer,
    maybe_clear_mps_cache,
    model_param_count,
    parse_env_paths,
    save_checkpoint,
    split_tensor,
)


PROJECT_DIR = os.path.dirname(__file__)
PROJECT2_DIR = os.path.join(PROJECT_DIR, "..", "project2", "data")

TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", os.path.join(PROJECT_DIR, "tokenizer.json"))
TRAIN_SOURCE_PATHS = parse_env_paths(
    os.getenv(
        "TRAIN_SOURCE_PATHS",
        ",".join(
            [
                os.path.join(PROJECT2_DIR, "wiki", "simplewiki-latest-pages-articles-multistream.xml"),
                os.path.join(PROJECT2_DIR, "books"),
                os.path.join(PROJECT2_DIR, "science", "arxiv_abstracts.jsonl"),
            ]
        ),
    )
)

OUTPUT_MODEL = os.getenv("OUTPUT_MODEL", os.path.join(PROJECT_DIR, "stage1_final.pt"))
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", os.path.join(PROJECT_DIR, "checkpoints", "stage1"))
TRAIN_STEPS = int(os.getenv("TRAIN_STEPS", "6000"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))
GRAD_ACCUM_STEPS = int(os.getenv("GRAD_ACCUM_STEPS", "8"))
SEQ_LEN = int(os.getenv("SEQ_LEN", str(MODEL_CONFIG["max_seq_len"])))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2.5e-4"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.1"))
EVAL_INTERVAL = int(os.getenv("EVAL_INTERVAL", "200"))
EVAL_BATCHES = int(os.getenv("EVAL_BATCHES", "4"))
MAX_CHARS_PER_FILE = int(os.getenv("MAX_CHARS_PER_FILE", "1000000"))
TOTAL_CHAR_CAP = int(os.getenv("TOTAL_CHAR_CAP", "0"))
JSON_MAX_RECORDS = int(os.getenv("JSON_MAX_RECORDS", "120000"))
STREAM_CHUNK_CHARS = int(os.getenv("STREAM_CHUNK_CHARS", "4000000"))
VAL_TOTAL_CHAR_CAP = int(os.getenv("VAL_TOTAL_CHAR_CAP", "6000000"))
VAL_MAX_CHARS_PER_FILE = int(os.getenv("VAL_MAX_CHARS_PER_FILE", "1000000"))
STEPS_PER_CHUNK = int(os.getenv("STEPS_PER_CHUNK", "24"))
TRAIN_EPOCHS = int(os.getenv("TRAIN_EPOCHS", "1"))


if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}. Run train_tokenizer.py first.")

device = get_device()
print(f"Using device: {device}")
print("Architecture: Transformer LM")
print(f"Model config: {MODEL_CONFIG}")

tokenizer = load_tokenizer(TOKENIZER_PATH)
expanded_train_paths = expand_source_paths(TRAIN_SOURCE_PATHS)
if not expanded_train_paths:
    raise FileNotFoundError(f"No training sources found in: {TRAIN_SOURCE_PATHS}")

val_text, loaded_train_paths = load_corpus_from_paths(
    TRAIN_SOURCE_PATHS,
    json_max_records=JSON_MAX_RECORDS,
    max_chars_per_file=VAL_MAX_CHARS_PER_FILE,
    total_char_cap=VAL_TOTAL_CHAR_CAP,
)

print("Stage 1 pretraining sources:")
for path in expanded_train_paths:
    print(f"  - {path}")
print(f"Streaming chunk chars: {STREAM_CHUNK_CHARS:,}")
print(f"Total char cap: {'unlimited' if TOTAL_CHAR_CAP <= 0 else f'{TOTAL_CHAR_CAP:,}'}")
print(f"Validation per-file char cap: {VAL_MAX_CHARS_PER_FILE:,}")
print(f"Validation total char cap: {VAL_TOTAL_CHAR_CAP:,}")

encoded_val, dropped = encode_text(val_text, tokenizer, add_bos=True, add_eos=True)
val_tensor_full = torch.tensor(encoded_val, dtype=torch.long)
_, val_data = split_tensor(val_tensor_full, ratio=0.1)

print(f"Tokenizer vocab size: {tokenizer.vocab_size()}")
print("Training dataset mode: full-source streaming")
print(f"Validation dataset size: {len(val_tensor_full):,} tokens")
print(f"Dropped tokens during encoding: {dropped}")
print(f"Approx parameter count: {model_param_count(tokenizer.vocab_size()):,}")

model = build_model(tokenizer.vocab_size(), device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

ensure_dir(CHECKPOINT_DIR)
best_val_loss = float("inf")
step = 0
chunk_count = 0
train_data = val_tensor_full

for epoch in range(TRAIN_EPOCHS):
    for chunk_path, chunk_text in iter_source_chunks(
        TRAIN_SOURCE_PATHS,
        json_max_records=JSON_MAX_RECORDS,
        chunk_chars=STREAM_CHUNK_CHARS,
        total_char_cap=TOTAL_CHAR_CAP,
    ):
        encoded_chunk, _ = encode_text(chunk_text, tokenizer, add_bos=True, add_eos=True)
        chunk_tensor = torch.tensor(encoded_chunk, dtype=torch.long)
        if len(chunk_tensor) <= max(SEQ_LEN + 2, 64):
            continue

        train_data, _ = split_tensor(chunk_tensor, ratio=0.98)
        chunk_count += 1

        for _ in range(STEPS_PER_CHUNK):
            if step >= TRAIN_STEPS:
                break

            model.train()
            optimizer.zero_grad(set_to_none=True)

            running_loss = 0.0
            for _ in range(GRAD_ACCUM_STEPS):
                x, y = get_batch(chunk_tensor, train_data, val_data, "train", BATCH_SIZE, SEQ_LEN, device)
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
                        x_val, y_val = get_batch(val_tensor_full, train_data, val_data, "val", BATCH_SIZE, SEQ_LEN, device)
                        _, val_loss = model(x_val, y_val)
                        val_losses.append(val_loss.item())

                train_loss = running_loss / GRAD_ACCUM_STEPS
                val_loss_mean = sum(val_losses) / len(val_losses)
                print(
                    f"Step {step:5d} | train: {train_loss:.4f} | val: {val_loss_mean:.4f} | chunk {chunk_count} | source: {os.path.basename(chunk_path)}"
                )

                save_checkpoint(
                    os.path.join(CHECKPOINT_DIR, f"stage1_step_{step}.pt"),
                    model,
                    optimizer=None,
                    tokenizer=tokenizer,
                    extra={
                        "step": step,
                        "stage": "stage1_pretrain",
                        "train_source_paths": expanded_train_paths,
                        "train_loss": train_loss,
                        "val_loss": val_loss_mean,
                        "chunk_count": chunk_count,
                        "chunk_path": chunk_path,
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
                            "stage": "stage1_pretrain",
                            "train_source_paths": expanded_train_paths,
                            "train_loss": train_loss,
                            "val_loss": val_loss_mean,
                            "chunk_count": chunk_count,
                            "chunk_path": chunk_path,
                        },
                    )
                    print(f"  New best checkpoint -> {OUTPUT_MODEL}")

                maybe_clear_mps_cache()

            step += 1

        if step >= TRAIN_STEPS:
            break

    if step >= TRAIN_STEPS:
        break

print(f"\nStage 1 complete -> {OUTPUT_MODEL}")
