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
    load_tokenizer,
    model_param_count,
    parse_env_paths,
    save_checkpoint,
    split_tensor,
)


PROJECT_DIR = os.path.dirname(__file__)
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", os.path.join(PROJECT_DIR, "tokenizer.json"))
TRAIN_SOURCE_PATHS = parse_env_paths(
    os.getenv(
        "TRAIN_SOURCE_PATHS",
        f"{PROJECT_DIR}/../data/english_basic,{PROJECT_DIR}/../shakespeare.txt",
    )
)

OUTPUT_MODEL = os.getenv("OUTPUT_MODEL", os.path.join(PROJECT_DIR, "stage1_final.pt"))
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", os.path.join(PROJECT_DIR, "checkpoints", "stage1"))
TRAIN_STEPS = int(os.getenv("TRAIN_STEPS", "5000"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
SEQ_LEN = int(os.getenv("SEQ_LEN", "256"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "3e-4"))


if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(
        f"Tokenizer not found at {TOKENIZER_PATH}. Run train_tokenizer.py first."
    )

device = get_device()
print(f"Using device: {device}")
print(f"Target model config: {MODEL_CONFIG}")

tokenizer = load_tokenizer(TOKENIZER_PATH)
train_text, loaded_train_paths = load_corpus_from_paths(TRAIN_SOURCE_PATHS)

print("Stage 1 training sources:")
for path in loaded_train_paths:
    print(f"  - {path}")

encoded, dropped = encode_text(train_text, tokenizer, add_bos=True, add_eos=True)
data = torch.tensor(encoded, dtype=torch.long)
train_data, val_data = split_tensor(data, ratio=0.9)

print(f"Tokenizer vocab size: {tokenizer.vocab_size()}")
print(f"Dataset size: {len(data):,} tokens")
print(f"Dropped tokens during encoding: {dropped}")
print(f"Approx parameter count: {model_param_count(tokenizer.vocab_size()):,}")

model = build_model(tokenizer.vocab_size(), device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

ensure_dir(CHECKPOINT_DIR)
best_val_loss = float("inf")

for step in range(TRAIN_STEPS):
    x, y = get_batch(data, train_data, val_data, "train", BATCH_SIZE, SEQ_LEN, device)
    _, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 250 == 0:
        x_val, y_val = get_batch(data, train_data, val_data, "val", BATCH_SIZE, SEQ_LEN, device)
        _, val_loss = model(x_val, y_val)
        print(f"Step {step:5d} | train: {loss.item():.4f} | val: {val_loss.item():.4f}")

        save_checkpoint(
            os.path.join(CHECKPOINT_DIR, f"stage1_step_{step}.pt"),
            model,
            optimizer,
            tokenizer,
            extra={
                "step": step,
                "stage": "stage1_pretrain",
                "tokenizer_path": TOKENIZER_PATH,
                "train_source_paths": loaded_train_paths,
                "train_loss": loss.item(),
                "val_loss": val_loss.item(),
            },
        )

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            save_checkpoint(
                OUTPUT_MODEL,
                model,
                optimizer,
                tokenizer,
                extra={
                    "step": step,
                    "stage": "stage1_pretrain",
                    "tokenizer_path": TOKENIZER_PATH,
                    "train_source_paths": loaded_train_paths,
                    "train_loss": loss.item(),
                    "val_loss": val_loss.item(),
                },
            )
            print(f"  New best checkpoint -> {OUTPUT_MODEL}")

print(f"\nStage 1 complete -> {OUTPUT_MODEL}")
