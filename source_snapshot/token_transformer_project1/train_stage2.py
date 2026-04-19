import os

import torch

from common import (
    MODEL_CONFIG,
    encode_text,
    ensure_dir,
    get_batch,
    get_device,
    load_corpus_from_paths,
    load_model_checkpoint,
    load_tokenizer_from_checkpoint,
    parse_env_paths,
    save_checkpoint,
    split_tensor,
)


PROJECT_DIR = os.path.dirname(__file__)
BASE_MODEL = os.getenv("BASE_MODEL", os.path.join(PROJECT_DIR, "stage1_final.pt"))
DOMAIN_SOURCE_PATHS = parse_env_paths(
    os.getenv(
        "DOMAIN_SOURCE_PATHS",
        f"{PROJECT_DIR}/../UAP sources/quotes_clean.txt,{PROJECT_DIR}/../UAP sources/norea.txt,{PROJECT_DIR}/../UAP sources/sightings.json",
    )
)

OUTPUT_MODEL = os.getenv("OUTPUT_MODEL", os.path.join(PROJECT_DIR, "stage2_final.pt"))
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", os.path.join(PROJECT_DIR, "checkpoints", "stage2"))
TRAIN_STEPS = int(os.getenv("TRAIN_STEPS", "4000"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "24"))
SEQ_LEN = int(os.getenv("SEQ_LEN", "256"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-4"))
SIGHTINGS_MAX_RECORDS = int(os.getenv("SIGHTINGS_MAX_RECORDS", "40000"))


device = get_device()
print(f"Using device: {device}")
print(f"Target model config: {MODEL_CONFIG}")

checkpoint, model = load_model_checkpoint(BASE_MODEL, device)
tokenizer = load_tokenizer_from_checkpoint(checkpoint)

domain_text, loaded_domain_paths = load_corpus_from_paths(
    DOMAIN_SOURCE_PATHS,
    json_max_records=SIGHTINGS_MAX_RECORDS,
)
encoded, dropped = encode_text(domain_text, tokenizer, add_bos=True, add_eos=True)
data = torch.tensor(encoded, dtype=torch.long)
train_data, val_data = split_tensor(data, ratio=0.9)

print("Stage 2 domain sources:")
for path in loaded_domain_paths:
    print(f"  - {path}")

print(f"Domain dataset size: {len(data):,} tokens")
print(f"Dropped tokens during encoding: {dropped}")

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
            os.path.join(CHECKPOINT_DIR, f"stage2_step_{step}.pt"),
            model,
            optimizer,
            tokenizer,
            extra={
                "step": step,
                "stage": "stage2_domain",
                "base_model": BASE_MODEL,
                "domain_source_paths": loaded_domain_paths,
                "train_loss": loss.item(),
                "val_loss": val_loss.item(),
                "sightings_max_records": SIGHTINGS_MAX_RECORDS,
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
                    "stage": "stage2_domain",
                    "base_model": BASE_MODEL,
                    "domain_source_paths": loaded_domain_paths,
                    "train_loss": loss.item(),
                    "val_loss": val_loss.item(),
                    "sightings_max_records": SIGHTINGS_MAX_RECORDS,
                },
            )
            print(f"  New best checkpoint -> {OUTPUT_MODEL}")

print(f"\nStage 2 complete -> {OUTPUT_MODEL}")
