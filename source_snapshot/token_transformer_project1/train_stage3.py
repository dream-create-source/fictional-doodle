import json
import os
import random
import re
import urllib.error
import urllib.request

import torch

from common import (
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
BASE_MODEL = os.getenv("BASE_MODEL", os.path.join(PROJECT_DIR, "stage2_final.pt"))
QA_DATASET_FILE = os.getenv("QA_DATASET_FILE", os.path.join(PROJECT_DIR, "qa_dataset.txt"))
OUTPUT_MODEL = os.getenv("OUTPUT_MODEL", os.path.join(PROJECT_DIR, "stage3_final.pt"))
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", os.path.join(PROJECT_DIR, "checkpoints", "stage3"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
TEACHER_MODELS = [
    model.strip()
    for model in os.getenv("OLLAMA_TEACHER_MODELS", "deepseek-r1:latest,llama3:latest").split(",")
    if model.strip()
]
QA_SOURCE_PATHS = parse_env_paths(
    os.getenv(
        "QA_SOURCE_PATHS",
        f"{PROJECT_DIR}/../UAP sources/quotes_clean.txt,{PROJECT_DIR}/../UAP sources/norea.txt,{PROJECT_DIR}/../UAP sources/sightings.json",
    )
)

TRAIN_STEPS = int(os.getenv("TRAIN_STEPS", "2500"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
SEQ_LEN = int(os.getenv("SEQ_LEN", "256"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-4"))
QA_PER_CHUNK = int(os.getenv("QA_PER_CHUNK", "6"))
MAX_CHUNKS = int(os.getenv("QA_MAX_CHUNKS", "80"))
CHUNK_CHARS = int(os.getenv("QA_CHUNK_CHARS", "2200"))
CHUNK_OVERLAP = int(os.getenv("QA_CHUNK_OVERLAP", "250"))
SIGHTINGS_QA_RECORDS = int(os.getenv("SIGHTINGS_QA_RECORDS", "12000"))


def sanitize_teacher_text(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"```(?:json)?", "", text)
    return text.strip()


def safe_json_loads(text):
    cleaned = sanitize_teacher_text(text)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def call_ollama(model_name, prompt):
    payload = json.dumps(
        {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.5},
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=180) as response:
        body = json.loads(response.read().decode("utf-8"))
        return sanitize_teacher_text(body.get("response", ""))


def build_qa_prompt(chunk, pair_count):
    return f"""Create exactly {pair_count} grounded question-answer pairs from the source passage.
Use only the passage.
Keep questions diverse and useful.
Return strict JSON:
{{
  "pairs": [
    {{"question": "Q text", "answer": "A text"}}
  ]
}}

Passage:
\"\"\"
{chunk}
\"\"\""""


def chunk_text(text, chunk_chars, overlap, max_chunks):
    chunks = []
    start = 0
    step = max(1, chunk_chars - overlap)
    while start < len(text) and len(chunks) < max_chunks:
        end = min(len(text), start + chunk_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    random.shuffle(chunks)
    return chunks


def generate_qa_pairs(text):
    chunks = chunk_text(text, CHUNK_CHARS, CHUNK_OVERLAP, MAX_CHUNKS)
    all_pairs = []
    seen_questions = set()

    for index, chunk in enumerate(chunks, start=1):
        prompt = build_qa_prompt(chunk, QA_PER_CHUNK)
        parsed = None
        for model_name in TEACHER_MODELS:
            try:
                parsed = safe_json_loads(call_ollama(model_name, prompt))
                if parsed:
                    break
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
                continue

        if not parsed:
            print(f"Chunk {index:02d}: teacher unavailable, skipping")
            continue

        for item in parsed.get("pairs", []):
            question = " ".join(str(item.get("question", "")).split())
            answer = " ".join(str(item.get("answer", "")).split())
            if len(question) < 8 or len(answer) < 20:
                continue
            key = question.lower()
            if key in seen_questions:
                continue
            seen_questions.add(key)
            all_pairs.append((question, answer))

        print(f"Chunk {index:02d}: total pairs so far = {len(all_pairs)}")

    return all_pairs


def format_pairs(pairs):
    if not pairs:
        return ""
    return "\n\n".join([f"Q: {question}\nA: {answer}" for question, answer in pairs]) + "\n"


device = get_device()
print(f"Using device: {device}")

checkpoint, model = load_model_checkpoint(BASE_MODEL, device)
tokenizer = load_tokenizer_from_checkpoint(checkpoint)

qa_source_text, loaded_paths = load_corpus_from_paths(QA_SOURCE_PATHS, json_max_records=SIGHTINGS_QA_RECORDS)
ensure_dir(CHECKPOINT_DIR)

if os.path.exists(QA_DATASET_FILE) and os.path.getsize(QA_DATASET_FILE) > 0:
    with open(QA_DATASET_FILE, "r", encoding="utf-8") as file:
        qa_text = file.read().strip()
    print(f"Using existing Q&A dataset: {QA_DATASET_FILE}")
else:
    print("Generating Q&A dataset with Ollama teachers...")
    qa_pairs = generate_qa_pairs(qa_source_text)
    qa_text = format_pairs(qa_pairs)
    with open(QA_DATASET_FILE, "w", encoding="utf-8") as file:
        file.write(qa_text)
    print(f"Saved Q&A dataset -> {QA_DATASET_FILE}")

encoded, dropped = encode_text(qa_text, tokenizer, add_bos=True, add_eos=True)
data = torch.tensor(encoded, dtype=torch.long)
train_data, val_data = split_tensor(data, ratio=0.8)

print("Stage 3 Q&A sources:")
for path in loaded_paths:
    print(f"  - {path}")
print(f"Q&A dataset size: {len(data):,} tokens")
print(f"Dropped tokens during encoding: {dropped}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
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
            os.path.join(CHECKPOINT_DIR, f"stage3_step_{step}.pt"),
            model,
            optimizer,
            tokenizer,
            extra={
                "step": step,
                "stage": "stage3_qna",
                "base_model": BASE_MODEL,
                "qa_dataset_file": QA_DATASET_FILE,
                "qa_source_paths": loaded_paths,
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
                    "stage": "stage3_qna",
                    "base_model": BASE_MODEL,
                    "qa_dataset_file": QA_DATASET_FILE,
                    "qa_source_paths": loaded_paths,
                    "train_loss": loss.item(),
                    "val_loss": val_loss.item(),
                },
            )
            print(f"  New best checkpoint -> {OUTPUT_MODEL}")

print(f"\nStage 3 complete -> {OUTPUT_MODEL}")
