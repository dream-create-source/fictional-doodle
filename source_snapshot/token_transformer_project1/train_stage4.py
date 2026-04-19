import json
import os
import re
import urllib.error
import urllib.request

import torch
import torch.nn.functional as F

from common import (
    clean_generated_answer,
    encode_text,
    ensure_dir,
    get_batch,
    get_device,
    load_model_checkpoint,
    load_tokenizer_from_checkpoint,
    save_checkpoint,
    split_tensor,
)


PROJECT_DIR = os.path.dirname(__file__)
BASE_MODEL = os.getenv("BASE_MODEL", os.path.join(PROJECT_DIR, "stage3_final.pt"))
QA_DATASET_FILE = os.getenv("QA_DATASET_FILE", os.path.join(PROJECT_DIR, "qa_dataset.txt"))
OUTPUT_MODEL = os.getenv("OUTPUT_MODEL", os.path.join(PROJECT_DIR, "stage4_final.pt"))
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", os.path.join(PROJECT_DIR, "checkpoints", "stage4"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
TEACHER_MODELS = [
    model.strip()
    for model in os.getenv("OLLAMA_TEACHER_MODELS", "deepseek-r1:latest,llama3:latest").split(",")
    if model.strip()
]

TRAIN_STEPS = int(os.getenv("TRAIN_STEPS", "2000"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
SEQ_LEN = int(os.getenv("SEQ_LEN", "256"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "8e-5"))
REWARD_EVAL_INTERVAL = int(os.getenv("REWARD_EVAL_INTERVAL", "25"))
REWARD_WEIGHT = float(os.getenv("REWARD_WEIGHT", "1.5"))
STUDENT_ANSWER_TOKENS = int(os.getenv("STUDENT_ANSWER_TOKENS", "60"))


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
            "options": {"temperature": 0.2},
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


def parse_qa_pairs(text):
    matches = re.findall(r"Q:\s*(.+?)\nA:\s*(.+?)(?=\nQ:\s|\Z)", text, flags=re.DOTALL)
    examples = []
    for question, answer in matches:
        question = " ".join(question.strip().split())
        answer = " ".join(answer.strip().split())
        if len(question) < 8 or len(answer) < 20:
            continue
        examples.append({"question": question, "answer": answer})
    return examples


def build_reward_prompt(question, reference_answer, student_answer):
    return f"""Score the student answer against the reference answer.
Return strict JSON:
{{
  "reward": 0.0,
  "feedback": "short feedback",
  "better_answer": "better answer"
}}

Question:
{question}

Reference answer:
{reference_answer}

Student answer:
{student_answer}
"""


def grade_answer(question, reference_answer, student_answer):
    for model_name in TEACHER_MODELS:
        try:
            parsed = safe_json_loads(call_ollama(model_name, build_reward_prompt(question, reference_answer, student_answer)))
            if parsed:
                reward = max(0.0, min(1.0, float(parsed.get("reward", 0.0))))
                better_answer = " ".join(str(parsed.get("better_answer", reference_answer)).split()) or reference_answer
                feedback = " ".join(str(parsed.get("feedback", "")).split())
                return {"reward": reward, "better_answer": better_answer, "feedback": feedback}
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
            continue
    return {"reward": 0.0, "better_answer": reference_answer, "feedback": "Teacher unavailable; fallback used."}


def sample_answer(model, prompt_ids, tokenizer, device):
    token_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(STUDENT_ANSWER_TOKENS):
            token_ids_crop = token_ids[:, -model.max_seq_len:]
            logits, _ = model(token_ids_crop)
            logits = logits[:, -1, :] / 0.8
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_ids = torch.cat([token_ids, next_token], dim=1)

            answer = tokenizer.decode(token_ids[0].tolist()[len(prompt_ids):])
            if "\nQ:" in answer or "\n\n" in answer:
                break

    if model_was_training:
        model.train()
    if device.type == "mps":
        torch.mps.empty_cache()

    return clean_generated_answer(tokenizer.decode(token_ids[0].tolist()[len(prompt_ids):]))


def compute_answer_loss(model, prompt_ids, answer_ids, device):
    tokens = prompt_ids + answer_ids
    if len(tokens) < 2:
        return None

    input_ids = torch.tensor([tokens[:-1]], dtype=torch.long).to(device)
    targets = torch.tensor(tokens[1:], dtype=torch.long).to(device)
    logits, _ = model(input_ids)
    logits = logits[0]
    losses = F.cross_entropy(logits, targets, reduction="none")
    mask = torch.zeros_like(targets, dtype=torch.float)
    mask[max(0, len(prompt_ids) - 1):] = 1.0
    if mask.sum().item() == 0:
        return None
    return (losses * mask).sum() / mask.sum()


device = get_device()
print(f"Using device: {device}")

checkpoint, model = load_model_checkpoint(BASE_MODEL, device)
tokenizer = load_tokenizer_from_checkpoint(checkpoint)

with open(QA_DATASET_FILE, "r", encoding="utf-8") as file:
    qa_text = file.read().strip()

examples = parse_qa_pairs(qa_text)
encoded, dropped = encode_text(qa_text, tokenizer, add_bos=True, add_eos=True)
data = torch.tensor(encoded, dtype=torch.long)
train_data, val_data = split_tensor(data, ratio=0.8)

print(f"Loaded {len(examples)} Q&A examples")
print(f"Q&A token count: {len(data):,}")
print(f"Dropped tokens during encoding: {dropped}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
ensure_dir(CHECKPOINT_DIR)
best_val_loss = float("inf")
recent_rewards = []

for step in range(TRAIN_STEPS):
    x, y = get_batch(data, train_data, val_data, "train", BATCH_SIZE, SEQ_LEN, device)
    _, base_loss = model(x, y)
    total_loss = base_loss
    reward_info = None
    student_answer = None

    if step % REWARD_EVAL_INTERVAL == 0 and examples:
        example = examples[step % len(examples)]
        prompt_text = f"Q: {example['question']}\nA:"
        prompt_ids = tokenizer.encode(prompt_text, add_bos=True, add_eos=False)
        student_answer = sample_answer(model, prompt_ids, tokenizer, device)
        reward_info = grade_answer(example["question"], example["answer"], student_answer)
        better_answer_ids = tokenizer.encode(" " + reward_info["better_answer"], add_bos=False, add_eos=True)
        guided_loss = compute_answer_loss(model, prompt_ids, better_answer_ids, device)
        if guided_loss is not None:
            reward_scale = 1.0 + REWARD_WEIGHT * (1.0 - reward_info["reward"])
            total_loss = total_loss + guided_loss * reward_scale
            recent_rewards.append(reward_info["reward"])
            recent_rewards = recent_rewards[-50:]

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if device.type == "mps" and step % 25 == 0:
        torch.mps.empty_cache()

    if step % 100 == 0:
        avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0
        print(f"Step {step:5d} | base: {base_loss.item():.4f} | reward: {avg_reward:.3f}")
        if reward_info is not None:
            print(f"  reward={reward_info['reward']:.3f} feedback={reward_info['feedback'][:140]}")
            print(f"  student: {student_answer[:140]}")

    if step % 250 == 0:
        x_val, y_val = get_batch(data, train_data, val_data, "val", BATCH_SIZE, SEQ_LEN, device)
        _, val_loss = model(x_val, y_val)
        avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0
        print(f"Validation {step:5d} | train: {base_loss.item():.4f} | val: {val_loss.item():.4f} | avg reward: {avg_reward:.3f}")

        save_checkpoint(
            os.path.join(CHECKPOINT_DIR, f"stage4_step_{step}.pt"),
            model,
            optimizer,
            tokenizer,
            extra={
                "step": step,
                "stage": "stage4_rlaif",
                "base_model": BASE_MODEL,
                "qa_dataset_file": QA_DATASET_FILE,
                "train_loss": base_loss.item(),
                "val_loss": val_loss.item(),
                "avg_reward": avg_reward,
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
                    "stage": "stage4_rlaif",
                    "base_model": BASE_MODEL,
                    "qa_dataset_file": QA_DATASET_FILE,
                    "train_loss": base_loss.item(),
                    "val_loss": val_loss.item(),
                    "avg_reward": avg_reward,
                },
            )
            print(f"  New best checkpoint -> {OUTPUT_MODEL}")

print(f"\nStage 4 complete -> {OUTPUT_MODEL}")
