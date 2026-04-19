import json
import os
import random
import re
import urllib.error
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

SOURCE_FILE = os.getenv("QA_SOURCE_FILE", "quotes_clean.txt")
QA_DATASET_FILE = os.getenv("QA_DATASET_FILE", "qa_dataset.txt")
BASE_MODEL = os.getenv("BASE_MODEL", "model_final.pt")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
TEACHER_MODELS = [
    model.strip()
    for model in os.getenv(
        "OLLAMA_TEACHER_MODELS",
        "deepseek-r1:latest,llama3:latest",
    ).split(",")
    if model.strip()
]

REBUILD_QA_DATASET = os.getenv("REBUILD_QA_DATASET", "1") == "1"
CHUNK_CHARS = int(os.getenv("QA_SOURCE_CHUNK_CHARS", "2200"))
CHUNK_OVERLAP = int(os.getenv("QA_SOURCE_CHUNK_OVERLAP", "250"))
MAX_CHUNKS = int(os.getenv("QA_MAX_CHUNKS", "48"))
QA_PER_CHUNK = int(os.getenv("QA_PER_CHUNK", "4"))
MIN_SEED_PAIRS = int(os.getenv("QA_MIN_SEED_PAIRS", "80"))
DYNAMIC_GROWTH_PAIRS = int(os.getenv("QA_DYNAMIC_GROWTH_PAIRS", "12"))
POOL_REFRESH_INTERVAL = int(os.getenv("QA_POOL_REFRESH_INTERVAL", "250"))
TRAIN_STEPS = int(os.getenv("QA_TRAIN_STEPS", "3000"))
REWARD_EVAL_INTERVAL = int(os.getenv("QA_REWARD_EVAL_INTERVAL", "20"))
STUDENT_ANSWER_CHARS = int(os.getenv("QA_STUDENT_ANSWER_CHARS", "140"))
REWARD_WEIGHT = float(os.getenv("QA_REWARD_WEIGHT", "1.5"))
BASELINE_MOMENTUM = float(os.getenv("QA_BASELINE_MOMENTUM", "0.9"))


# ─────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Metal GPU (MPS) ✓")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA CUDA GPU ✓")
else:
    device = torch.device("cpu")
    print("Using CPU")


# ─────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, channels = x.shape
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.out_proj(attended)


class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.2):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.feedforward = FeedForward(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.feedforward(self.norm2(x)))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, token_ids, targets=None):
        _, seq_len = token_ids.shape
        tok_emb = self.token_embedding(token_ids)
        positions = torch.arange(seq_len, device=token_ids.device)
        pos_emb = self.position_embedding(positions)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(batch_size * seq_len, vocab_size)
            targets_flat = targets.view(batch_size * seq_len)
            loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss

    def generate(self, token_ids, max_new_tokens, temperature=0.8, rep_penalty=1.3):
        for _ in range(max_new_tokens):
            token_ids_crop = token_ids[:, -self.max_seq_len:]
            logits, _ = self.forward(token_ids_crop)
            logits = logits[:, -1, :]
            for token_id in set(token_ids[0].tolist()):
                logits[0, token_id] /= rep_penalty
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_ids = torch.cat([token_ids, next_token], dim=1)
        return token_ids


# ─────────────────────────────────────────
# FILES AND TEXT
# ─────────────────────────────────────────

def read_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        return file.read().strip()


def write_text(path, text):
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)


def chunk_text(text, chunk_chars, overlap, max_chunks):
    if not text:
        return []

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


def normalize_space(text):
    return " ".join(text.strip().split())


def sanitize_teacher_text(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"```(?:json)?", "", text)
    return text.strip()


def iter_teacher_models():
    models = TEACHER_MODELS[:]
    random.shuffle(models)
    return models


def safe_json_loads(text):
    cleaned = sanitize_teacher_text(text)
    for pattern in [r"\{.*\}", r"\[.*\]"]:
        match = re.search(pattern, cleaned, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
    return None


# ─────────────────────────────────────────
# OLLAMA TEACHER
# ─────────────────────────────────────────

def call_ollama(model_name, prompt, temperature=0.4):
    payload = json.dumps(
        {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
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


def build_pair_prompt(source_name, chunk, pair_count):
    return f"""You are generating grounded supervised training data for a custom language model.
Use only the source passage.

Create exactly {pair_count} diverse question-answer pairs.
Vary the questions across definition, cause, implication, comparison, mechanism, and timeline when the passage supports it.
Keep answers concise, factual, and directly grounded in the passage.
Do not invent facts.
Return strict JSON only in this schema:
{{
  "pairs": [
    {{"question": "Q text", "answer": "A text"}}
  ]
}}

Source: {source_name}
Passage:
\"\"\"
{chunk}
\"\"\""""


def build_reward_prompt(question, reference_answer, student_answer, source_excerpt):
    return f"""You are a reward model grading a student answer for grounded Q&A training.
Use only the source excerpt and reference answer.

Grade the student answer on:
- correctness
- groundedness
- completeness

Return strict JSON only:
{{
  "reward": 0.0,
  "correctness": 0.0,
  "groundedness": 0.0,
  "completeness": 0.0,
  "feedback": "short feedback",
  "better_answer": "improved answer grounded in the source"
}}

Scoring rules:
- reward must be between 0.0 and 1.0
- low reward if the student answer is vague, incorrect, or unsupported
- better_answer must stay grounded in the source and be concise

Question:
{question}

Reference answer:
{reference_answer}

Student answer:
{student_answer}

Source excerpt:
\"\"\"
{source_excerpt}
\"\"\""""


def request_teacher_json(prompt, temperature=0.4):
    last_error = None
    for teacher_model in iter_teacher_models():
        try:
            response = call_ollama(teacher_model, prompt, temperature=temperature)
            parsed = safe_json_loads(response)
            if parsed is not None:
                return teacher_model, parsed
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError("No Ollama teacher models produced valid JSON.")


def parse_generated_pairs(obj, source_excerpt, teacher_model):
    pairs = []
    items = obj.get("pairs", []) if isinstance(obj, dict) else []
    for item in items:
        question = normalize_space(str(item.get("question", "")))
        answer = normalize_space(str(item.get("answer", "")))
        if len(question) < 8 or len(answer) < 20:
            continue
        pairs.append(
            {
                "question": question,
                "answer": answer,
                "source_excerpt": source_excerpt,
                "teacher_model": teacher_model,
            }
        )
    return pairs


def generate_pairs_from_chunk(chunk, pair_count):
    prompt = build_pair_prompt(SOURCE_FILE, chunk, pair_count)
    teacher_model, obj = request_teacher_json(prompt, temperature=0.8)
    return parse_generated_pairs(obj, chunk, teacher_model)


# ─────────────────────────────────────────
# DATASET POOL
# ─────────────────────────────────────────

def parse_qa_pairs(text):
    matches = re.findall(r"Q:\s*(.+?)\nA:\s*(.+?)(?=\nQ:\s|\Z)", text, flags=re.DOTALL)
    pairs = []
    for question, answer in matches:
        question = normalize_space(question)
        answer = normalize_space(answer)
        if len(question) < 8 or len(answer) < 20:
            continue
        pairs.append(
            {
                "question": question,
                "answer": answer,
                "source_excerpt": "",
                "teacher_model": "existing_dataset",
            }
        )
    return pairs


def dataset_text_from_examples(examples):
    if not examples:
        return ""
    blocks = [f"Q: {item['question']}\nA: {item['answer']}" for item in examples]
    return "\n\n".join(blocks) + "\n"


def sync_dataset_file(examples):
    write_text(QA_DATASET_FILE, dataset_text_from_examples(examples))


def dedupe_examples(examples):
    deduped = []
    seen_questions = set()
    for item in examples:
        key = item["question"].lower()
        if key in seen_questions:
            continue
        seen_questions.add(key)
        deduped.append(item)
    return deduped


def expand_example_pool(source_chunks, example_pool, target_pairs):
    if target_pairs <= 0 or not source_chunks:
        return []

    seen_questions = {item["question"].lower() for item in example_pool}
    generated = []
    attempts = 0
    max_attempts = max(len(source_chunks) * 3, target_pairs * 4)

    while len(generated) < target_pairs and attempts < max_attempts:
        chunk = random.choice(source_chunks)
        needed = min(QA_PER_CHUNK, target_pairs - len(generated))
        try:
            pairs = generate_pairs_from_chunk(chunk, needed)
            print(f"Teacher generated {len(pairs)} pairs on attempt {attempts + 1}")
        except Exception as exc:
            attempts += 1
            print(f"Teacher generation attempt {attempts} failed: {exc}")
            continue

        for item in pairs:
            key = item["question"].lower()
            if key in seen_questions:
                continue
            seen_questions.add(key)
            generated.append(item)
            if len(generated) >= target_pairs:
                break
        attempts += 1

    return generated


def build_example_pool():
    if not os.path.exists(SOURCE_FILE):
        raise FileNotFoundError(
            f"Source file '{SOURCE_FILE}' not found. Set QA_SOURCE_FILE to your corpus path."
        )

    source_text = read_text(SOURCE_FILE)
    source_chunks = chunk_text(
        source_text,
        chunk_chars=CHUNK_CHARS,
        overlap=CHUNK_OVERLAP,
        max_chunks=MAX_CHUNKS,
    )

    existing_examples = []
    if os.path.exists(QA_DATASET_FILE) and os.path.getsize(QA_DATASET_FILE) > 0:
        existing_examples = parse_qa_pairs(read_text(QA_DATASET_FILE))

    if REBUILD_QA_DATASET:
        example_pool = []
    else:
        example_pool = existing_examples

    target_seed_pairs = max(MIN_SEED_PAIRS, len(example_pool))
    missing_pairs = target_seed_pairs - len(example_pool)

    if missing_pairs > 0:
        print(f"Growing training pool by {missing_pairs} teacher-generated pairs")
        new_examples = expand_example_pool(source_chunks, example_pool, missing_pairs)
        example_pool.extend(new_examples)

    if not example_pool and existing_examples:
        example_pool = existing_examples

    example_pool = dedupe_examples(example_pool)

    if not example_pool:
        raise RuntimeError(
            "Could not build any training examples. Start Ollama with a teacher model "
            "or keep a populated qa_dataset.txt as fallback."
        )

    sync_dataset_file(example_pool)
    print(f"Training pool size: {len(example_pool)} Q&A pairs")
    return example_pool, source_chunks


# ─────────────────────────────────────────
# LOAD BASE MODEL
# ─────────────────────────────────────────

if not os.path.exists(BASE_MODEL):
    print(f"ERROR: {BASE_MODEL} not found!")
    print("Available .pt files:")
    for file_name in os.listdir("."):
        if file_name.endswith(".pt"):
            print(f"  {file_name}")
    raise SystemExit(1)

checkpoint = torch.load(BASE_MODEL, map_location=device)
char_to_int = checkpoint["char_to_int"]
int_to_char = checkpoint["int_to_char"]
vocab_size = checkpoint["vocab_size"]

encode = lambda s: [char_to_int[c] for c in s if c in char_to_int]
decode = lambda l: "".join([int_to_char[i] for i in l])

model = GPT(
    vocab_size=vocab_size,
    embed_dim=512,
    num_heads=16,
    num_layers=12,
    max_seq_len=512,
).to(device)

model.load_state_dict(checkpoint["model_state"])
print(f"Loaded base model from '{BASE_MODEL}'")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


# ─────────────────────────────────────────
# CHAR-LEVEL TRAINING DATA
# ─────────────────────────────────────────

example_pool, source_chunks = build_example_pool()


def rebuild_char_dataset(examples):
    text = dataset_text_from_examples(examples)
    encoded = encode(text)
    if len(encoded) < 128:
        raise RuntimeError(
            "Q&A dataset is too small after encoding. Increase the teacher pool or use a larger source."
        )

    tensor_data = torch.tensor(encoded, dtype=torch.long)
    split = max(1, int(0.8 * len(tensor_data)))
    train_tensor = tensor_data[:split]
    val_tensor = tensor_data[split:] if split < len(tensor_data) else tensor_data[-max(64, len(tensor_data) // 5):]
    return tensor_data, train_tensor, val_tensor


data, train_data, val_data = rebuild_char_dataset(example_pool)
print(f"Encoded Q&A dataset size: {len(data):,} tokens")


def get_batch(split_name, batch_size=16, seq_len=256):
    dataset = train_data if split_name == "train" else val_data
    effective_seq_len = min(seq_len, max(8, len(dataset) - 1))
    max_start = len(dataset) - effective_seq_len - 1

    if max_start <= 0:
        dataset = data
        effective_seq_len = min(seq_len, max(8, len(dataset) - 1))
        max_start = len(dataset) - effective_seq_len - 1

    if max_start <= 0:
        raise RuntimeError("Not enough encoded data to create a training batch.")

    index = torch.randint(max_start + 1, (batch_size,))
    x = torch.stack([dataset[i:i + effective_seq_len] for i in index]).to(device)
    y = torch.stack([dataset[i + 1:i + effective_seq_len + 1] for i in index]).to(device)
    return x, y


# ─────────────────────────────────────────
# REWARD-GUIDED TRAINING
# ─────────────────────────────────────────

def tokenize_words(text):
    return set(re.findall(r"[a-z0-9][a-z0-9-]*", text.lower()))


def heuristic_reward(reference_answer, student_answer, source_excerpt):
    reference_words = tokenize_words(reference_answer)
    student_words = tokenize_words(student_answer)
    source_words = tokenize_words(source_excerpt)

    if not student_words:
        return {
            "reward": 0.0,
            "correctness": 0.0,
            "groundedness": 0.0,
            "completeness": 0.0,
            "feedback": "Empty or unparseable student answer.",
            "better_answer": reference_answer,
        }

    overlap = len(reference_words & student_words) / max(1, len(reference_words))
    precision = len(reference_words & student_words) / max(1, len(student_words))
    groundedness = len(student_words & source_words) / max(1, len(student_words)) if source_words else precision
    correctness = 0.65 * overlap + 0.35 * precision
    completeness = min(1.0, len(student_answer) / max(1, len(reference_answer)))
    reward = max(0.0, min(1.0, 0.45 * correctness + 0.35 * groundedness + 0.20 * completeness))
    return {
        "reward": reward,
        "correctness": correctness,
        "groundedness": groundedness,
        "completeness": completeness,
        "feedback": "Fallback lexical reward was used because Ollama grading was unavailable.",
        "better_answer": reference_answer,
    }


def grade_student_answer(example, student_answer):
    if not student_answer.strip():
        return heuristic_reward(example["answer"], student_answer, example["source_excerpt"])

    try:
        prompt = build_reward_prompt(
            question=example["question"],
            reference_answer=example["answer"],
            student_answer=student_answer,
            source_excerpt=example["source_excerpt"] or example["answer"],
        )
        teacher_model, obj = request_teacher_json(prompt, temperature=0.2)
        reward = float(obj.get("reward", 0.0))
        reward = max(0.0, min(1.0, reward))
        correctness = max(0.0, min(1.0, float(obj.get("correctness", reward))))
        groundedness = max(0.0, min(1.0, float(obj.get("groundedness", reward))))
        completeness = max(0.0, min(1.0, float(obj.get("completeness", reward))))
        better_answer = normalize_space(str(obj.get("better_answer", example["answer"]))) or example["answer"]
        feedback = normalize_space(str(obj.get("feedback", "")))
        return {
            "reward": reward,
            "correctness": correctness,
            "groundedness": groundedness,
            "completeness": completeness,
            "feedback": feedback,
            "better_answer": better_answer,
            "teacher_model": teacher_model,
        }
    except Exception:
        return heuristic_reward(example["answer"], student_answer, example["source_excerpt"])


def clean_generated_answer(text):
    text = text.split("\nQ:")[0]
    text = text.split("\n\n")[0]
    return text.strip()


def sample_student_answer(model, prompt_ids, max_new_tokens, temperature=0.9, rep_penalty=1.2):
    token_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    sampled_ids = []
    seen_tokens = set(prompt_ids)

    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            token_ids_crop = token_ids[:, -model.max_seq_len:]
            logits, _ = model(token_ids_crop)
            logits = logits[:, -1, :]
            for token_id in seen_tokens:
                logits[0, token_id] /= rep_penalty
            logits = logits / max(temperature, 1e-5)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

            next_token_id = next_token.item()
            sampled_ids.append(next_token_id)
            seen_tokens.add(next_token_id)
            token_ids = torch.cat([token_ids, next_token.view(1, 1)], dim=1)

            partial_text = decode(sampled_ids)
            if "\nQ:" in partial_text or "\n\n" in partial_text:
                break

    if model_was_training:
        model.train()

    if device.type == "mps":
        torch.mps.empty_cache()

    return clean_generated_answer(decode(sampled_ids)), sampled_ids


def compute_answer_loss(prompt_ids, answer_ids):
    if len(prompt_ids) == 0 or len(answer_ids) == 0:
        return None

    tokens = prompt_ids + answer_ids
    if len(tokens) < 2:
        return None

    input_ids = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
    targets = torch.tensor(tokens[1:], dtype=torch.long, device=device)
    logits, _ = model(input_ids)
    logits = logits[0]

    losses = F.cross_entropy(logits, targets, reduction="none")
    mask = torch.zeros_like(targets, dtype=torch.float)
    answer_start = max(0, len(prompt_ids) - 1)
    mask[answer_start:] = 1.0

    if mask.sum().item() == 0:
        return None

    return (losses * mask).sum() / mask.sum()


def choose_trainable_example():
    random.shuffle(example_pool)
    for item in example_pool:
        prompt_text = f"Q: {item['question']}\nA:"
        answer_text = " " + item["answer"]
        prompt_ids = encode(prompt_text)
        answer_ids = encode(answer_text)
        if prompt_ids and answer_ids:
            return item, prompt_ids, answer_ids
    return None, None, None


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
best_val_loss = float("inf")
patience = 5
strikes = 0
recent_rewards = []

print("\nReward-guided fine-tuning on dynamic teacher Q&A...")
print("Teacher models:", ", ".join(TEACHER_MODELS))
print("Source file:", SOURCE_FILE)
print("─" * 60)

for step in range(TRAIN_STEPS):
    if step > 0 and step % POOL_REFRESH_INTERVAL == 0:
        print(f"\nRefreshing teacher pool at step {step}")
        new_examples = expand_example_pool(source_chunks, example_pool, DYNAMIC_GROWTH_PAIRS)
        if new_examples:
            example_pool.extend(new_examples)
            example_pool = dedupe_examples(example_pool)
            sync_dataset_file(example_pool)
            data, train_data, val_data = rebuild_char_dataset(example_pool)
            print(f"Pool expanded to {len(example_pool)} pairs")
        else:
            print("Teacher pool refresh did not add new pairs")

    x, y = get_batch("train")
    _, base_loss = model(x, y)
    total_loss = base_loss
    reward_info = None
    student_answer = None

    if step % REWARD_EVAL_INTERVAL == 0:
        example, prompt_ids, answer_ids = choose_trainable_example()
        if example is not None:
            student_answer, sampled_ids = sample_student_answer(
                model,
                prompt_ids,
                max_new_tokens=STUDENT_ANSWER_CHARS,
            )
            reward_info = grade_student_answer(example, student_answer)
            target_answer = reward_info.get("better_answer", example["answer"]) or example["answer"]
            guided_answer_ids = encode(" " + target_answer)
            guided_loss = compute_answer_loss(prompt_ids, guided_answer_ids)

            if guided_loss is not None:
                reward = reward_info["reward"]
                reward_scale = 1.0 + REWARD_WEIGHT * (1.0 - reward)
                total_loss = total_loss + guided_loss * reward_scale
                recent_rewards.append(reward)
                recent_rewards = recent_rewards[-50:]

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if device.type == "mps" and step % 25 == 0:
        torch.mps.empty_cache()

    if step % 100 == 0:
        avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0
        reward_text = f" | reward: {avg_reward:.3f}" if recent_rewards else ""
        print(f"Step {step:4d} | base: {base_loss.item():.4f}{reward_text}")
        if reward_info is not None:
            preview = student_answer[:120].replace("\n", " ")
            print(
                f"  reward={reward_info['reward']:.3f} "
                f"correct={reward_info['correctness']:.3f} "
                f"grounded={reward_info['groundedness']:.3f} "
                f"complete={reward_info['completeness']:.3f}"
            )
            print(f"  feedback: {reward_info['feedback'][:160]}")
            print(f"  student: {preview}")

    if step % 300 == 0:
        x_val, y_val = get_batch("val")
        _, val_loss = model(x_val, y_val)
        avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0
        print(
            f"Validation step {step:4d} | train: {base_loss.item():.4f} "
            f"| val: {val_loss.item():.4f} | avg reward: {avg_reward:.3f}"
        )

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            strikes = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "char_to_int": char_to_int,
                    "int_to_char": int_to_char,
                    "vocab_size": vocab_size,
                    "teacher_models": TEACHER_MODELS,
                    "qa_source_file": SOURCE_FILE,
                    "qa_dataset_file": QA_DATASET_FILE,
                    "reward_average": avg_reward,
                    "example_pool_size": len(example_pool),
                },
                "model_qa.pt",
            )
            print("★ New best! Saved → model_qa.pt")
        else:
            strikes += 1
            print(f"No improvement ({strikes}/{patience})")
            if strikes >= patience:
                print(f"\n★ Early stopping. Best val loss: {best_val_loss:.4f}")
                break


# ─────────────────────────────────────────
# TEST Q&A STYLE GENERATION
# ─────────────────────────────────────────

print("\n" + "═" * 60)
print("Testing reward-trained Q&A responses:")
print("═" * 60)

test_questions = [f"Q: {item['question']}\nA:" for item in example_pool[:5]]

model.eval()
with torch.no_grad():
    for question in test_questions:
        context = torch.tensor([encode(question)], dtype=torch.long).to(device)
        output = model.generate(context, max_new_tokens=150, temperature=0.7)
        print(f"\n{question}")
        full = decode(output[0].tolist())
        answer = clean_generated_answer(full[len(question):])
        print(answer[:300])
        print("─" * 40)
