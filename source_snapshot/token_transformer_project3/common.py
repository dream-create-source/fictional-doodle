import bz2
import collections
import glob
import html
import json
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_CONFIG = {
    "embed_dim": 768,
    "num_heads": 12,
    "num_layers": 10,
    "max_seq_len": 256,
    "dropout": 0.1,
}

TOKENIZER_CONFIG = {
    "vocab_size": 8192,
    "special_tokens": ["<pad>", "<unk>", "<bos>", "<eos>", "<nl>"],
}

TOKEN_SPLIT_RE = re.compile(r"\s+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,:/-]\d+)*|[^\w\s]", re.UNICODE)
TAG_RE = re.compile(r"<[^>]+>")
WIKI_LINK_WITH_LABEL_RE = re.compile(r"\[\[[^|\]]+\|([^\]]+)\]\]")
WIKI_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
WIKI_TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
WIKI_FILE_RE = re.compile(r"\[\[(?:File|Image):[^\]]+\]\]", re.IGNORECASE)
WIKI_CATEGORY_RE = re.compile(r"\[\[(?:Category):([^\]]+)\]\]", re.IGNORECASE)
EXTERNAL_LINK_RE = re.compile(r"\[(https?://[^\s\]]+)\s+([^\]]+)\]")
MULTISPACE_RE = re.compile(r"[ \t]+")
MULTINEWLINE_RE = re.compile(r"\n{3,}")


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, channels = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if hasattr(F, "scaled_dot_product_attention"):
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float("-inf"))
            weights = F.softmax(scores, dim=-1)
            weights = self.attn_dropout(weights)
            attn_output = torch.matmul(weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.resid_dropout(self.out_proj(attn_output))


class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout=dropout)
        self.ff = FeedForward(embed_dim, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len, dropout=0.1):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, token_ids, targets=None):
        _, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            batch_size, seq_len, vocab_size = logits.shape
            loss = F.cross_entropy(
                logits.view(batch_size * seq_len, vocab_size),
                targets.view(batch_size * seq_len),
            )
        return logits, loss

    def generate(self, token_ids, max_new_tokens, temperature=0.8):
        for _ in range(max_new_tokens):
            token_ids_crop = token_ids[:, -self.max_seq_len:]
            logits, _ = self.forward(token_ids_crop)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_ids = torch.cat([token_ids, next_token], dim=1)
        return token_ids


class LocalTokenizer:
    def __init__(self, token_to_id):
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self.pad_id = self.token_to_id["<pad>"]
        self.unk_id = self.token_to_id["<unk>"]
        self.bos_id = self.token_to_id["<bos>"]
        self.eos_id = self.token_to_id["<eos>"]
        self.nl_id = self.token_to_id["<nl>"]

    def vocab_size(self):
        return len(self.token_to_id)

    def _pretokenize(self, text):
        pieces = []
        needs_space = False
        at_start = True
        for match in TOKEN_SPLIT_RE.finditer(text):
            piece = match.group(0)
            if piece.isspace():
                newline_count = piece.count("\n")
                for _ in range(newline_count):
                    pieces.append("<nl>")
                needs_space = True
                at_start = newline_count > 0 or at_start
                continue

            token = f"▁{piece}" if needs_space or at_start else piece
            pieces.append(token)
            needs_space = False
            at_start = False
        return pieces

    def encode(self, text, add_bos=False, add_eos=False):
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        for piece in self._pretokenize(text):
            if piece in self.token_to_id:
                ids.append(self.token_to_id[piece])
                continue
            if piece == "<nl>":
                ids.append(self.nl_id)
                continue
            if piece.startswith("▁") and len(piece) > 1:
                raw = piece[1:]
                for index, char in enumerate(raw):
                    fallback_piece = f"▁{char}" if index == 0 else char
                    ids.append(self.token_to_id.get(fallback_piece, self.unk_id))
                continue
            for char in piece:
                ids.append(self.token_to_id.get(char, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids, skip_special_tokens=True):
        text = []
        at_line_start = True
        for idx in ids:
            token = self.id_to_token.get(int(idx), "<unk>")
            if skip_special_tokens and token in {"<pad>", "<bos>", "<eos>"}:
                continue
            if token == "<nl>":
                text.append("\n")
                at_line_start = True
                continue
            if token.startswith("▁"):
                piece = token[1:]
                if text and not at_line_start:
                    text.append(" ")
                text.append(piece)
                at_line_start = False
                continue
            text.append(token)
            at_line_start = False
        return "".join(text)

    def to_dict(self):
        ordered_tokens = [token for token, _ in sorted(self.token_to_id.items(), key=lambda item: item[1])]
        return {
            "tokens": ordered_tokens,
            "special_tokens": TOKENIZER_CONFIG["special_tokens"],
        }

    @classmethod
    def from_dict(cls, payload):
        return cls({token: idx for idx, token in enumerate(payload["tokens"])})


def parse_env_paths(value):
    return [path.strip() for path in value.split(",") if path.strip()]


def expand_source_paths(paths):
    expanded = []
    for path in paths:
        if os.path.isdir(path):
            txt_files = sorted(glob.glob(os.path.join(path, "**", "*.txt"), recursive=True))
            json_files = sorted(glob.glob(os.path.join(path, "**", "*.json"), recursive=True))
            jsonl_files = sorted(glob.glob(os.path.join(path, "**", "*.jsonl"), recursive=True))
            xml_files = sorted(glob.glob(os.path.join(path, "**", "*.xml"), recursive=True))
            bz2_files = sorted(glob.glob(os.path.join(path, "**", "*.bz2"), recursive=True))
            expanded.extend(txt_files + json_files + jsonl_files + xml_files + bz2_files)
        else:
            expanded.append(path)

    expanded = list(dict.fromkeys(expanded))
    existing = set(expanded)
    filtered = []
    for path in expanded:
        if path.endswith(".bz2") and path[:-4] in existing:
            continue
        filtered.append(path)
    return filtered


def clean_wiki_text(text):
    text = html.unescape(text)
    text = TAG_RE.sub(" ", text)
    text = WIKI_FILE_RE.sub(" ", text)
    text = WIKI_CATEGORY_RE.sub(r" \1 ", text)
    for _ in range(3):
        updated = WIKI_TEMPLATE_RE.sub(" ", text)
        if updated == text:
            break
        text = updated
    text = WIKI_LINK_WITH_LABEL_RE.sub(r" \1 ", text)
    text = WIKI_LINK_RE.sub(r" \1 ", text)
    text = EXTERNAL_LINK_RE.sub(r" \2 ", text)
    text = text.replace("'''", "")
    text = text.replace("''", "")
    text = re.sub(r"^=+\s*(.*?)\s*=+$", r"\1", text, flags=re.MULTILINE)
    text = MULTISPACE_RE.sub(" ", text)
    text = MULTINEWLINE_RE.sub("\n\n", text)
    return text.strip()


def read_text_file(path, max_chars=0):
    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        return (file.read(max_chars) if max_chars > 0 else file.read()).strip()


def read_bz2_text_file(path, max_chars=0):
    with bz2.open(path, "rt", encoding="utf-8", errors="ignore") as file:
        return (file.read(max_chars) if max_chars > 0 else file.read()).strip()


def format_json_record(record):
    lines = []
    for key, value in record.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            text = str(value).strip()
            if text:
                lines.append(f"{key}: {text}")
        elif isinstance(value, list):
            joined = "; ".join(str(item).strip() for item in value if str(item).strip())
            if joined:
                lines.append(f"{key}: {joined}")
    return "\n".join(lines)


def format_instruction_record(record):
    if not isinstance(record, dict):
        return None

    if "instruction" in record and "response" in record:
        user_text = str(record.get("instruction", "")).strip()
        context = str(record.get("context", "")).strip()
        assistant_text = str(record.get("response", "")).strip()
        if context:
            user_text = f"{user_text}\n\nContext:\n{context}"
    elif "prompt" in record and "completion" in record:
        user_text = str(record.get("prompt", "")).strip()
        assistant_text = str(record.get("completion", "")).strip()
    elif "question" in record and "answer" in record:
        user_text = str(record.get("question", "")).strip()
        assistant_text = str(record.get("answer", "")).strip()
    else:
        return None

    if not user_text or not assistant_text:
        return None
    return f"User: {user_text}\nAssistant: {assistant_text}"


def load_json_as_text(path, max_records=0):
    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        obj = json.load(file)

    if isinstance(obj, dict) and isinstance(obj.get("results"), list):
        records = obj["results"][:max_records] if max_records > 0 else obj["results"]
        blocks = [format_json_record(record) for record in records if isinstance(record, dict)]
        return "\n\n".join(block for block in blocks if block).strip()

    if isinstance(obj, list):
        records = obj[:max_records] if max_records > 0 else obj
        blocks = []
        for record in records:
            block = format_instruction_record(record) or format_json_record(record)
            if block:
                blocks.append(block)
        return "\n\n".join(blocks).strip()

    if isinstance(obj, dict):
        block = format_instruction_record(obj) or format_json_record(obj)
        return block.strip()
    return str(obj).strip()


def load_jsonl_as_text(path, max_records=0, instruction_mode=False):
    blocks = []
    record_count = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if max_records > 0 and record_count >= max_records:
                break

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                blocks.append(line)
                record_count += 1
                continue

            if instruction_mode:
                block = format_instruction_record(obj)
                if block:
                    blocks.append(block)
            else:
                if isinstance(obj, dict):
                    block = format_json_record(obj)
                    if block:
                        blocks.append(block)
                else:
                    blocks.append(str(obj))
            record_count += 1
    return "\n\n".join(blocks).strip()


def load_source_text(path, json_max_records=0, max_chars_per_file=0, instruction_mode=False):
    path_lower = path.lower()
    if path_lower.endswith(".json"):
        text = load_json_as_text(path, max_records=json_max_records)
    elif path_lower.endswith(".jsonl"):
        text = load_jsonl_as_text(path, max_records=json_max_records, instruction_mode=instruction_mode)
    elif path_lower.endswith(".bz2"):
        text = read_bz2_text_file(path, max_chars=max_chars_per_file)
    elif path_lower.endswith(".xml"):
        text = read_text_file(path, max_chars=max_chars_per_file)
    elif path_lower.endswith(".parquet") or path_lower.endswith(".7z"):
        return ""
    else:
        text = read_text_file(path, max_chars=max_chars_per_file)

    if path_lower.endswith(".xml") or path_lower.endswith(".bz2"):
        text = clean_wiki_text(text)
    return text.strip()


def load_corpus_from_paths(paths, json_max_records=0, max_chars_per_file=0, total_char_cap=0, instruction_mode=False):
    texts = []
    loaded_paths = []
    total_chars = 0
    for path in expand_source_paths(paths):
        if not os.path.exists(path):
            print(f"Skipping missing file: {path}")
            continue

        text = load_source_text(
            path,
            json_max_records=json_max_records,
            max_chars_per_file=max_chars_per_file,
            instruction_mode=instruction_mode,
        )
        if not text:
            continue

        if total_char_cap > 0:
            remaining = total_char_cap - total_chars
            if remaining <= 0:
                break
            if len(text) > remaining:
                text = text[:remaining]

        texts.append(text)
        loaded_paths.append(path)
        total_chars += len(text)

        if total_char_cap > 0 and total_chars >= total_char_cap:
            break

    if not texts:
        raise FileNotFoundError(f"No readable sources found in: {paths}")
    return "\n\n".join(texts), loaded_paths


def iter_source_chunks(paths, json_max_records=0, chunk_chars=4000000, total_char_cap=0, instruction_mode=False):
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be greater than 0")

    total_chars = 0
    total_json_records = 0

    for path in expand_source_paths(paths):
        if not os.path.exists(path):
            print(f"Skipping missing file: {path}")
            continue

        path_lower = path.lower()

        if path_lower.endswith(".parquet") or path_lower.endswith(".7z"):
            continue

        if path_lower.endswith(".jsonl"):
            buffer = []
            buffer_chars = 0
            record_count = 0
            with open(path, "r", encoding="utf-8", errors="ignore") as file:
                for line in file:
                    if json_max_records > 0 and total_json_records >= json_max_records:
                        break
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        block = line
                    else:
                        if instruction_mode:
                            block = format_instruction_record(obj)
                        elif isinstance(obj, dict):
                            block = format_json_record(obj)
                        else:
                            block = str(obj)

                    if not block:
                        continue

                    total_json_records += 1
                    record_count += 1
                    block = block.strip()
                    if not block:
                        continue

                    if total_char_cap > 0:
                        remaining = total_char_cap - total_chars
                        if remaining <= 0:
                            break
                        if len(block) > remaining:
                            block = block[:remaining]

                    buffer.append(block)
                    buffer_chars += len(block) + 2
                    total_chars += len(block)

                    if buffer_chars >= chunk_chars:
                        yield path, "\n\n".join(buffer).strip()
                        buffer = []
                        buffer_chars = 0

                    if total_char_cap > 0 and total_chars >= total_char_cap:
                        break

            if buffer:
                yield path, "\n\n".join(buffer).strip()

            if (total_char_cap > 0 and total_chars >= total_char_cap) or (json_max_records > 0 and total_json_records >= json_max_records):
                break
            continue

        if path_lower.endswith(".json"):
            text = load_json_as_text(path, max_records=json_max_records)
            if not text:
                continue
            start = 0
            while start < len(text):
                if total_char_cap > 0:
                    remaining = total_char_cap - total_chars
                    if remaining <= 0:
                        break
                    size = min(chunk_chars, remaining)
                else:
                    size = chunk_chars
                chunk = text[start:start + size].strip()
                start += size
                if not chunk:
                    continue
                total_chars += len(chunk)
                yield path, chunk
            if total_char_cap > 0 and total_chars >= total_char_cap:
                break
            continue

        open_fn = bz2.open if path_lower.endswith(".bz2") else open
        with open_fn(path, "rt", encoding="utf-8", errors="ignore") as file:
            while True:
                if total_char_cap > 0:
                    remaining = total_char_cap - total_chars
                    if remaining <= 0:
                        break
                    read_chars = min(chunk_chars, remaining)
                else:
                    read_chars = chunk_chars

                raw = file.read(read_chars)
                if not raw:
                    break

                chunk = clean_wiki_text(raw) if (path_lower.endswith(".xml") or path_lower.endswith(".bz2")) else raw.strip()
                if not chunk:
                    continue

                total_chars += len(chunk)
                yield path, chunk

            if total_char_cap > 0 and total_chars >= total_char_cap:
                break


def build_tokenizer_from_text(text, vocab_size=8192):
    special_tokens = TOKENIZER_CONFIG["special_tokens"][:]
    token_counter = collections.Counter()
    fallback_pieces = set(special_tokens)
    temp_tokenizer = LocalTokenizer({token: idx for idx, token in enumerate(special_tokens)})

    for piece in temp_tokenizer._pretokenize(text):
        token_counter[piece] += 1
        if piece == "<nl>":
            continue
        if piece.startswith("▁") and len(piece) > 1:
            raw = piece[1:]
            for index, char in enumerate(raw):
                fallback_pieces.add(f"▁{char}" if index == 0 else char)
        else:
            for char in piece:
                fallback_pieces.add(char)

    ordered_fallback = sorted(fallback_pieces - set(special_tokens))
    available_slots = max(0, vocab_size - len(special_tokens) - len(ordered_fallback))
    frequent_tokens = [
        token
        for token, _ in token_counter.most_common()
        if token not in special_tokens and token not in ordered_fallback
    ]
    tokens = special_tokens + frequent_tokens[:available_slots] + ordered_fallback
    return LocalTokenizer({token: idx for idx, token in enumerate(tokens)})


def save_tokenizer(path, tokenizer, extra=None):
    payload = tokenizer.to_dict()
    payload["tokenizer_config"] = {"vocab_size": tokenizer.vocab_size()}
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def load_tokenizer(path):
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    return LocalTokenizer.from_dict(payload)


def encode_text(text, tokenizer, add_bos=False, add_eos=False):
    return tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos), 0


def split_tensor(data, ratio=0.9):
    split = max(1, int(ratio * len(data)))
    train_data = data[:split]
    val_data = data[split:] if split < len(data) else data[-max(128, len(data) // 10):]
    return train_data, val_data


def get_batch(data, train_data, val_data, split_name, batch_size, seq_len, device):
    dataset = train_data if split_name == "train" else val_data
    effective_seq_len = min(seq_len, max(8, len(dataset) - 1))
    max_start = len(dataset) - effective_seq_len - 1
    if max_start <= 0:
        dataset = data
        effective_seq_len = min(seq_len, max(8, len(dataset) - 1))
        max_start = len(dataset) - effective_seq_len - 1
    if max_start <= 0:
        raise RuntimeError("Dataset is too small for batching.")
    indices = torch.randint(max_start + 1, (batch_size,))
    x = torch.stack([dataset[i:i + effective_seq_len] for i in indices]).to(device)
    y = torch.stack([dataset[i + 1:i + effective_seq_len + 1] for i in indices]).to(device)
    return x, y


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_checkpoint(path, model, optimizer, tokenizer, extra=None, include_optimizer=False):
    payload = {
        "model_state": model.state_dict(),
        "optimizer": optimizer.state_dict() if include_optimizer and optimizer is not None else None,
        "token_to_id": tokenizer.token_to_id,
        "id_to_token": tokenizer.id_to_token,
        "vocab_size": tokenizer.vocab_size(),
        "model_config": MODEL_CONFIG,
        "architecture": "transformer_lm",
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def build_model(vocab_size, device):
    return GPTLanguageModel(
        vocab_size=vocab_size,
        embed_dim=MODEL_CONFIG["embed_dim"],
        num_heads=MODEL_CONFIG["num_heads"],
        num_layers=MODEL_CONFIG["num_layers"],
        max_seq_len=MODEL_CONFIG["max_seq_len"],
        dropout=MODEL_CONFIG["dropout"],
    ).to(device)


def load_tokenizer_from_checkpoint(checkpoint):
    if "token_to_id" not in checkpoint:
        raise KeyError("This checkpoint does not contain tokenizer metadata.")
    return LocalTokenizer(checkpoint["token_to_id"])


def load_model_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("model_config", MODEL_CONFIG)
    model = GPTLanguageModel(
        vocab_size=checkpoint["vocab_size"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        max_seq_len=config["max_seq_len"],
        dropout=config.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return checkpoint, model


def model_param_count(vocab_size):
    model = GPTLanguageModel(
        vocab_size=vocab_size,
        embed_dim=MODEL_CONFIG["embed_dim"],
        num_heads=MODEL_CONFIG["num_heads"],
        num_layers=MODEL_CONFIG["num_layers"],
        max_seq_len=MODEL_CONFIG["max_seq_len"],
        dropout=MODEL_CONFIG["dropout"],
    )
    return sum(param.numel() for param in model.parameters())


def maybe_clear_mps_cache():
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def clean_generated_answer(text):
    for stop_marker in ("\nUser:", "\n### User:", "\nInstruction:", "\nQ:"):
        text = text.split(stop_marker)[0]
    return text.strip()
