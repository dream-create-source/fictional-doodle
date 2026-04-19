import collections
import glob
import json
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_CONFIG = {
    "embed_dim": 256,
    "num_heads": 8,
    "num_layers": 8,
    "max_seq_len": 256,
}

TOKENIZER_CONFIG = {
    "vocab_size": 8192,
    "special_tokens": ["<pad>", "<unk>", "<bos>", "<eos>", "<nl>"],
}

TOKEN_SPLIT_RE = re.compile(r"\s+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,:/-]\d+)*|[^\w\s]", re.UNICODE)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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
        tokens = payload["tokens"]
        token_to_id = {token: idx for idx, token in enumerate(tokens)}
        return cls(token_to_id)


def parse_env_paths(value):
    return [path.strip() for path in value.split(",") if path.strip()]


def expand_source_paths(paths):
    expanded = []
    for path in paths:
        if os.path.isdir(path):
            txt_files = sorted(glob.glob(os.path.join(path, "**", "*.txt"), recursive=True))
            json_files = sorted(glob.glob(os.path.join(path, "**", "*.json"), recursive=True))
            expanded.extend(txt_files + json_files)
        else:
            expanded.append(path)
    return list(dict.fromkeys(expanded))


def read_text_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        return file.read().strip()


def format_json_record(record):
    lines = []
    for key, value in record.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            text = str(value).strip()
            if text:
                lines.append(f"{key}: {text}")
    return "\n".join(lines)


def load_json_as_text(path, max_records=0):
    with open(path, "r", encoding="utf-8") as file:
        obj = json.load(file)

    if isinstance(obj, dict) and isinstance(obj.get("results"), list):
        records = obj["results"]
        if max_records > 0:
            records = records[:max_records]
        blocks = [format_json_record(record) for record in records if isinstance(record, dict)]
        header = []
        for key in ("exported_at", "exported_from", "count", "disclaimer"):
            value = obj.get(key)
            if value:
                header.append(f"{key}: {value}")
        joined = "\n\n".join(header + blocks)
        return joined.strip()

    return json.dumps(obj, ensure_ascii=False, indent=2)


def load_corpus_from_paths(paths, json_max_records=0):
    texts = []
    loaded_paths = []
    for path in expand_source_paths(paths):
        if not os.path.exists(path):
            print(f"Skipping missing file: {path}")
            continue
        if path.lower().endswith(".json"):
            text = load_json_as_text(path, max_records=json_max_records)
        else:
            text = read_text_file(path)
        text = text.strip()
        if not text:
            continue
        texts.append(text)
        loaded_paths.append(path)
    if not texts:
        raise FileNotFoundError(f"No readable sources found in: {paths}")
    return "\n\n".join(texts), loaded_paths


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
    top_tokens = [token for token, _ in token_counter.most_common() if token not in special_tokens and token not in ordered_fallback]
    tokens = special_tokens + top_tokens[:available_slots] + ordered_fallback
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return LocalTokenizer(token_to_id)


def save_tokenizer(path, tokenizer, extra=None):
    payload = tokenizer.to_dict()
    payload["tokenizer_config"] = {
        "vocab_size": tokenizer.vocab_size(),
    }
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def load_tokenizer(path):
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    return LocalTokenizer.from_dict(payload)


def encode_text(text, tokenizer, add_bos=False, add_eos=False):
    ids = tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
    return ids, 0


def split_tensor(data, ratio=0.9):
    split = max(1, int(ratio * len(data)))
    train_data = data[:split]
    val_data = data[split:] if split < len(data) else data[-max(64, len(data) // 10):]
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


def save_checkpoint(path, model, optimizer, tokenizer, extra=None):
    payload = {
        "model_state": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "token_to_id": tokenizer.token_to_id,
        "id_to_token": tokenizer.id_to_token,
        "vocab_size": tokenizer.vocab_size(),
        "model_config": MODEL_CONFIG,
        "tokenizer_type": "local_wordpiece_fallback",
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def build_model(vocab_size, device):
    return GPT(
        vocab_size=vocab_size,
        embed_dim=MODEL_CONFIG["embed_dim"],
        num_heads=MODEL_CONFIG["num_heads"],
        num_layers=MODEL_CONFIG["num_layers"],
        max_seq_len=MODEL_CONFIG["max_seq_len"],
    ).to(device)


def load_tokenizer_from_checkpoint(checkpoint):
    if "token_to_id" not in checkpoint:
        raise KeyError(
            "This checkpoint does not contain tokenizer metadata. "
            "It looks like an older character-level checkpoint. "
            "Retrain project1 from train_tokenizer.py and train_stage1.py onward."
        )
    token_to_id = checkpoint["token_to_id"]
    return LocalTokenizer(token_to_id)


def load_model_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("model_config", MODEL_CONFIG)
    model = GPT(
        vocab_size=checkpoint["vocab_size"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        max_seq_len=config["max_seq_len"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint, model


def model_param_count(vocab_size):
    model = GPT(
        vocab_size=vocab_size,
        embed_dim=MODEL_CONFIG["embed_dim"],
        num_heads=MODEL_CONFIG["num_heads"],
        num_layers=MODEL_CONFIG["num_layers"],
        max_seq_len=MODEL_CONFIG["max_seq_len"],
    )
    return sum(param.numel() for param in model.parameters())


def clean_generated_answer(text):
    text = text.split("\nQ:")[0]
    text = text.split("\n\n")[0]
    return text.strip()
