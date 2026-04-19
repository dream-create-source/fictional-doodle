import glob
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

TRAIN_SOURCE_PATHS = [
    path.strip()
    for path in os.getenv(
        "TRAIN_SOURCE_PATHS",
        "data/english_basic,shakespeare.txt",
    ).split(",")
    if path.strip()
]
VOCAB_SOURCE_PATHS = [
    path.strip()
    for path in os.getenv(
        "VOCAB_SOURCE_PATHS",
        "data/english_basic,shakespeare.txt,quotes_clean.txt,qa_dataset.txt",
    ).split(",")
    if path.strip()
]

OUTPUT_MODEL = os.getenv("OUTPUT_MODEL", "model_english.pt")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "models_english")
TRAIN_STEPS = int(os.getenv("TRAIN_STEPS", "5000"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
SEQ_LEN = int(os.getenv("SEQ_LEN", "128"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "3e-4"))

# ─────────────────────────────────────────
# DEVICE SETUP
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
# MODEL COMPONENTS  (classes must come first)
# ─────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        
        self.query    = nn.Linear(embed_dim, embed_dim)
        self.key      = nn.Linear(embed_dim, embed_dim)
        self.value    = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        B, T, C = x.shape
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (self.head_dim ** 0.5)
        
        # ← device fix
        mask   = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
        scores = scores.masked_fill(mask, float('-inf'))
        
        weights  = F.softmax(scores, dim=-1)
        attended = torch.matmul(weights, V)
        
        attended = attended.transpose(1, 2).contiguous().view(B, T, C)
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
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention   = MultiHeadAttention(embed_dim, num_heads)
        self.feedforward = FeedForward(embed_dim)
        self.norm1       = nn.LayerNorm(embed_dim)
        self.norm2       = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feedforward(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len):
        super().__init__()
        self.max_seq_len        = max_seq_len
        self.token_embedding    = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.blocks             = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.norm    = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
    
    def forward(self, token_ids, targets=None):
        B, T = token_ids.shape
        
        tok_emb   = self.token_embedding(token_ids)
        positions = torch.arange(T, device=token_ids.device)
        pos_emb   = self.position_embedding(positions)
        
        x      = tok_emb + pos_emb
        x      = self.blocks(x)
        x      = self.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            B, T, V      = logits.shape
            logits_flat  = logits.view(B * T, V)
            targets_flat = targets.view(B * T)
            loss         = F.cross_entropy(logits_flat, targets_flat)
        
        return logits, loss
    
    def generate(self, token_ids, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            token_ids_crop = token_ids[:, -self.max_seq_len:]
            logits, _      = self.forward(token_ids_crop)
            logits         = logits[:, -1, :] / temperature
            probs          = F.softmax(logits, dim=-1)
            next_token     = torch.multinomial(probs, num_samples=1)
            token_ids      = torch.cat([token_ids, next_token], dim=1)
        return token_ids


# ─────────────────────────────────────────
# DATA
# ─────────────────────────────────────────

def expand_source_paths(paths):
    expanded = []

    for path in paths:
        if os.path.isdir(path):
            txt_files = sorted(glob.glob(os.path.join(path, "**", "*.txt"), recursive=True))
            if txt_files:
                expanded.extend(txt_files)
            else:
                print(f"Skipping empty directory: {path}")
            continue
        expanded.append(path)

    # Keep stable ordering while removing duplicates.
    return list(dict.fromkeys(expanded))


def load_corpus(paths):
    loaded_paths = []
    parts = []

    for path in expand_source_paths(paths):
        if not os.path.exists(path):
            print(f"Skipping missing file: {path}")
            continue

        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            parts.append(f.read().strip())
        loaded_paths.append(path)

    if not parts:
        raise FileNotFoundError(f"No readable training files found in: {paths}")

    return "\n\n".join(parts), loaded_paths


train_text, loaded_train_paths = load_corpus(TRAIN_SOURCE_PATHS)
print("Training corpus:")
for path in loaded_train_paths:
    print(f"  - {path}")

try:
    vocab_text, loaded_vocab_paths = load_corpus(VOCAB_SOURCE_PATHS)
except FileNotFoundError:
    vocab_text = train_text
    loaded_vocab_paths = loaded_train_paths

print("Vocabulary sources:")
for path in loaded_vocab_paths:
    print(f"  - {path}")

chars      = sorted(set(vocab_text))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size} characters")

char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}

encode = lambda s: [char_to_int[c] for c in s if c in char_to_int]
decode = lambda l: ''.join([int_to_char[i] for i in l])

data = torch.tensor(encode(train_text), dtype=torch.long)
print(f"English dataset size: {len(data):,} tokens")

split      = int(0.9 * len(data))
train_data = data[:split]
val_data   = data[split:]

def get_batch(split, batch_size=BATCH_SIZE, seq_len=SEQ_LEN):
    d = train_data if split == 'train' else val_data
    effective_seq_len = min(seq_len, max(8, len(d) - 1))
    max_start = len(d) - effective_seq_len - 1

    if max_start <= 0:
        d = data
        effective_seq_len = min(seq_len, max(8, len(d) - 1))
        max_start = len(d) - effective_seq_len - 1

    if max_start <= 0:
        raise RuntimeError("Dataset is too small for batching. Add more English text.")

    ix = torch.randint(max_start + 1, (batch_size,))
    x = torch.stack([d[i:i+effective_seq_len]     for i in ix]).to(device)
    y = torch.stack([d[i+1:i+effective_seq_len+1] for i in ix]).to(device)
    return x, y


# ─────────────────────────────────────────
# MODEL  (created AFTER classes are defined)
# ─────────────────────────────────────────

model = GPT(
    vocab_size   = vocab_size,
    embed_dim    = 512,
    num_heads    = 16,
    num_layers   = 12,
    max_seq_len  = 512
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


# ─────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("\nStage 1: English pretraining")
print(f"Checkpoints directory: {CHECKPOINT_DIR}")
print(f"Final model path: {OUTPUT_MODEL}")

for step in range(TRAIN_STEPS):
    x, y = get_batch('train')
    
    logits, loss = model(x, y)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if step % 500 == 0:
        x_val, y_val = get_batch('val')
        _, val_loss  = model(x_val, y_val)
        print(f"Step {step:4d} | train loss: {loss.item():.4f} | val loss: {val_loss.item():.4f}")
        
        torch.save({
            'step':        step,
            'model_state': model.state_dict(),
            'optimizer':   optimizer.state_dict(),
            'train_loss':  loss.item(),
            'val_loss':    val_loss.item(),
            'char_to_int': char_to_int,
            'int_to_char': int_to_char,
            'vocab_size':  vocab_size,
            'stage':       'english_pretrain',
            'train_source_files': loaded_train_paths,
            'vocab_source_files': loaded_vocab_paths,
        }, os.path.join(CHECKPOINT_DIR, f'model_step_{step}.pt'))
        print(f"Checkpoint saved → {os.path.join(CHECKPOINT_DIR, f'model_step_{step}.pt')}")


# ─────────────────────────────────────────
# GENERATE
# ─────────────────────────────────────────

prompts = ["To be", "What is", "The mind"]

for prompt in prompts:
    context = torch.tensor([encode(prompt)], dtype=torch.long).to(device)  # ← device
    output  = model.generate(context, max_new_tokens=200, temperature=0.8)
    print(f"\n--- Prompt: '{prompt}' ---")
    print(decode(output[0].tolist()))


# ─────────────────────────────────────────
# SAVE FINAL MODEL
# ─────────────────────────────────────────

torch.save({
    'step':        TRAIN_STEPS,
    'model_state': model.state_dict(),
    'optimizer':   optimizer.state_dict(),
    'char_to_int': char_to_int,
    'int_to_char': int_to_char,
    'vocab_size':  vocab_size,
    'stage':       'english_pretrain',
    'train_source_files': loaded_train_paths,
    'vocab_source_files': loaded_vocab_paths,
}, OUTPUT_MODEL)

print(f"\nFinal model saved → {OUTPUT_MODEL}")
