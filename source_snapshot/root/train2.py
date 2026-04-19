import torch
import torch.nn as nn
import torch.nn.functional as F
import os

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
        self.head_dim  = embed_dim // num_heads
        self.query     = nn.Linear(embed_dim, embed_dim)
        self.key       = nn.Linear(embed_dim, embed_dim)
        self.value     = nn.Linear(embed_dim, embed_dim)
        self.out_proj  = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.shape
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        scores   = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask     = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
        scores   = scores.masked_fill(mask, float('-inf'))
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
    def __init__(self, embed_dim, num_heads, dropout=0.2):
        super().__init__()
        self.attention   = MultiHeadAttention(embed_dim, num_heads)
        self.feedforward = FeedForward(embed_dim)
        self.norm1       = nn.LayerNorm(embed_dim)
        self.norm2       = nn.LayerNorm(embed_dim)
        self.dropout     = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.feedforward(self.norm2(x)))
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
        B, T      = token_ids.shape
        tok_emb   = self.token_embedding(token_ids)
        positions = torch.arange(T, device=token_ids.device)
        pos_emb   = self.position_embedding(positions)
        x         = tok_emb + pos_emb
        x         = self.blocks(x)
        x         = self.norm(x)
        logits    = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, V      = logits.shape
            logits_flat  = logits.view(B * T, V)
            targets_flat = targets.view(B * T)
            loss         = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss

    def generate(self, token_ids, max_new_tokens, temperature=1.0, rep_penalty=1.3):
        for _ in range(max_new_tokens):
            token_ids_crop = token_ids[:, -self.max_seq_len:]
            logits, _      = self.forward(token_ids_crop)
            logits         = logits[:, -1, :]
            # Repetition penalty
            for token_id in set(token_ids[0].tolist()):
                logits[0, token_id] /= rep_penalty
            logits     = logits / temperature
            probs      = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_ids  = torch.cat([token_ids, next_token], dim=1)
        return token_ids


# ─────────────────────────────────────────
# DATA
# ─────────────────────────────────────────

with open('training_data.txt', 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()

chars       = sorted(set(text))
vocab_size  = len(chars)
print(f"Vocabulary size: {vocab_size} characters")

char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}
encode      = lambda s: [char_to_int[c] for c in s if c in char_to_int]
decode      = lambda l: ''.join([int_to_char[i] for i in l])

data        = torch.tensor(encode(text), dtype=torch.long)
print(f"Dataset size: {len(data):,} tokens")

split       = int(0.9 * len(data))
train_data  = data[:split]
val_data    = data[split:]

def get_batch(split, batch_size=64, seq_len=256):
    d  = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - seq_len, (batch_size,))
    x  = torch.stack([d[i:i+seq_len]     for i in ix]).to(device)
    y  = torch.stack([d[i+1:i+seq_len+1] for i in ix]).to(device)
    return x, y


# ─────────────────────────────────────────
# MODEL INIT
# ─────────────────────────────────────────

model = GPT(
    vocab_size   = vocab_size,
    embed_dim    = 256,
    num_heads    = 8,
    num_layers   = 8,
    max_seq_len  = 256,
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max   = 50000,
    eta_min = 1e-5
)

os.makedirs('checkpoints', exist_ok=True)


# ─────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────

best_val_loss = float('inf')
patience      = 5
strikes       = 0

for step in range(50000):
    x, y         = get_batch('train')
    logits, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if step % 1000 == 0:
        x_val, y_val = get_batch('val')
        _, val_loss  = model(x_val, y_val)
        lr_now       = scheduler.get_last_lr()[0]
        print(f"Step {step:5d} | train: {loss.item():.4f} | val: {val_loss.item():.4f} | lr: {lr_now:.6f}")

        torch.save({
            'step':        step,
            'model_state': model.state_dict(),
            'optimizer':   optimizer.state_dict(),
            'train_loss':  loss.item(),
            'val_loss':    val_loss.item(),
            'char_to_int': char_to_int,
            'int_to_char': int_to_char,
            'vocab_size':  vocab_size,
        }, f'checkpoints/model_step_{step}.pt')

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            strikes       = 0
            torch.save({
                'step':        step,
                'model_state': model.state_dict(),
                'optimizer':   optimizer.state_dict(),
                'train_loss':  loss.item(),
                'val_loss':    val_loss.item(),
                'char_to_int': char_to_int,
                'int_to_char': int_to_char,
                'vocab_size':  vocab_size,
            }, 'model_best.pt')
            print(f"★ New best val loss: {best_val_loss:.4f} → model_best.pt")
        else:
            strikes += 1
            print(f"No improvement ({strikes}/{patience})")
            if strikes >= patience:
                print(f"\n★ Early stopping at step {step}")
                print(f"★ Best model → model_best.pt")
                break


# ─────────────────────────────────────────
# GENERATE SAMPLES
# ─────────────────────────────────────────

prompts = [
    "The nature of consciousness",
    "In the beginning",
    "The Giselians",
    "There was once",
    "Ra: I am Ra.",
]

for prompt in prompts:
    context = torch.tensor([encode(prompt)], dtype=torch.long).to(device)
    output  = model.generate(context, max_new_tokens=200, temperature=0.8)
    print(f"\n--- '{prompt}' ---")
    print(decode(output[0].tolist()))

torch.save({
    'step':        50000,
    'model_state': model.state_dict(),
    'optimizer':   optimizer.state_dict(),
    'char_to_int': char_to_int,
    'int_to_char': int_to_char,
    'vocab_size':  vocab_size,
}, 'model_final.pt')

print("\nFinal model saved → model_final.pt")