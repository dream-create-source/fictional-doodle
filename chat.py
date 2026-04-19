import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────
# DEVICE SETUP
# ─────────────────────────────────────────

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# ─────────────────────────────────────────
# MODEL COMPONENTS (must redefine to load)
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
# LOAD MODEL
# ─────────────────────────────────────────

import os
import glob

def list_checkpoints():
    checkpoints = glob.glob('model_step_*.pt') + glob.glob('model_final.pt') + glob.glob('model_qa.pt')
    checkpoints = sorted(checkpoints)
    return checkpoints

def load_model(path):
    checkpoint  = torch.load(path, map_location=device)
    char_to_int = checkpoint['char_to_int']
    int_to_char = checkpoint['int_to_char']
    vocab_size  = checkpoint['vocab_size']
    
    model = GPT(
        vocab_size   = vocab_size,
            embed_dim    = 512,
    num_heads    = 16,
    num_layers   = 12,
    max_seq_len  = 512
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state'])
    model.eval()  # turn off training mode
    
    encode = lambda s: [char_to_int[c] for c in s if c in char_to_int]
    decode = lambda l: ''.join([int_to_char[i] for i in l])
    
    step = checkpoint.get('step', '?')
    print(f"Loaded model from '{path}' (step {step})")

    qa_mode = ('qa_dataset_file' in checkpoint) or ('model_qa' in os.path.basename(path))

    return model, encode, decode, qa_mode


def generate_qa_answer(model, encode, decode, prompt, max_new_tokens, temperature):
    qa_prompt = f"Q: {prompt}\nA:"
    prompt_ids = encode(qa_prompt)

    if not prompt_ids:
        return None

    token_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            token_ids_crop = token_ids[:, -model.max_seq_len:]
            logits, _ = model(token_ids_crop)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_ids = torch.cat([token_ids, next_token], dim=1)

            generated_text = decode(token_ids[0].tolist()[len(prompt_ids):])
            if "\nQ:" in generated_text or "\n\n" in generated_text:
                break

    answer = decode(token_ids[0].tolist()[len(prompt_ids):])
    answer = answer.split("\nQ:")[0].split("\n\n")[0].strip()
    return answer or "(no answer generated)"


# ─────────────────────────────────────────
# INTERACTIVE CHAT
# ─────────────────────────────────────────

def chat(model, encode, decode, qa_mode=False):
    print("\n" + "═" * 50)
    if qa_mode:
        print("  Q&A AI — Interactive Mode")
    else:
        print("  SHAKESPEARE AI — Interactive Mode")
    print("═" * 50)
    print("Commands:")
    print("  /temp 0.8     → set temperature (default 0.8)")
    print("  /length 200   → set output length (default 200)")
    print("  /quit         → exit")
    print("═" * 50 + "\n")
    
    temperature = 0.8
    length      = 200
    
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not prompt:
            continue
        
        # Commands
        if prompt.startswith('/quit'):
            print("Goodbye!")
            break
        
        elif prompt.startswith('/temp'):
            try:
                temperature = float(prompt.split()[1])
                print(f"Temperature set to {temperature}")
                print("  → Lower (0.2) = focused, repetitive")
                print("  → Higher (1.5) = creative, chaotic")
            except:
                print("Usage: /temp 0.8")
            continue
        
        elif prompt.startswith('/length'):
            try:
                length = int(prompt.split()[1])
                print(f"Output length set to {length} characters")
            except:
                print("Usage: /length 200")
            continue
        
        elif prompt.startswith('/'):
            print("Unknown command. Try /temp, /length, or /quit")
            continue
        
        if qa_mode:
            answer = generate_qa_answer(model, encode, decode, prompt, length, temperature)
            if answer is None:
                print("(couldn't encode prompt — try different characters)\n")
                continue
            print(f"\nAI: {answer}")
            print("─" * 50 + "\n")
            continue

        # Generate response
        encoded = encode(prompt)

        if not encoded:
            print("(couldn't encode prompt — try different characters)\n")
            continue

        context = torch.tensor([encoded], dtype=torch.long).to(device)

        with torch.no_grad():  # no gradients needed for inference
            output = model.generate(context, max_new_tokens=length, temperature=temperature)

        result = decode(output[0].tolist())

        print(f"\nAI: {result}")
        print("─" * 50 + "\n")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    # Show available checkpoints
    checkpoints = list_checkpoints()
    
    if not checkpoints:
        print("No saved models found!")
        print("Run train.py first to create a model.")
        exit()
    
    print("\nAvailable models:")
    for i, cp in enumerate(checkpoints):
        size = os.path.getsize(cp) / 1024 / 1024
        print(f"  [{i}] {cp}  ({size:.1f} MB)")
    
    # Pick model
    if len(checkpoints) == 1:
        choice = 0
    else:
        try:
            choice = int(input(f"\nPick a model [0-{len(checkpoints)-1}]: "))
        except:
            choice = len(checkpoints) - 1  # default to latest
    
    model, encode, decode, qa_mode = load_model(checkpoints[choice])
    chat(model, encode, decode, qa_mode=qa_mode)
