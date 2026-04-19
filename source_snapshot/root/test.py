import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        
        # embed_dim must be divisible by num_heads
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads  # each head works on a slice
        
        # Single matrices for all heads combined (more efficient)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        # Final projection after combining all heads
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        B, T, C = x.shape  # batch, sequence length, embed_dim
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Split into multiple heads
        # Reshape: (B, T, embed_dim) → (B, num_heads, T, head_dim)
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores for each head
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (self.head_dim ** 0.5)
        
        # Causal mask
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        
        weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attended = torch.matmul(weights, V)
        
        # Recombine all heads back together
        # (B, num_heads, T, head_dim) → (B, T, embed_dim)
        attended = attended.transpose(1, 2).contiguous().view(B, T, C)
        
        # Final projection
        output = self.out_proj(attended)
        
        return output

class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        
        # Convention: inner layer is 4x the embed_dim
        # This is true in GPT-2, LLaMA, basically everything
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),              # Smooth activation function
            nn.Linear(embed_dim * 4, embed_dim),
        )
    
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        
        self.attention   = MultiHeadAttention(embed_dim, num_heads)
        self.feedforward = FeedForward(embed_dim)
        
        # LayerNorm stabilizes training
        # Applied BEFORE each sub-layer (Pre-LN, modern standard)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # Residual connection around attention
        # x + attention(...) means original signal is preserved
        x = x + self.attention(self.norm1(x))
        
        # Residual connection around feedforward
        x = x + self.feedforward(self.norm2(x))
        
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len):
        super().__init__()
        
        # Convert token IDs to vectors
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Learn a vector for each position (0 to max_seq_len)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Stack of transformer blocks — this is the "depth" of the model
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )
        
        # Final LayerNorm before output
        self.norm = nn.LayerNorm(embed_dim)
        
        # Project to vocabulary size — produces a score for every possible next token
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
    
    def forward(self, token_ids, targets=None):
        B, T = token_ids.shape
        
        # Get token embeddings
        tok_emb = self.token_embedding(token_ids)   # (B, T, embed_dim)
        
        # Get position embeddings
        positions = torch.arange(T, device=token_ids.device)
        pos_emb = self.position_embedding(positions) # (T, embed_dim)
        
        # Add them together — now each token knows what it is AND where it is
        x = tok_emb + pos_emb
        
        # Run through all transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)   # (B, T, vocab_size)
        
        # If we have targets, calculate loss
        loss = None
        if targets is not None:
            # Reshape for cross entropy
            B, T, V = logits.shape
            logits_flat  = logits.view(B * T, V)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        
        return logits, loss
    
    def generate(self, token_ids, max_new_tokens):
        # Generate one token at a time
        for _ in range(max_new_tokens):
            # Get predictions
            logits, _ = self.forward(token_ids)
            
            # Focus on last token's predictions only
            logits = logits[:, -1, :]  # (B, vocab_size)
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample the next token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence and continue
            token_ids = torch.cat([token_ids, next_token], dim=1)
        
        return token_ids
# Small config for testing
model = GPT(
    vocab_size   = 1000,   # 1000 possible tokens
    embed_dim    = 64,     # each token = 64 numbers
    num_heads    = 8,      # 8 attention heads
    num_layers   = 4,      # 4 transformer blocks stacked
    max_seq_len  = 128     # max 128 tokens in a sequence
)

# Fake token IDs — batch of 2 sequences, 10 tokens each
token_ids = torch.randint(0, 1000, (2, 10))
targets   = torch.randint(0, 1000, (2, 10))

# Forward pass with loss
logits, loss = model(token_ids, targets)

print(f"Logits shape: {logits.shape}")       # (2, 10, 1000)
print(f"Loss: {loss.item():.4f}")            # ~6.9 (log of 1000, random baseline)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test generation
prompt = torch.randint(0, 1000, (1, 5))     # 1 sequence, 5 token prompt
output = model.generate(prompt, max_new_tokens=10)
print(f"Generated sequence shape: {output.shape}")  # (1, 15)