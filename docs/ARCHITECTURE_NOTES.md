# Architecture Notes

## Root GPT Implementation

The root snapshot is the main source package for the GPT-style language model implementation.

### Model Shape

The main root training script instantiates:

```python
model = GPT(
    vocab_size   = vocab_size,
    embed_dim    = 512,
    num_heads    = 16,
    num_layers   = 12,
    max_seq_len  = 512
)
```

This appears in `train.py` and the archived copy at `source_snapshot/root/train.py`.

### Multi-Head Causal Self-Attention

`train.py` defines:

- Query, key, value projections with `nn.Linear(embed_dim, embed_dim)`
- Head splitting by reshaping to `(batch, num_heads, sequence, head_dim)`
- Scaled dot-product attention
- Causal masking with an upper-triangular mask
- Output projection back to the embedding dimension

`train3.py` repeats the architecture for teacher-guided Q&A fine-tuning.

### Positional Embeddings

The `GPT` class uses learned token and positional embeddings:

```python
self.token_embedding = nn.Embedding(vocab_size, embed_dim)
self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
```

These embeddings are added before the transformer block stack.

### Stacked Transformer Blocks

The `GPT` class stacks `num_layers` transformer blocks:

```python
self.blocks = nn.Sequential(
    *[TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
)
```

Each block uses:

- Pre-layer normalization
- Residual connection around attention
- Feed-forward MLP with GELU
- Residual connection around the feed-forward path

### Autoregressive Generation

The `generate()` method performs next-token sampling:

- Crops context to `max_seq_len`
- Runs the model forward
- Takes logits from the last position
- Applies temperature
- Samples the next token
- Appends the sampled token to the running sequence

The root `chat.py` file loads checkpoints and exposes this generation behavior interactively.

## Teacher-Guided Fine-Tuning

`train3.py` adds a teacher-guided Q&A training stage:

- Defaults to `deepseek-r1:latest` and `llama3:latest`
- Calls local Ollama's `/api/generate`
- Generates grounded Q&A pairs from source chunks
- Builds reward prompts for correctness, groundedness, and completeness
- Uses teacher feedback to guide answer loss
- Falls back to a lexical reward when the teacher call is unavailable
- Refreshes the Q&A pool dynamically during training
- Saves the best reward-guided model to `model_qa.pt`

## Token-Based Transformer Iterations

The later token-based transformer work is preserved under:

- `source_snapshot/token_transformer_project1`
- `source_snapshot/token_transformer_project3`

These include a local tokenizer, token-level datasets, transformer language models, staged training scripts, and chat scripts.
