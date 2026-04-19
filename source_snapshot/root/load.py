import torch
from test import GPT  # imports your model class

# Load everything
checkpoint = torch.load('model.pt')

model = GPT(
    vocab_size   = checkpoint['vocab_size'],
    embed_dim    = 256,
    num_heads    = 8,
    num_layers   = 8,
    max_seq_len  = 256
)
model.load_state_dict(checkpoint['model_state'])
model.eval()  # switch off training mode

char_to_int = checkpoint['char_to_int']
int_to_char = checkpoint['int_to_char']

encode = lambda s: [char_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_char[i] for i in l])

# Generate
context = torch.tensor([encode("ROMEO:")], dtype=torch.long)
output  = model.generate(context, max_new_tokens=200)
print(decode(output[0].tolist()))