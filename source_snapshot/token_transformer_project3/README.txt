Project 3
=========

Goal
----
Train a tokenizer-based Transformer that talks properly before we worry about narrower domain behavior.

Model
-----
- Architecture: decoder-only Transformer
- Approx size: about 80M to 100M parameters
- Current config:
  - embed_dim = 768
  - num_heads = 12
  - num_layers = 10
  - max_seq_len = 256

Data
----
This project reuses the already-downloaded project2 data under:
- /Users/admin/AI/project2/data/wiki
- /Users/admin/AI/project2/data/books
- /Users/admin/AI/project2/data/science
- /Users/admin/AI/project2/data/instruct
- /Users/admin/AI/project2/data/synthetic

Default stages
--------------
1. train_tokenizer.py
   Builds a local tokenizer from a mixed English/general corpus.

2. train_stage1.py
   Broad next-token pretraining on Wikipedia/books/science text.

3. train_stage2.py
   Instruction tuning on Dolly plus Ollama synthetic pairs so the model behaves more like an assistant.

4. chat.py
   Interactive chat over the best checkpoint.

Why this is simpler than project2
---------------------------------
- Transformer instead of SSM
- safer defaults for MPS
- no domain-only guardrails
- focused on talking properly first

Recommended run order
---------------------
python3 /Users/admin/AI/project3/train_tokenizer.py
python3 /Users/admin/AI/project3/train_stage1.py
python3 /Users/admin/AI/project3/train_stage2.py
python3 /Users/admin/AI/project3/chat.py

Memory notes
------------
This model is much larger than project1. If stage1 runs out of memory, try:

MAX_CHARS_PER_FILE=12000000 TOTAL_CHAR_CAP=40000000 BATCH_SIZE=1 GRAD_ACCUM_STEPS=12 python3 /Users/admin/AI/project3/train_stage1.py

The final checkpoints are saved without optimizer state to keep them much smaller on disk.
