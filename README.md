# Custom Transformer LLM

This repository folder is a GitHub-ready package for the project:

**Custom Transformer LLM - Python, PyTorch**

The core implementation is a GPT-style decoder-only language model written from scratch in PyTorch. The root-level snapshot includes the main model/training files from `/Users/admin/AI`, including multi-head causal self-attention, learned positional embeddings, stacked transformer blocks, checkpointing, autoregressive generation, and teacher-guided Q&A fine-tuning.

## Main Files

These are the original folderless Python files from `/Users/admin/AI`, copied directly into this repo root because they are the primary project files:

- `train.py`: main English pretraining script for the 38.3M GPT-style model.
- `train2.py`: earlier training iteration with scheduler and early stopping.
- `train3.py`: teacher-guided Q&A fine-tuning with DeepSeek-R1 and LLaMA3 through Ollama.
- `chat.py`: interactive checkpoint loader and generation interface.
- `load.py`: minimal checkpoint loading helper.
- `test.py`: compact transformer component smoke test.
- `clean.py`: text cleanup helper.

## Quick Overview

| Project detail | Where to find it |
| --- | --- |
| Designed GPT-style model from scratch in PyTorch | `train.py`, `train3.py`, and `test.py` define the attention layer, feed-forward network, transformer block, GPT wrapper, and generation loop. |
| Implemented multi-head self-attention | See `MultiHeadAttention` in `train.py` and `train3.py`. |
| Used positional embeddings and stacked transformer blocks | See the `GPT` class in `train.py` and `train3.py`. |
| Trained on 2.69M+ tokens | `docs/DATASET_MANIFEST.md` records 2,727,753 root pretraining characters/tokens from `data/english_basic` plus `shakespeare.txt`. |
| Used checkpointing and validation | `train.py` saves periodic checkpoints under `models_english`; `docs/CHECKPOINT_METADATA.md` records metadata from saved `.pt` files. |
| Used autoregressive generation | `GPT.generate()` appears in the root training and chat scripts. |
| Used DeepSeek-R1 and LLaMA3 teacher models | `train3.py` defaults to `deepseek-r1:latest,llama3:latest`; `model_qa.pt` metadata also records both teacher model names. |
| Built reinforcement-style training | `train3.py` includes reward prompts, teacher grading, fallback lexical rewards, guided answer loss, and dynamic dataset refresh. |
| Transitioned from character-level to token-based modeling | `source_snapshot/token_transformer_project1` and `source_snapshot/token_transformer_project3` contain later tokenizer-based transformer iterations. |

## Folder Layout

```text
custom-transformer-llm/
  README.md
  .gitignore
  requirements.txt
  train.py
  train2.py
  train3.py
  chat.py
  load.py
  test.py
  clean.py
  data_samples/
    qa_dataset.txt
    quotes.txt
    shakespeare.txt
  models/
    model_final.pt
    model_english.pt
    model_qa.pt
  docs/
    ARCHITECTURE_NOTES.md
    CHECKPOINT_METADATA.md
    DATASET_MANIFEST.md
    REPRODUCIBILITY.md
    ROOT_FILE_MANIFEST.md
  source_snapshot/
    root/
      train.py
      train2.py
      train3.py
      chat.py
      load.py
      test.py
      clean.py
    token_transformer_project1/
      common.py
      train_tokenizer.py
      train_stage1.py
      train_stage2.py
      train_stage3.py
      train_stage4.py
      chat.py
    token_transformer_project3/
      common.py
      train_tokenizer.py
      train_stage1.py
      train_stage2.py
      chat.py
      README.txt
```

## Included Models

The trained root checkpoint files are included under `models/`:

- `models/model_final.pt`
- `models/model_english.pt`
- `models/model_qa.pt`

These are large binaries, so this folder includes `.gitattributes` for Git LFS. If GitHub rejects a normal push or web upload, publish the `.pt` files through Git LFS or as GitHub Release artifacts.

## Main Result

The largest root GPT checkpoints measure about **38.3M parameters**:

- `model_final.pt`: 38,303,744 parameters
- `model_english.pt`: 38,320,128 parameters
- `model_qa.pt`: 38,303,744 parameters

This is the project artifact behind the 38.2M parameter summary, with the exact count varying slightly by vocabulary size.
