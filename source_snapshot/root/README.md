# Root Source Snapshot

This directory copies the important root-level files from `/Users/admin/AI`.

## Files

- `train.py`: main 38.3M-parameter English pretraining script.
- `train2.py`: earlier transformer training iteration with scheduler and early stopping.
- `train3.py`: teacher-guided Q&A fine-tuning with DeepSeek-R1 and LLaMA3 via Ollama.
- `chat.py`: interactive checkpoint loader and generation interface.
- `load.py`: minimal checkpoint loading helper.
- `test.py`: compact architectural smoke test for the GPT components.
- `clean.py`: text cleanup helper.

Large root checkpoint binaries are included in `../../models/` and documented in `../../docs/CHECKPOINT_METADATA.md`.
