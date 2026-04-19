# Root File Manifest

This manifest covers the files that lived directly in `/Users/admin/AI` without being inside a project folder.

## Copied Into This Project Folder

| Original root file | Project path | Purpose |
| --- | --- | --- |
| `train.py` | `train.py` and `source_snapshot/root/train.py` | Main English pretraining script for the 38.3M GPT-style model. |
| `train2.py` | `train2.py` and `source_snapshot/root/train2.py` | Earlier training iteration with scheduler and early stopping. |
| `train3.py` | `train3.py` and `source_snapshot/root/train3.py` | Teacher-guided Q&A fine-tuning with DeepSeek-R1 and LLaMA3. |
| `chat.py` | `chat.py` and `source_snapshot/root/chat.py` | Interactive checkpoint loading and generation. |
| `load.py` | `load.py` and `source_snapshot/root/load.py` | Minimal checkpoint loader helper. |
| `test.py` | `test.py` and `source_snapshot/root/test.py` | Compact transformer component smoke test. |
| `clean.py` | `clean.py` and `source_snapshot/root/clean.py` | Text cleaning helper. |
| `qa_dataset.txt` | `data_samples/qa_dataset.txt` | Q&A fine-tuning data snapshot. |
| `quotes.txt` | `data_samples/quotes.txt` | Default source file for teacher-generated Q&A. |
| `shakespeare.txt` | `data_samples/shakespeare.txt` | Root pretraining source included in the 2.73M token count. |

## Copied Into `models/`

| Original root file | Reason |
| --- | --- |
| `model_final.pt` | Included at `models/model_final.pt`; also documented in `CHECKPOINT_METADATA.md`. |
| `model_english.pt` | Included at `models/model_english.pt`; also documented in `CHECKPOINT_METADATA.md`. |
| `model_qa.pt` | Included at `models/model_qa.pt`; also documented in `CHECKPOINT_METADATA.md`. |

## Not Copied

| Original root file | Reason |
| --- | --- |
| `.DS_Store` | Local OS metadata, ignored. |

## Root File Sizes

| File | Size / count measured |
| --- | ---: |
| `shakespeare.txt` | 1,115,394 characters |
| `quotes.txt` | 535,869 characters |
| `qa_dataset.txt` | 30,859 characters, 563 lines |
| `train.py` | 10,912 characters, 344 lines |
| `train2.py` | 8,966 characters, 264 lines |
| `train3.py` | 28,371 characters, 835 lines |
| `chat.py` | 10,028 characters, 297 lines |
| `test.py` | 6,849 characters, 190 lines |
| `load.py` | 704 characters, 25 lines |
| `clean.py` | 2,012 characters, 68 lines |
