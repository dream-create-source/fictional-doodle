# Dataset Manifest

The original root GPT training loop is character-level, so one encoded character corresponds to one training token after vocabulary filtering. The measured root pretraining corpus combines `data/english_basic` and `shakespeare.txt`.

## Root Pretraining Corpus

| Source | Characters / char-level tokens |
| --- | ---: |
| `data/english_basic/a_little_princess.txt` | 391,888 |
| `data/english_basic/alice_in_wonderland.txt` | 74,651 |
| `data/english_basic/peter_pan_kensington_gardens.txt` | 149,181 |
| `data/english_basic/pride_and_prejudice.txt` | 763,083 |
| `data/english_basic/README.txt` | 776 |
| `data/english_basic/SOURCES.txt` | 537 |
| `data/english_basic/the_wonderful_wizard_of_oz.txt` | 232,243 |
| `shakespeare.txt` | 1,115,394 |
| **Total** | **2,727,753** |

This is the measured basis for describing the root pretraining corpus as 2.69M+ tokens.

## Root Fine-Tuning / Prompt Data

| Source | Characters |
| --- | ---: |
| `quotes.txt` | 535,869 |
| `qa_dataset.txt` | 30,859 |

The teacher-guided Q&A script reads `quotes.txt` by default, grows or rebuilds `qa_dataset.txt`, and saves the best fine-tuned model as `model_qa.pt`.

## Included Data Samples

This project folder includes lightweight root data copies under `data_samples/`:

- `qa_dataset.txt`
- `quotes.txt`
- `shakespeare.txt`

The public-domain book corpus under `data/english_basic` is documented here and in checkpoint metadata, but it is not copied into this folder to keep the upload compact.
