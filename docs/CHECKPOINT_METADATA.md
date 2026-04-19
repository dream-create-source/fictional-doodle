# Checkpoint Metadata

This metadata was read from the original workspace checkpoints in `/Users/admin/AI` using PyTorch.

The three root model checkpoints are included in `models/`. They should be pushed with Git LFS or published as GitHub Release artifacts because normal GitHub blob upload rejects files this large.

## Root Checkpoints

| Original checkpoint | File size | Vocabulary | Parameters | Notes |
| --- | ---: | ---: | ---: | --- |
| `model_final.pt` | 459,880,063 bytes | 207 | 38,303,744 | Root GPT final checkpoint from the 512-dim, 16-head, 12-layer run. |
| `model_english.pt` | 460,079,539 bytes | 223 | 38,320,128 | English pretraining checkpoint; includes source file metadata. |
| `model_qa.pt` | 153,288,064 bytes | 207 | 38,303,744 | Teacher-guided Q&A checkpoint; stores teacher model metadata. |

## Supporting Checkpoint Metadata

| Original checkpoint | File size | Vocabulary | Parameters | Step | Stage | Loss metadata |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `models_4/model_final.pt` | 459,880,063 bytes | 207 | 38,303,744 | 5000 | n/a | n/a |
| `models_english/model_step_4500.pt` | 460,081,191 bytes | 223 | 38,320,128 | 4500 | `english_pretrain` | train 1.0837, val 1.7287 |
| `models_3/model_final.pt` | 78,078,719 bytes | 213 | 6,493,184 | 5000 | n/a | n/a |
| `models_2/model_final.pt` | 15,252,031 bytes | 213 | 1,260,800 | 5000 | n/a | n/a |
| `models_1/model_final.pt` | 14,794,239 bytes | 65 | 1,222,912 | 5000 | n/a | n/a |

## Teacher Metadata

`model_qa.pt` contains:

- `qa_dataset_file`: `qa_dataset.txt`
- `teacher_models`: `deepseek-r1:latest`, `llama3:latest`

This matches the teacher defaults in `source_snapshot/root/train3.py`.

## Source Metadata From `model_english.pt`

The English checkpoint records the following root training sources:

- `data/english_basic/README.txt`
- `data/english_basic/SOURCES.txt`
- `data/english_basic/a_little_princess.txt`
- `data/english_basic/alice_in_wonderland.txt`
- `data/english_basic/peter_pan_kensington_gardens.txt`
- `data/english_basic/pride_and_prejudice.txt`
- `data/english_basic/the_wonderful_wizard_of_oz.txt`
- `shakespeare.txt`

Its vocabulary source list also includes `qa_dataset.txt` and the quote corpus.
