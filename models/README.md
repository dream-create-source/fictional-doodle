Link to the models: https://huggingface.co/Unknown898/Test1/tree/main/models

# Included Model Checkpoints

This directory contains the trained root model artifacts from `/Users/admin/AI`.

## Root Character-Level Models

| File | Size | Bytes | Parameters | Description |
| --- | ---: | ---: | ---: | --- |
| `model_final.pt` | 438.58 MiB | 459,880,063 | 38,303,744 | Root 38.3M GPT-style final checkpoint. |
| `model_english.pt` | 438.77 MiB | 460,079,539 | 38,320,128 | English-pretrained checkpoint with training source metadata. |
| `model_qa.pt` | 146.19 MiB | 153,288,064 | 38,303,744 | Teacher-guided Q&A checkpoint with DeepSeek-R1 and LLaMA3 metadata. |

## Tokenizer-Based Project 1 Models

These use the tokenizer at `project1/tokenizer.json`.

| File | Size | Bytes | Parameters | Stage |
| --- | ---: | ---: | ---: | --- |
| `project1/stage1_final.pt` | 121.45 MiB | 127,347,609 | 10,578,432 | `stage1_pretrain` |
| `project1/stage2_final.pt` | 121.45 MiB | 127,347,225 | 10,578,432 | `stage2_domain` |
| `project1/stage3_final.pt` | 121.45 MiB | 127,347,225 | 10,578,432 | `stage3_qna` |
| `project1/stage4_final.pt` | 121.45 MiB | 127,347,033 | 10,578,432 | `stage4_rlaif` |
| `project1/tokenizer.json` | 139.64 KiB | 142,995 | n/a | tokenizer |

## Tokenizer-Based Project 3 Models

These use the tokenizer at `project3/tokenizer.json`.

| File | Size | Bytes | Parameters | Stage |
| --- | ---: | ---: | ---: | --- |
| `project3/stage1_final.pt` | 319.43 MiB | 334,945,100 | 83,659,776 | `stage1_pretrain` |
| `project3/stage2_final.pt` | 319.43 MiB | 334,944,588 | 83,659,776 | `stage2_instruct` |
| `project3/tokenizer.json` | 138.86 KiB | 142,190 | n/a | tokenizer |

## GitHub Upload Note

These files are too large for normal GitHub blob upload. Use Git LFS before pushing:

```sh
git lfs install
git lfs track "*.pt"
git add .gitattributes models/
git commit -m "Add custom transformer model checkpoints"
```

If Git LFS is not available, publish the `.pt` files as GitHub Release artifacts and keep the metadata docs in this repository.
