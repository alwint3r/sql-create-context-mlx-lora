# LoRA Fine-tuning for Text-to-SQL with Table Context Tasks using MLX

Requirements:
- Python 3.11
- mlx==0.3.0
- mlx-lm==0.0.12

## Preprocessing Data

Get the [sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) data from HuggingFace.
Then run the `preprocess-data.py` script to transform the data into a JSONL format and split the data into train, validation, and test sets.

```bash
python3 preprocess-data.py \
    --file /path/to/sql-create-context/sql_create_context_v4.json \
    --output-dir data \
    --validation-split-ratio 0.1 \
    --test-split-ratio 0.2
```

## Run LoRA Fine-tuning

```bash
python3 -m mlx_lm.lora --model TinyLlama/TinyLlama-1.1B-Chat-v0.1 \
    --train \
    --data data \
    --lora-layers 8 \
    --iters 1000
```
