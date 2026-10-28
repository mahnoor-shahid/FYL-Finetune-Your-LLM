# FYL: Finetune Your LLM

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-Trainer-FFD21E)](https://huggingface.co/docs/transformers)
[![Datasets](https://img.shields.io/badge/🤗%20Datasets-Hub-FFD21E)](https://huggingface.co/docs/datasets)
[![scikit-learn](https://img.shields.io/badge/Metrics-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Classification](https://img.shields.io/badge/Classification-ready-success)](#features)
[![SFT · LoRA · DPO](https://img.shields.io/badge/SFT%20·%20LoRA%20·%20DPO-on%20the%20roadmap-lightgrey)](#roadmap)

**A config-driven fine-tuning pipeline for Hugging Face models: you choose the model, the data and the method, and FYL turns a general pretrained model into one built for your task.**

FYL's aim is to make every kind of fine-tuning (classification heads, SFT, LoRA/QLoRA, preference tuning) a matter of editing one JSON file instead of writing a new training script.

**Today**, FYL does single-label text classification with any Hugging Face model. Pick a model and a dataset in `config.json`, run `python main.py`, and you get a trained, evaluated, saved classifier. Optionally, it's pushed to the Hugging Face Hub. See the [Roadmap](#roadmap) for what's coming next.

```
config ─▶ device ─▶ data ─▶ labels ─▶ tokenization ─▶ model ─▶ training
       ─▶ test evaluation ─▶ inference demo ─▶ (optional) push to Hub
```

## Features

- **Works with any model.** `AutoModelForSequenceClassification` adds a fresh classification head to encoders (BERT, RoBERTa, DeBERTa, ModernBERT), decoders (GPT-2, Llama, Qwen) and encoder-decoders (T5, BART). Missing pad tokens on decoder models are handled for you.
- **Works with many datasets.** It finds the label column automatically: a `ClassLabel` feature, a common name such as `label`, `target` or `sentiment`, or the only low-cardinality column. ClassLabel, string and integer labels are all supported.
- **Always has train / validation / test splits.** Existing splits are kept. A split named `dev`, `valid` or `val` is used as validation. Missing splits, or a test split with hidden `-1` labels like GLUE's, are rebuilt from train with a stratified, seeded split.
- **Handles imbalanced classes.** It can weight the cross-entropy loss by inverse class frequency, and it selects the best model by **macro F1** instead of accuracy.
- **Uses the hardware it finds.** CUDA, Apple MPS or CPU. bf16 on GPUs that support it, fp16 on older GPUs, and the fused AdamW optimizer when CUDA is available.
- **Avoids wasted epochs.** Early stopping, evaluation and checkpointing every epoch, and the best checkpoint is reloaded at the end.
- **Makes the run easy to inspect.** Class distribution and token-length plots, a check of how much text `max_length` cuts off, a printed tokenized example, a per-class report and a confusion matrix.
- **Is reproducible.** One seed (42) controls Python, NumPy, PyTorch and the data splits.

## Quick start

```bash
git clone https://github.com/<your-username>/finetune-your-llm.git
cd finetune-your-llm

python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env    # optional: add HUGGINGFACE_API_KEY for gated models or Hub uploads
python main.py
```

> **Note on PyTorch:** `requirements.txt` pins the CUDA 13.0 build of PyTorch (`torch==2.14.1+cu130`, for NVIDIA driver 580 or newer). On a CPU-only machine, a Mac, or an older CUDA version, install the matching build from [pytorch.org](https://pytorch.org/get-started/locally/) first, then remove the `torch` line and the `--extra-index-url` line before you install the rest.

## Configuration

`config.json` controls everything:

```json
{
  "model":   { "name": "distilbert-base-uncased" },
  "dataset": {
    "name": "google-research-datasets/poem_sentiment",
    "subset": "default",
    "text_column": "verse_text",
    "label_column": null
  },
  "training": {
    "learning_rate": 5e-5,
    "batch_size": 32,
    "num_epochs": 10,
    "weight_decay": 0.01,
    "max_length": 128,
    "early_stopping_patience": 3,
    "class_weights": true
  },
  "output": {
    "show_plots": false,
    "push_to_hub": false,
    "hub_model_id": "your-username/distilbert-poem-sentiment"
  }
}
```

| Key | What it does |
|---|---|
| `model.name` | Any Hugging Face model id, e.g. `roberta-base`, `microsoft/deberta-v3-base`, `Qwen/Qwen2.5-0.5B` |
| `dataset.name` / `subset` | Hub dataset id and its config folder. The dataset is loaded from the Hub's parquet export (`refs/convert/parquet`). |
| `dataset.text_column` | The column that holds the input text |
| `dataset.label_column` | `null` detects the label column automatically. Set it to a column name if detection is ambiguous. |
| `training.max_length` | Longest input in tokens. It's capped at the model's own limit. Check the printed truncation rate to see whether it's long enough. |
| `training.class_weights` | `true` weights the loss by inverse class frequency |
| `training.early_stopping_patience` | Stops after this many epochs with no improvement in validation macro F1 |
| `output.show_plots` | `true` opens the chart windows. The run pauses until you close them. |
| `output.push_to_hub` / `hub_model_id` | Uploads the best model and tokenizer to the Hub. Needs a token with write access. |

## Environment variables

The pipeline itself reads only **one** key from `.env`:

| Variable | Needed for |
|---|---|
| `HUGGINGFACE_API_KEY` (or `HF_TOKEN`) | Gated or private models and datasets, and `push_to_hub` |

The other entries in `.env.example` belong to the wider course environment. You can leave them empty. `.env` is git-ignored, so never commit your real keys.

## Outputs

Each run writes to `outputs/`, which is git-ignored:

```
outputs/
├── model/                          # best model + tokenizer, ready for from_pretrained()
├── checkpoints/                    # up to 2 epoch checkpoints (the best one is always kept)
├── test_metrics.json               # accuracy, macro and weighted F1, loss
├── test_report.txt                 # per-class precision / recall / F1
├── class_distribution.png
├── class_distribution_by_split.png
├── token_lengths.png
└── confusion_matrix.png
```

## Using the trained model

```python
from src.inference import load_trained_model, predict, print_predictions

model, tokenizer = load_trained_model()   # defaults to outputs/model
print_predictions(predict(["the sun rose gently over the quiet hills"], model, tokenizer))
```

`predict` returns the label, the confidence, and the score for every class for each input text. `load_trained_model` also accepts a Hub id, so you can load a model you pushed earlier.

## Example result

With the default config (DistilBERT on `poem_sentiment`) on the held-out test set:

| Metric | Score |
|---|---|
| Accuracy | 0.837 |
| Macro F1 | 0.794 |
| Weighted F1 | 0.857 |

## Project structure

```
├── main.py              # runs the whole pipeline, step by step
├── config.json          # model, dataset, training and output settings
├── src/
│   ├── dataloader.py    # loading, label detection, split creation
│   ├── tokenization.py  # tokenizer, length stats, dataset tokenization, collator
│   ├── model.py         # model with a classification head
│   ├── trainer.py       # TrainingArguments, weighted-loss Trainer, metrics, Hub push
│   ├── evaluate.py      # test-set report, metrics JSON, confusion matrix
│   └── inference.py     # reload a saved model and classify new text
└── utils/
    ├── config.py        # config loading and .env key loading
    ├── device.py        # device and mixed-precision detection
    └── visualize.py     # plots
```

## Roadmap

Each step is chosen in `config.json` and uses the same data, evaluation and output steps as the others.

- [x] **Sequence classification**: full fine-tuning with a classification head, class weighting and early stopping
- [ ] **Parameter-efficient fine-tuning**: LoRA and QLoRA through `peft`, for any method below
- [ ] **Supervised fine-tuning (SFT)**: instruction and chat datasets with chat templates, through `trl`
- [ ] **Preference tuning**: DPO, ORPO and KTO on chosen/rejected pairs
- [ ] **More task types**: multi-label classification, token classification (NER), sequence-to-sequence
- [ ] **Local datasets**: CSV, JSON and JSONL files alongside Hub datasets
- [ ] **Experiment tracking**: optional Weights & Biases or MLflow logging

## Current limitations

- Only single-label sequence classification is implemented so far.
- Datasets must be on the Hugging Face Hub with a parquet export, which most public datasets have.
