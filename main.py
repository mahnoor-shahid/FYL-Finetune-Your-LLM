"""
Fine-tuning pipeline: text classification with any Hugging Face model.

Everything is driven by config.json:
  config -> device -> data -> labels -> tokenization -> model -> training
  -> test evaluation -> inference demo -> (optional) push to Hub

Run from this folder:  python main.py
"""
import os

from transformers import set_seed

from utils.config import CONFIG, load_api_keys, print_configuration
from utils.device import get_device
from utils.visualize import plot_class_distribution, plot_token_lengths
from src.dataloader import get_dataset, find_label_column, get_label_names
from src.tokenization import (get_tokenizer, print_token_length_stats, tokenize_dataset,
                              show_tokenized_example, get_data_collator)
from src.model import get_model
from src.trainer import get_training_args, build_trainer, train, push_to_hub
from src.evaluate import evaluate_on_test
from src.inference import load_trained_model, predict, print_predictions

SEED = 42
OUTPUT = CONFIG.get("output", {})
SHOW_PLOTS = OUTPUT.get("show_plots", False)  # True opens chart windows (pauses until closed)


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ----------------------------------------------------------------------------
section("1. Setup")
set_seed(SEED)  # python, numpy and torch (incl. CUDA) in one call
device = get_device()

# Hugging Face libraries read HF_TOKEN; accept HUGGINGFACE_API_KEY from .env as well
keys = load_api_keys("HF_TOKEN", "HUGGINGFACE_API_KEY")
if keys:
    os.environ["HF_TOKEN"] = keys.get("HF_TOKEN") or keys["HUGGINGFACE_API_KEY"]

print_configuration()

# ----------------------------------------------------------------------------
section("2. Data")
dataset = get_dataset(seed=SEED)  # always returns train / validation / test
print(dataset)

LABEL_COLUMN = find_label_column(dataset)
LABEL_NAMES = get_label_names(dataset, LABEL_COLUMN)
NUM_LABELS = len(LABEL_NAMES)
id2label = dict(enumerate(LABEL_NAMES))
label2id = {name: i for i, name in id2label.items()}
print(f"\nLabel column: '{LABEL_COLUMN}'")
print(f"Labels ({NUM_LABELS}): {LABEL_NAMES}")

plot_class_distribution(dataset, LABEL_COLUMN, LABEL_NAMES, show=SHOW_PLOTS)

# ----------------------------------------------------------------------------
section("3. Tokenization")
tokenizer = get_tokenizer()
token_lengths = print_token_length_stats(dataset, tokenizer)
plot_token_lengths(token_lengths, tokenizer.model_max_length, show=SHOW_PLOTS)
tokenized_datasets = tokenize_dataset(dataset, tokenizer, LABEL_COLUMN, label2id)
show_tokenized_example(dataset, tokenized_datasets, tokenizer, id2label)
data_collator = get_data_collator(tokenizer)

# ----------------------------------------------------------------------------
section("4. Model")
model = get_model(NUM_LABELS, id2label, label2id, tokenizer, device=device)

# ----------------------------------------------------------------------------
section("5. Training")
training_args = get_training_args(seed=SEED)
trainer = build_trainer(model, tokenizer, tokenized_datasets, data_collator, training_args)
val_metrics = train(trainer)

# ----------------------------------------------------------------------------
section("6. Test evaluation")
test_metrics = evaluate_on_test(trainer, tokenized_datasets, LABEL_NAMES, show_plots=SHOW_PLOTS)

# ----------------------------------------------------------------------------
section("7. Inference (reloaded from disk)")
trained_model, trained_tokenizer = load_trained_model(device=device)
examples = list(dataset["test"][CONFIG["dataset"]["text_column"]][:5])
print_predictions(predict(examples, trained_model, trained_tokenizer))

# ----------------------------------------------------------------------------
if OUTPUT.get("push_to_hub"):
    section("8. Push to Hugging Face Hub")
    push_to_hub(trainer)

section("✅ Done")
print(f"Validation f1_macro: {val_metrics['eval_f1_macro']:.4f}")
print(f"Test       f1_macro: {test_metrics['test_f1_macro']:.4f}  accuracy: {test_metrics['test_accuracy']:.4f}")
