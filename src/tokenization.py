import numpy as np
from transformers import AutoTokenizer, DataCollatorWithPadding

from utils.config import CONFIG

MODEL_NAME = CONFIG["model"]["name"]
MAX_LENGTH = CONFIG["training"]["max_length"]
TEXT_COLUMN = CONFIG["dataset"]["text_column"]


def get_tokenizer(model_name=MODEL_NAME, max_length=MAX_LENGTH):
    """Load the tokenizer for the model in config.json."""
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Decoder models (GPT-2, Llama, Qwen, ...) have no pad token; reuse end-of-sequence
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("   No pad token found, using eos token for padding")

    # Never go beyond what the model supports (e.g. 512 for BERT)
    tokenizer.model_max_length = min(max_length, tokenizer.model_max_length)
    print(f"✅ Tokenizer loaded (vocab size: {tokenizer.vocab_size:,}, "
          f"max length: {tokenizer.model_max_length})")
    return tokenizer


def print_token_length_stats(dataset, tokenizer, text_column=TEXT_COLUMN, split="train"):
    """Print token-length statistics, to check that max_length fits the data. Returns the lengths."""
    texts = list(dataset[split][text_column])  # datasets 5.x returns a Column, the tokenizer wants a list
    lengths = np.array([len(ids) for ids in tokenizer(texts)["input_ids"]])
    truncated = (lengths > tokenizer.model_max_length).mean() * 100
    print(f"Token lengths ({split}): mean {lengths.mean():.1f}, "
          f"95th pct {np.percentile(lengths, 95):.0f}, max {lengths.max()} "
          f"-> {truncated:.1f}% truncated at max_length={tokenizer.model_max_length}")
    return lengths


def show_tokenized_example(dataset, tokenized, tokenizer, id2label,
                           text_column=TEXT_COLUMN, split="train", index=0):
    """Print one example as original text, tokens and label, to sanity-check tokenization."""
    example = tokenized[split][index]
    tokens = tokenizer.convert_ids_to_tokens(example["input_ids"])
    print(f"\nTokenized example ({split}[{index}]):")
    print(f"  Text:   {dataset[split][index][text_column]}")
    print(f"  Tokens: {' '.join(tokens)}")
    print(f"  IDs:    {example['input_ids']}")
    print(f"  Label:  {example['labels']} ({id2label[example['labels']]})")


def tokenize_dataset(dataset, tokenizer, label_column, label2id, text_column=TEXT_COLUMN):
    """
    Tokenize every split and prepare it for the Trainer.

    - truncates to the tokenizer's max length (padding happens per batch in the collator)
    - converts labels to integer ids ('labels' column, the name the Trainer expects);
      ClassLabel ids are kept, string/other labels are mapped through label2id
    - drops every other column (text, id, ...)
    """
    is_class_label = hasattr(dataset["train"].features[label_column], "names")

    def tokenize_batch(batch):
        encoded = tokenizer(batch[text_column], truncation=True)
        labels = batch[label_column]
        encoded["labels"] = labels if is_class_label else [label2id[str(v)] for v in labels]
        return encoded

    print("Tokenizing dataset...")
    tokenized = dataset.map(tokenize_batch, batched=True,
                            remove_columns=dataset["train"].column_names, desc="Tokenizing")
    print(f"✅ Tokenization complete! Columns: {tokenized['train'].column_names}")
    return tokenized


def get_data_collator(tokenizer):
    """Pads each batch to its longest example (faster than padding everything to max_length)."""
    return DataCollatorWithPadding(tokenizer=tokenizer)
