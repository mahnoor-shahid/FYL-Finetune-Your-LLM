from transformers import AutoModelForSequenceClassification

from utils.config import CONFIG

MODEL_NAME = CONFIG["model"]["name"]


def get_model(num_labels, id2label, label2id, tokenizer, device=None, model_name=MODEL_NAME):
    """
    Load a pretrained model with a fresh classification head on top.

    Model-agnostic: AutoModelForSequenceClassification picks the right class for
    encoders (BERT, RoBERTa, DeBERTa, ModernBERT), decoders (GPT-2, Llama, Qwen)
    and encoder-decoders (T5, BART) from the model's config.
    """
    print(f"Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        # allows starting from a checkpoint already fine-tuned with a different number of labels
        ignore_mismatched_sizes=True,
    )

    # Decoder models classify from the last non-padding token, so they must know the pad id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if device is not None:
        model.to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model loaded: {type(model).__name__} on {next(model.parameters()).device} "
          f"({total / 1e6:.1f}M parameters, {trainable / 1e6:.1f}M trainable)")
    return model
