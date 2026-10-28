import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils.config import PROJECT_ROOT

MODEL_DIR = PROJECT_ROOT / "outputs" / "model"


def load_trained_model(model_dir=MODEL_DIR, device="cpu"):
    """Load a fine-tuned model + tokenizer (a local folder or a Hub id) onto the device."""
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()
    print(f"✅ Loaded fine-tuned model from '{model_dir}' on {device}")
    return model, tokenizer


@torch.no_grad()
def predict(texts, model, tokenizer, device=None):
    """
    Classify one text or a list of texts.
    Returns a list of {"text", "label", "confidence", "scores"} dicts.
    """
    if isinstance(texts, str):
        texts = [texts]
    device = device or next(model.parameters()).device

    # Inputs must be on the same device as the model
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    logits = model(**inputs).logits
    probs = torch.softmax(logits.float(), dim=-1).cpu()

    id2label = model.config.id2label
    results = []
    for text, p in zip(texts, probs):
        best = int(p.argmax())
        results.append({
            "text": text,
            "label": id2label[best],
            "confidence": float(p[best]),
            "scores": {id2label[i]: float(s) for i, s in enumerate(p)},
        })
    return results


def print_predictions(results):
    for r in results:
        print(f"  {r['label']:<10} {r['confidence']:6.1%}  {r['text']}")
