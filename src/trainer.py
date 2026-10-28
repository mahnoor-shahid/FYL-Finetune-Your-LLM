import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from utils.config import CONFIG, PROJECT_ROOT
from utils.device import get_precision_flags

TRAINING = CONFIG["training"]
OUTPUT = CONFIG.get("output", {})
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
MODEL_DIR = PROJECT_ROOT / "outputs" / "model"


def compute_metrics(eval_pred):
    """
    Accuracy plus macro/weighted F1. Macro F1 treats every class equally (honest on
    imbalanced data); it averages over the classes present in the evaluated split, so a
    class with no examples there (e.g. 'mixed' in poem_sentiment's test set) doesn't count as 0.
    """
    logits, labels = eval_pred
    if isinstance(logits, tuple):  # some models (e.g. T5) also return hidden states
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    present = np.unique(labels)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, labels=present, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, predictions, labels=present, average="weighted", zero_division=0),
    }


def compute_class_weights(train_dataset, num_labels):
    """
    Inverse-frequency class weights (n_samples / (n_classes * count)), so mistakes on
    rare classes cost more. A class with no training examples gets weight 0.
    """
    counts = np.bincount(np.array(train_dataset["labels"]), minlength=num_labels).astype(float)
    weights = np.where(counts > 0, counts.sum() / (num_labels * np.maximum(counts, 1)), 0.0)
    return torch.tensor(weights, dtype=torch.float32)


class WeightedLossTrainer(Trainer):
    """Trainer whose cross-entropy loss is weighted per class (for imbalanced datasets)."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = self.class_weights.to(device=logits.device, dtype=logits.dtype)
        loss = torch.nn.functional.cross_entropy(logits, labels, weight=weights)
        return (loss, outputs) if return_outputs else loss


def get_training_args(output_dir=OUTPUT_DIR, seed=42):
    """TrainingArguments from config.json, with mixed precision and optimizer chosen for the device."""
    precision = get_precision_flags()  # bf16 / fp16 on GPU, full precision on CPU
    return TrainingArguments(
        output_dir=str(output_dir),

        # Hyperparameters (config.json)
        learning_rate=TRAINING["learning_rate"],
        per_device_train_batch_size=TRAINING["batch_size"],
        per_device_eval_batch_size=TRAINING["batch_size"],
        num_train_epochs=TRAINING["num_epochs"],
        weight_decay=TRAINING["weight_decay"],
        warmup_steps=0.1,  # float < 1 = ratio of total steps (transformers 5 replaced warmup_ratio)

        # Evaluate and save every epoch, keep the best model by macro F1
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",

        # Device-dependent speed-ups
        **precision,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        dataloader_pin_memory=torch.cuda.is_available(),

        # Hugging Face Hub (only used when output.push_to_hub is true)
        hub_model_id=OUTPUT.get("hub_model_id") if OUTPUT.get("push_to_hub") else None,

        logging_steps=10,
        report_to="none",
        seed=seed,
    )


def build_trainer(model, tokenizer, tokenized_datasets, data_collator, training_args,
                  early_stopping_patience=TRAINING.get("early_stopping_patience", 3),
                  use_class_weights=TRAINING.get("class_weights", False)):
    """
    Create the Trainer. It moves the model and every batch to the GPU automatically,
    and saves the tokenizer together with the model (processing_class).
    """
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )
    if use_class_weights:
        weights = compute_class_weights(tokenized_datasets["train"], model.config.num_labels)
        trainer = WeightedLossTrainer(**trainer_kwargs, class_weights=weights)
        names = [model.config.id2label[i] for i in range(len(weights))]
        print("⚖️ Class weights: " + ", ".join(f"{n}={w:.2f}" for n, w in zip(names, weights.tolist())))
    else:
        trainer = Trainer(**trainer_kwargs)

    n_train = len(tokenized_datasets["train"])
    steps_per_epoch = -(-n_train // training_args.per_device_train_batch_size)  # ceil
    precision = "bf16" if training_args.bf16 else "fp16" if training_args.fp16 else "fp32"
    optim = getattr(training_args.optim, "value", training_args.optim)
    print(f"✅ Trainer ready: device {training_args.device}, {precision}, optimizer {optim}")
    print(f"   {n_train} train / {len(tokenized_datasets['validation'])} validation examples, "
          f"{steps_per_epoch} steps/epoch x {training_args.num_train_epochs:g} epochs "
          f"(early stopping patience {early_stopping_patience})")
    return trainer


def train(trainer, save_dir=MODEL_DIR):
    """Train, report the best validation scores, and save the best model + tokenizer."""
    print("\n🚀 Training...")
    trainer.train()

    metrics = trainer.evaluate()
    print("\n📊 Best model on validation:")
    for name in ("eval_accuracy", "eval_f1_macro", "eval_f1_weighted"):
        print(f"   {name.removeprefix('eval_'):<12} {metrics[name]:.4f}")

    trainer.save_model(str(save_dir))
    print(f"💾 Model and tokenizer saved to '{save_dir}'")
    return metrics


def push_to_hub(trainer):
    """Upload the best model + tokenizer to the Hub (needs HF_TOKEN with write access)."""
    print(f"\n☁️ Pushing to the Hub as '{trainer.args.hub_model_id}'...")
    trainer.push_to_hub(commit_message="Fine-tuned with finetune-pipeline")
    print(f"✅ Pushed: https://huggingface.co/{trainer.args.hub_model_id}")
