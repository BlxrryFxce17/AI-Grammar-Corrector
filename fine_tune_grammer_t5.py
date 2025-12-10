# ------------------------------------------------------
# AI Grammar Correction - Full Training + Metrics Script
# ------------------------------------------------------

from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# -------------------------------
# 1. Device setup
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# 2. Load dataset
# -------------------------------
print("Loading dataset...")
dataset = load_dataset("agentlans/grammar-correction")
print("Columns:", dataset["train"].column_names)

def preprocess_data(example):
    incorrect = "fix: " + example["input"]
    corrected = example["output"]
    return {"src": incorrect, "tgt": corrected}

dataset = dataset.map(preprocess_data)

train_size = 30000
val_size = 4000
dataset["train"] = dataset["train"].select(range(train_size))
dataset["validation"] = dataset["validation"].select(range(val_size))

# -------------------------------
# 3. Tokenizer setup
# -------------------------------
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)

# -------------------------------
# 4. Model loading logic
# -------------------------------
model_path = "./grammar_model_lora"
last_checkpoint = None

if os.path.isdir("./grammar_lora_model"):
    checkpoints = [
        os.path.join("./grammar_lora_model", f)
        for f in os.listdir("./grammar_lora_model")
        if f.startswith("checkpoint-")
    ]
    if checkpoints:
        last_checkpoint = max(checkpoints, key=os.path.getmtime)

base_model = T5ForConditionalGeneration.from_pretrained(model_name)

if last_checkpoint and os.path.exists(os.path.join(last_checkpoint, "adapter_model.safetensors")):
    print(f"Resuming from {last_checkpoint}")
    model = PeftModel.from_pretrained(base_model, last_checkpoint)
elif os.path.exists(model_path):
    print(f"Loading LoRA model from {model_path}")
    model = PeftModel.from_pretrained(base_model, model_path)
else:
    print("Starting new fine-tune with LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q", "v"]
    )
    model = get_peft_model(base_model, lora_config)

model = model.to(device)

# -------------------------------
# 5. Tokenization
# -------------------------------
def tokenize(batch):
    model_inputs = tokenizer(batch["src"], max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(
        text_target=batch["tgt"], max_length=128, truncation=True, padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenizing dataset...")
tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)

# -------------------------------
# 6. Compute metrics
# -------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    mask = labels != -100
    acc = (preds[mask] == labels[mask]).mean()
    f1 = f1_score(labels[mask].flatten(), preds[mask].flatten(), average="weighted")
    return {"accuracy": acc, "f1": f1}

# -------------------------------
# 7. Training settings
# -------------------------------
args = TrainingArguments(
    output_dir="./grammar_lora_model",
    learning_rate=3e-4,
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=200,
    eval_strategy="epoch",  # both match now
    save_strategy="epoch",
    load_best_model_at_end=True,  # ensures best checkpoint loaded at end
    fp16=True,
    weight_decay=0.01,
    report_to=[],
    push_to_hub=False
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    compute_metrics=compute_metrics,
)

# -------------------------------
# 8. Train or resume
# -------------------------------
if last_checkpoint:
    print(f"Continuing training from {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    print("Starting new training session...")
    trainer.train()

# -------------------------------
# 9. Save model and tokenizer
# -------------------------------
os.makedirs("./results", exist_ok=True)
print("Saving model...")
model.save_pretrained("./grammar_model_lora")
tokenizer.save_pretrained("./grammar_model_lora")

# -------------------------------
# 10. Training Curves
# -------------------------------
print("Plotting training metrics...")

train_logs = trainer.state.log_history
steps = [i["step"] for i in train_logs if "loss" in i]
loss = [i["loss"] for i in train_logs if "loss" in i]

plt.figure(figsize=(8, 5))
plt.plot(steps, loss, label="Training Loss", color="blue")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid(True)
plt.savefig("./results/training_loss_curve.png")
plt.close()

# -------------------------------
# 11. Evaluation Curves
# -------------------------------
eval_logs = [i for i in train_logs if "eval_loss" in i]
eval_steps = [i["step"] for i in eval_logs]
eval_loss = [i["eval_loss"] for i in eval_logs]
eval_acc = [i.get("eval_accuracy", 0) for i in eval_logs]

plt.figure(figsize=(8, 5))
plt.plot(eval_steps, eval_loss, label="Validation Loss", color="orange")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Validation Loss Curve")
plt.legend()
plt.grid(True)
plt.savefig("./results/validation_loss_curve.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(eval_steps, eval_acc, label="Validation Accuracy", color="green")
plt.xlabel("Steps")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy Curve")
plt.legend()
plt.grid(True)
plt.savefig("./results/validation_accuracy_curve.png")
plt.close()

print("✅ All graphs saved in ./results/")
