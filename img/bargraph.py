import json
import matplotlib.pyplot as plt

# ======== SET YOUR CHECKPOINT PATH HERE ========
checkpoint_path = "./grammar_lora_model/checkpoint-7500/trainer_state.json"
# Example: "D:/AI Grammer Corrector/grammar_lora_model/checkpoint-7500/trainer_state.json"

# ======== LOAD TRAINING LOG DATA ========
with open(checkpoint_path, "r") as f:
    data = json.load(f)

log_history = data.get("log_history", [])

# ======== EXTRACT METRICS ========
train_loss, eval_loss, train_acc, eval_acc, steps = [], [], [], [], []

for entry in log_history:
    if "loss" in entry:
        train_loss.append(entry["loss"])
        steps.append(entry["step"])
    elif "eval_loss" in entry:
        eval_loss.append(entry["eval_loss"])
    if "accuracy" in entry:
        train_acc.append(entry["accuracy"])
    elif "eval_accuracy" in entry:
        eval_acc.append(entry["eval_accuracy"])

# ======== PLOT LOSS ========
plt.figure(figsize=(10, 6))
plt.plot(steps[:len(train_loss)], train_loss, 'r-', label='Training Loss')
if eval_loss:
    plt.plot(steps[:len(eval_loss)], eval_loss, 'b-', label='Validation Loss')

# ======== PLOT ACCURACY (if available) ========
if train_acc or eval_acc:
    ax2 = plt.twinx()
    if train_acc:
        ax2.plot(steps[:len(train_acc)], train_acc, 'g-', label='Training Accuracy')
    if eval_acc:
        ax2.plot(steps[:len(eval_acc)], eval_acc, 'g--', label='Validation Accuracy')
    ax2.set_ylabel("Accuracy", color='g')

# ======== FINAL PLOT SETTINGS ========
plt.title("Training and Validation Loss / Accuracy over Steps", fontsize=14)
plt.xlabel("Training Steps", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.grid(True)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("training_loss_accuracy.png", dpi=300)
plt.show()
