import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from datasets import load_dataset

# Load dataset
dataset = load_dataset("agentlans/grammar-correction")

# True labels (corrected sentences)
true_labels = dataset["validation"]["output"]

# Example dummy predictions (replace with your model outputs)
# Must be same length as true_labels
pred_labels = true_labels[:100] + true_labels[100:]  # <-- Replace this line with your actual predictions

# For confusion matrix: you can’t directly plot raw sentences.
# You need to convert them to categorical labels, e.g.:
#   1. exact match (correct vs incorrect)
#   2. or cluster/group them if you have classes

# Example: comparing sentence equality (Correct vs Incorrect)
pred_class = ["Correct" if p == t else "Incorrect" for p, t in zip(pred_labels, true_labels)]
true_class = ["Correct"] * len(true_labels)

class_names = ["Correct", "Incorrect"]

cm = confusion_matrix(true_class, pred_class, labels=class_names)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

print(classification_report(true_class, pred_class, target_names=class_names))
