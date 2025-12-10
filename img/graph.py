import pandas as pd

df = pd.DataFrame({
    "review": [
        "Amazing movie, I loved it!",
        "Very boring and too long.",
        "Fantastic acting and story.",
        "Worst film, waste of time.",
        "Great experience, well directed."
    ],
    "sentiment": ["positive", "negative", "positive", "negative", "positive"]
})

df.head()


import matplotlib.pyplot as plt
from collections import Counter
import re

def clean(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()

df["clean_words"] = df["review"].apply(clean)

all_words = [w for row in df["clean_words"] for w in row]
freq = Counter(all_words).most_common(10)

words = [w for w, c in freq]
counts = [c for w, c in freq]

plt.figure(figsize=(10,5))
plt.bar(words, counts)
plt.title("Top 10 Words After Preprocessing")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.show()


from wordcloud import WordCloud

text = " ".join(all_words)

wc = WordCloud(width=800, height=400).generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wc)
plt.axis("off")
plt.title("Word Cloud of Cleaned Words")
plt.show()


sent_counts = df["sentiment"].value_counts()

plt.figure(figsize=(6,6))
plt.pie(sent_counts, labels=sent_counts.index, autopct="%1.1f%%")
plt.title("Sentiment Distribution")
plt.show()


import seaborn as sns
import numpy as np

# Dummy results
cm = np.array([[85, 15],
               [10, 90]])

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred Neg", "Pred Pos"],
            yticklabels=["Actual Neg", "Actual Pos"])
plt.title("Confusion Matrix")
plt.show()


epochs = [1, 2, 3, 4, 5]
accuracy = [0.72, 0.78, 0.81, 0.84, 0.87]
loss = [0.55, 0.48, 0.42, 0.39, 0.35]

plt.figure(figsize=(10,5))
plt.plot(epochs, accuracy, marker='o')
plt.title("Model Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

plt.figure(figsize=(10,5))
plt.plot(epochs, loss, marker='o')
plt.title("Model Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()


features = ["good", "bad", "great", "worst", "boring"]
importance = [0.45, 0.30, 0.60, 0.25, 0.40]

plt.figure(figsize=(10,5))
plt.bar(features, importance)
plt.title("TF-IDF Feature Importance (Dummy Values)")
plt.xlabel("Words")
plt.ylabel("Importance")
plt.show()