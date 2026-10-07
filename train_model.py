import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from preprocess import clean_text


# -----------------------------
# 1. Load dataset
# -----------------------------

DATA_PATH = "data/news.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "data/news.csv not found. Please add your dataset."
    )

df = pd.read_csv(DATA_PATH)


# -----------------------------
# 2. Check required columns
# -----------------------------

if "text" not in df.columns or "label" not in df.columns:
    raise ValueError(
        "CSV must contain two columns: text and label"
    )


# Remove empty rows
df = df.dropna(subset=["text", "label"])

# Convert text to string
df["text"] = df["text"].astype(str)

# Clean text
df["clean_text"] = df["text"].apply(clean_text)


# -----------------------------
# 3. Convert labels
# -----------------------------

df["label"] = df["label"].astype(str).str.upper().str.strip()

label_mapping = {
    "FAKE": 0,
    "REAL": 1
}

df = df[df["label"].isin(label_mapping.keys())]

df["target"] = df["label"].map(label_mapping)


# -----------------------------
# 4. Split dataset
# -----------------------------

X = df["clean_text"]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# 5. Create ML pipeline
# -----------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2)
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000
        )
    )
])


# -----------------------------
# 6. Train model
# -----------------------------

print("Training model...")

model.fit(X_train, y_train)


# -----------------------------
# 7. Test model
# -----------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["FAKE", "REAL"]
    )
)


# -----------------------------
# 8. Save model
# -----------------------------

os.makedirs("model", exist_ok=True)

import joblib

joblib.dump(
    model,
    "model/fake_news_model.pkl"
)

print("\nModel saved successfully!")
print("Location: model/fake_news_model.pkl")