# train_model.py
"""
Train disease-prediction model.
Input: Final_Augmented_dataset_Diseases_and_Symptoms.csv (in same folder)
Output: model.pkl, mlb.pkl, symptom_vocab.json
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score
import joblib
from utils import extract_symptom_lists_from_df, normalize_symptom_str

DATA_PATH = "Final_Augmented_dataset_Diseases_and_Symptoms.csv"
MODEL_PATH = "model.pkl"
MLB_PATH = "mlb.pkl"
LE_PATH = "label_encoder.pkl"
VOCAB_PATH = "symptom_vocab.json"

def load_dataset(path):
    print("Loading dataset:", path)
    df = pd.read_csv(path)
    print("Shape:", df.shape)
    return df

def prepare_features(df):
    """
    Returns X (binary symptom matrix), y (disease labels), mlb, label_encoder, symptom_vocab
    This tries to detect common dataset formats:
     - a column named 'symptoms' with comma-separated text per row
     - or many boolean/int columns representing symptoms (non-disease columns)
    """
    symptom_lists = extract_symptom_lists_from_df(df)
    # clean each symptom
    symptom_lists = [[normalize_symptom_str(s) for s in row if str(s).strip() != ""] for row in symptom_lists]

    # Multi-label binarizer
    mlb = MultiLabelBinarizer(sparse_output=False)
    X = mlb.fit_transform(symptom_lists)  # shape (n_samples, n_symptoms)
    symptom_vocab = mlb.classes_.tolist()

    # Determine disease labels column
    disease_col = None
    for candidate in ["disease", "Disease", "DISEASE", "label", "Label"]:
        if candidate in df.columns:
            disease_col = candidate
            break
    if disease_col is None:
        # try to find a column with many unique small strings
        possible = [c for c in df.columns if df[c].dtype == object and df[c].nunique() < df.shape[0]]
        if possible:
            disease_col = possible[0]
        else:
            raise ValueError("Could not find disease column automatically. Please ensure a 'disease' column exists.")

    y_raw = df[disease_col].astype(str).str.strip().values
    # encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    return X, y, mlb, le, symptom_vocab, y_raw

def train_and_save(X, y, mlb, le):
    print("Train/test split")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    print("Shapes:", X_train.shape, X_test.shape)

    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy on test set: {acc:.4f}")
    print("Classification report (labels encoded):")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save artifacts
    joblib.dump(model, MODEL_PATH)
    joblib.dump(mlb, MLB_PATH)
    joblib.dump(le, LE_PATH)
    print("Saved model, mlb, and label encoder.")

def main():
    if not Path(DATA_PATH).exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Place the CSV in the same folder.")

    df = load_dataset(DATA_PATH)
    X, y, mlb, le, symptom_vocab, y_raw = prepare_features(df)
    # Save vocabulary file
    with open(VOCAB_PATH, "w", encoding="utf8") as f:
        json.dump(symptom_vocab, f, ensure_ascii=False, indent=2)
    print(f"Symptom vocab saved ({len(symptom_vocab)} symptoms).")
    train_and_save(X, y, mlb, le)
    print("Training finished. Artifacts:")
    print(" -", MODEL_PATH)
    print(" -", MLB_PATH)
    print(" -", LE_PATH)
    print(" -", VOCAB_PATH)

if __name__ == "__main__":
    main()

