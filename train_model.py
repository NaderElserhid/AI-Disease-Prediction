# train_model.py
"""
Train disease-prediction model.

Input:
  Final_Augmented_dataset_Diseases_and_Symptoms.csv (in the same folder)

Outputs:
  - model.pkl
  - mlb.pkl
  - label_encoder.pkl
  - symptom_vocab.json
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

from utils import extract_symptom_lists_from_df, normalize_symptom_str

DATA_PATH = "Final_Augmented_dataset_Diseases_and_Symptoms.csv"
MODEL_PATH = "model.pkl"
MLB_PATH = "mlb.pkl"
LE_PATH = "label_encoder.pkl"
VOCAB_PATH = "symptom_vocab.json"

# drop disease classes with < 2 rows (required for stratify)
MIN_CLASS_COUNT = 2


def load_dataset(path: str) -> pd.DataFrame:
    print(f"Loading dataset: {path}")
    df = pd.read_csv(path)
    print("Shape:", df.shape)
    return df


def detect_disease_column(df: pd.DataFrame) -> str:
    # Common names first
    for candidate in ["disease", "Disease", "DISEASE", "label", "Label"]:
        if candidate in df.columns:
            return candidate
    # Heuristic fallback: an object-typed column with duplicates
    for c in df.columns:
        if df[c].dtype == object and df[c].nunique() < len(df):
            return c
    raise ValueError(
        "Could not find a disease/label column. "
        "Add a column named 'disease' (or 'label')."
    )


def build_features(df: pd.DataFrame):
    """
    Returns:
      X (np.ndarray): multi-hot symptom matrix
      y_raw (np.ndarray[str]): original disease labels (unencoded)
      mlb (MultiLabelBinarizer)
      symptom_vocab (list[str])
      disease_col (str)
    """
    # Extract per-row symptom lists, then normalize each token
    symptom_lists = extract_symptom_lists_from_df(df)
    symptom_lists = [
        [normalize_symptom_str(s) for s in row if str(s).strip() != ""]
        for row in symptom_lists
    ]

    # MultiLabelBinarizer -> X
    mlb = MultiLabelBinarizer(sparse_output=False)
    X = mlb.fit_transform(symptom_lists)
    symptom_vocab = mlb.classes_.tolist()

    disease_col = detect_disease_column(df)
    y_raw = df[disease_col].astype(str).str.strip().values

    return X, y_raw, mlb, symptom_vocab, disease_col


def filter_rare_classes(X: np.ndarray, y_raw: np.ndarray, min_count: int = MIN_CLASS_COUNT):
    """
    Drop rows whose disease label frequency < min_count.
    """
    counts = pd.Series(y_raw).value_counts()
    keep = counts[counts >= min_count].index
    drop = counts[counts < min_count].index

    if len(drop) > 0:
        print(
            f"Dropping {len(drop)} disease class(es) with < {min_count} samples. "
            f"Examples: {list(drop)[:10]}{' ...' if len(drop) > 10 else ''}"
        )

    mask = pd.Series(y_raw).isin(keep).to_numpy()
    X_f = X[mask]
    y_f = y_raw[mask]
    return X_f, y_f, keep.tolist()


def train_and_save(X: np.ndarray, y_raw: np.ndarray, mlb: MultiLabelBinarizer):
    """
    Fits LabelEncoder on filtered y, trains RandomForest, evaluates, and saves artifacts.
    """
    # Encode labels AFTER filtering so the encoder only knows kept classes
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)

    # Sanity: every class should have >= 2 instances now
    vc = pd.Series(y_enc).value_counts()
    min_cls = int(vc.min())
    if min_cls < 2:
        raise RuntimeError(
            f"After filtering, some class still has <2 samples (min={min_cls}). "
            "Check your data or increase MIN_CLASS_COUNT filtering."
        )

    print("Train/test split (stratified)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.15, random_state=42, stratify=y_enc
    )
    print("Shapes:", X_train.shape, X_test.shape)

    print("Training RandomForestClassifier...")

    model = RandomForestClassifier(
        n_estimators=100,       # moderate number of trees
        max_depth=20,           # prevents huge memory use
        min_samples_leaf=2,     # avoids overly deep trees
        max_features="sqrt",    # speeds up training
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy on test set: {acc:.4f}")
    print("Classification report (encoded labels):")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Saving artifacts...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(mlb, MLB_PATH)
    joblib.dump(le, LE_PATH)
    print(f"Saved: {MODEL_PATH}, {MLB_PATH}, {LE_PATH}")


def main():
    if not Path(DATA_PATH).exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Place the CSV in the same folder."
        )

    df = load_dataset(DATA_PATH)
    X, y_raw, mlb, symptom_vocab, disease_col = build_features(df)

    # Save symptom vocabulary for the app UI
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(symptom_vocab, f, ensure_ascii=False, indent=2)
    print(
        f"Symptom vocab saved ({len(symptom_vocab)} symptoms) -> {VOCAB_PATH}")

    # Filter rare disease classes so stratify won't fail
    X_f, y_f, kept = filter_rare_classes(X, y_raw, MIN_CLASS_COUNT)
    print(
        f"After filtering: X={X_f.shape}, y={len(y_f)}. "
        f"Kept disease classes: {len(kept)}"
    )

    # Train, evaluate, and save
    train_and_save(X_f, y_f, mlb)

    print("Training finished. Artifacts:")
    print(" -", MODEL_PATH)
    print(" -", MLB_PATH)
    print(" -", LE_PATH)
    print(" -", VOCAB_PATH)


if __name__ == "__main__":
    main()
