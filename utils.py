# utils.py
"""
Utility functions for dataset parsing and symptom normalization.
"""
import re
import json
from typing import List
import pandas as pd
from difflib import get_close_matches
import unicodedata

def normalize_symptom_str(s: str) -> str:
    """Lowercase, remove extra spaces, accents, punctuation at ends."""
    if s is None:
        return ""
    s = str(s).strip()
    # remove surrounding punctuation
    s = re.sub(r'^[\W_]+|[\W_]+$', '', s)
    # remove multiple spaces
    s = re.sub(r'\s+', ' ', s)
    # normalize unicode
    s = unicodedata.normalize("NFKD", s)
    return s.lower()

def extract_symptom_lists_from_df(df: pd.DataFrame):
    """
    Try various heuristics to extract per-row symptom lists.
    Returns list of lists: [[symptom1, symptom2, ...], ...]
    """
    # 1) If a 'symptoms' column exists (or 'symptom'), parse comma/semicolon separated
    for col in df.columns:
        if col.lower() in ("symptoms", "symptom", "symptom_list"):
            lists = []
            for v in df[col].fillna("").astype(str):
                # split on commas, semicolons, '|'
                parts = re.split(r'[;,|/]+', v)
                parts = [p.strip() for p in parts if p.strip() != ""]
                lists.append(parts)
            return lists

    # 2) If there's a single text column (other than disease) with many commas, try it
    text_cols = [c for c in df.columns if df[c].dtype == object]
    candidate = None
    for c in text_cols:
        # skip likely disease-like columns (few unique)
        if df[c].nunique() > 5 and df[c].astype(str).str.contains(',').sum() > 0:
            candidate = c
            break
    if candidate:
        lists = []
        for v in df[candidate].fillna("").astype(str):
            parts = re.split(r'[;,|/]+', v)
            parts = [p.strip() for p in parts if p.strip() != ""]
            lists.append(parts)
        return lists

    # 3) Else assume multiple one-hot columns with boolean/integer values representing symptoms.
    # We'll take all non-disease columns and treat each as a symptom column.
    # Skip columns with many unique values (like patient id etc.)
    non_numeric_cols = []
    non_symptom_cols = []
    for c in df.columns:
        # Heuristic: disease-like columns often have many duplicates/low unique relative to rows
        pass

    # Try numeric/binary columns
    symptom_cols = []
    for c in df.columns:
        # skip obviously disease-like names
        if c.lower() in ("disease", "label", "diagnosis", "diag"):
            continue
        # choose columns with only values {0,1} or small integer range
        unique_vals = set(df[c].dropna().unique())
        if unique_vals <= {0, 1} or (len(unique_vals) <= 5 and df[c].dtype != object):
            symptom_cols.append(c)
    if symptom_cols:
        lists = []
        for _, row in df[symptom_cols].fillna(0).iterrows():
            row_symptoms = [col for col, val in row.items() if int(val) == 1]
            lists.append(row_symptoms)
        return lists

    # Fallback: for each row, return empty list
    return [[] for _ in range(len(df))]

def fuzzy_map_input_to_vocab(input_tokens: List[str], vocab: List[str], cutoff=0.7):
    """
    Map a list of input symptom tokens (strings) to the closest terms in vocab using difflib.
    Returns list of mapped vocab terms (unique).
    """
    mapped = []
    for token in input_tokens:
        token_norm = normalize_symptom_str(token)
        if token_norm == "":
            continue
        # exact match
        if token_norm in vocab:
            mapped.append(token_norm)
            continue
        # fuzzy match
        matches = get_close_matches(token_norm, vocab, n=3, cutoff=cutoff)
        if matches:
            # choose best match
            mapped.append(matches[0])
        else:
            # as last resort, include the token as-is (so model's mlb won't match; downstream will ignore unknown)
            mapped.append(token_norm)
    # remove duplicates preserving order
    seen = set()
    result = []
    for m in mapped:
        if m not in seen:
            seen.add(m)
            result.append(m)
    return result

