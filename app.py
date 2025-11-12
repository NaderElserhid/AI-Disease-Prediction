# app.py
import streamlit as st
import joblib
import json
import numpy as np
from utils import normalize_symptom_str, fuzzy_map_input_to_vocab
import pandas as pd

MODEL_PATH = "model.pkl"
MLB_PATH = "mlb.pkl"
LE_PATH = "label_encoder.pkl"
VOCAB_PATH = "symptom_vocab.json"

try:
    cache_resource = st.cache_resource          # new API
except AttributeError:
    cache_resource = lambda **kw: st.cache(allow_output_mutation=True)


@cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    mlb = joblib.load(MLB_PATH)
    le = joblib.load(LE_PATH)
    with open(VOCAB_PATH, "r", encoding="utf8") as f:
        vocab = json.load(f)
    return model, mlb, le, vocab


def main():
    st.set_page_config(page_title="Disease Prediction", layout="centered")
    st.title("Belirtilere Göre Hastalık Tahmin Sistemi")
    st.write("Symptom → Disease prediction. Enter symptoms or tick the common ones.")

    try:
        model, mlb, le, vocab = load_artifacts()
    except Exception as e:
        st.error("Model artifacts not found. Please run train_model.py first.")
        st.stop()

    # Show top N symptom checkboxes (vocab might be large)
    top_symptoms = vocab[:120]  # show first 120 for UI simplicity
    st.subheader("Common Symptoms (tick boxes)")
    cols = st.columns(3)
    checked = []
    idx = 0
    for s in top_symptoms:
        c = cols[idx % 3]
        if c.checkbox(s):
            checked.append(s)
        idx += 1

    st.subheader("Or write symptoms in free text (comma separated)")
    free_text = st.text_input("Example: ateş, öksürük, baş ağrısı")

    if st.button("Tahmin Et"):
        # parse free text
        tokens = [normalize_symptom_str(t) for t in free_text.split(",")]
        tokens = [t for t in tokens if t]
        # fuzzy map tokens to vocab
        mapped = fuzzy_map_input_to_vocab(tokens, vocab, cutoff=0.6)
        # Combine with checked
        # preserve order unique
        final_symptoms = list(dict.fromkeys(checked + mapped))
        st.write("Girdi belirtiler (interpreted):", final_symptoms)

        # transform to mlb vector
        X_vec = mlb.transform([final_symptoms])
        # predict probabilities
        probs = model.predict_proba(X_vec)[0]  # shape (n_classes,)
        # get top 3 indices
        top_idx = np.argsort(probs)[::-1][:5]
        results = []
        for i in top_idx:
            results.append((le.inverse_transform([i])[0], float(probs[i])))
        # show nicely
        st.subheader("Tahminler:")
        for disease, p in results:
            st.write(f"- **{disease}** — {p*100:.1f}%")

        # small note
        st.info("Bu sistem yalnızca eğitildiği veriye göre tahmin yapar. Gerçek tıbbi teşhis yerine kullanılmamalıdır.")


if __name__ == "__main__":
    main()
