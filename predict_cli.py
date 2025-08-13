
import os, json, argparse, pickle, numpy as np, pandas as pd, tensorflow as tf

BASE = r"C:\Users\sagni\Downloads\Dynamic Curriculum Designer"
PREPROC = os.path.join(BASE, "preprocessor.pkl")
LABEL   = os.path.join(BASE, "label_encoder.pkl")
MODEL_K = os.path.join(BASE, "model.keras")
MODEL_H = os.path.join(BASE, "model.h5")
THRESH  = os.path.join(BASE, "threshold.json")

def load_threshold(default=0.5):
    try:
        with open(THRESH, "r", encoding="utf-8") as f:
            return float(json.load(f).get("best_threshold", default))
    except Exception:
        return default

def load_artifacts():
    with open(PREPROC, "rb") as f:
        preproc = pickle.load(f)
    with open(LABEL, "rb") as f:
        lab = pickle.load(f)
    try:
        model = tf.keras.models.load_model(MODEL_K)
    except Exception:
        model = tf.keras.models.load_model(MODEL_H)
    return preproc, lab, model

def predict(payload: dict):
    preproc, lab, model = load_artifacts()
    cat_cols = list(preproc.transformers_[0][2])
    num_cols = list(preproc.transformers_[1][2])
    expected = cat_cols + num_cols
    row = {col: payload.get(col, np.nan) for col in expected}
    X_new = pd.DataFrame([row], columns=expected)
    X_new_proc = preproc.transform(X_new)
    prob = float(model.predict(X_new_proc).ravel()[0])
    thr = load_threshold()
    pred = int(prob >= thr)
    label = lab.inverse_transform([pred])[0]
    return {"probability_pass": prob, "threshold": thr, "predicted_class": pred, "label_decoded": str(label)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Student pass/fail predictor")
    ap.add_argument("--json", type=str, help="Path to JSON file with feature dict")
    ap.add_argument("--kv", action="append", default=[], help="Inline key=value (repeatable)")
    args = ap.parse_args()

    data = {}
    if args.json:
        with open(args.json, "r", encoding="utf-8") as f:
            data.update(json.load(f))
    for item in args.kv:
        if "=" in item:
            k, v = item.split("=", 1)
            # try cast numeric
            try:
                v2 = float(v)
                if v2.is_integer(): v2 = int(v2)
                data[k] = v2
            except Exception:
                data[k] = v
    out = predict(data)
    print(json.dumps(out, indent=2))
