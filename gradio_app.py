
import os, pickle, json, numpy as np, pandas as pd, tensorflow as tf, gradio as gr

BASE = r"C:\Users\sagni\Downloads\Dynamic Curriculum Designer"

with open(os.path.join(BASE, "preprocessor.pkl"), "rb") as f:
    preproc = pickle.load(f)
with open(os.path.join(BASE, "label_encoder.pkl"), "rb") as f:
    lab = pickle.load(f)

# columns and categories
cat_cols = list(preproc.transformers_[0][2])
num_cols = list(preproc.transformers_[1][2])
ohe = preproc.transformers_[0][1]
cat_choices = {c: list(ohe.categories_[i]) for i, c in enumerate(cat_cols)}

# load model
try:
    model = tf.keras.models.load_model(os.path.join(BASE, "model.keras"))
except Exception:
    model = tf.keras.models.load_model(os.path.join(BASE, "model.h5"))

# threshold
thr = 0.5
try:
    with open(os.path.join(BASE, "threshold.json"), "r", encoding="utf-8") as f:
        thr = float(json.load(f).get("best_threshold", 0.5))
except Exception:
    pass

def predict_ui(*vals):
    cat_vals = vals[:len(cat_cols)]
    num_vals = vals[len(cat_cols):]
    payload = {c: v for c, v in zip(cat_cols, cat_vals)}
    payload.update({c: v for c, v in zip(num_cols, num_vals)})
    # build row
    expected = cat_cols + num_cols
    import pandas as pd, numpy as np
    row = {col: payload.get(col, np.nan) for col in expected}
    X_new = pd.DataFrame([row], columns=expected)
    X_proc = preproc.transform(X_new)
    prob = float(model.predict(X_proc).ravel()[0])
    pred = int(prob >= thr)
    label = lab.inverse_transform([pred])[0]
    return f"Predicted: {label} | Probability(pass): {prob:.3f} | Threshold: {thr:.2f}"

inputs = []
for c in cat_cols:
    opts = cat_choices.get(c, [])
    inputs.append(gr.Dropdown(choices=opts, value=opts[0] if opts else None, label=c))
for c in num_cols:
    inputs.append(gr.Number(label=c))

demo = gr.Interface(
    fn=predict_ui,
    inputs=inputs,
    outputs=gr.Textbox(label="Output"),
    title="Student Pass/Fail Predictor",
    description="Select categorical values and enter numeric features. Uses saved preprocessor + model."
)
demo.launch()
