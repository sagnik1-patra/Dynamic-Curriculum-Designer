Dynamic Curriculum Designer (AI-Powered Adaptive Curriculum)

End‑to‑end ML project in Jupyter that loads student performance data, trains a neural classifier (Pass/Fail), and exports production artifacts (.keras, .h5, .pkl, .yaml), plus accuracy/ROC/PR plots, a confusion‑matrix heatmap, SHAP explainability, and a CLI/optional UI for user‑input predictions.

 Project Location (Windows)

All code and outputs are saved under:

C:\Users\sagni\Downloads\Dynamic Curriculum Designer

Key input files you provided:

C:\Users\sagni\Downloads\Dynamic Curriculum Designer\archive (1)\Maths.csv
C:\Users\sagni\Downloads\Dynamic Curriculum Designer\archive (1)\Portuguese.csv
C:\Users\sagni\Downloads\Dynamic Curriculum Designer\archive (2)\2012-2013-data-with-predictions-4-final.csv  (optional)

Note: Some .csv files may actually be Excel workbooks; the notebook includes a robust loader that auto-detects/reads such files.

 What This Project Does

Load & clean data from two Student Performance datasets (Maths & Portuguese)

Define target: Pass if final grade G3 ≥ 10, else Fail

Avoid leakage: Remove G1, G2, G3 as features

Preprocess: One‑hot encode categoricals, standardize numerics (scikit‑learn ColumnTransformer)

Train: Keras MLP (dense network) with early stopping

Evaluate: Accuracy, classification report, confusion matrix heatmap, ROC, PR

Export artifacts: preprocessor.pkl, label_encoder.pkl, model.keras, model.h5, model_config.yaml/json, plots

Predict: Helper function in notebook, CLI script, and optional Gradio UI

Explain: SHAP summary & bar plots

 Requirements

Install once (in your environment):

pip install pandas numpy scikit-learn tensorflow matplotlib seaborn openpyxl pyyaml shap
# Optional UI
pip install gradio

 How to Run (Jupyter)

Open the notebook cells you pasted earlier.

Run the Training + Saving Artifacts cell (robust loader → preprocess → train → save artifacts).

Run the Accuracy & Confusion Matrix cell to generate plots.

(Optional) Run the ROC/PR + Native .keras save + CLI/UI cells.

(Optional) Run the SHAP explainability cell.

Artifacts are written into the project folder.

 Outputs & Artifacts

File

Purpose

model.keras

Native Keras saved model (recommended).

model.h5

HDF5 legacy format (also saved for compatibility).

model_config.yaml

Model architecture config (YAML; JSON fallback as model_config.json).

preprocessor.pkl

Fitted scikit‑learn ColumnTransformer (OHE + scaler).

label_encoder.pkl

Fitted label encoder for the target.

accuracy_plot.png

Train vs validation accuracy per epoch.

confusion_matrix.png

Confusion‑matrix heatmap (test set).

roc_curve.png

ROC curve on test set (if cell executed).

pr_curve.png

Precision–Recall curve (if cell executed).

classification_report.txt

Precision/Recall/F1 per class.

metrics.json

Accuracy, ROC‑AUC, Average Precision summary.

threshold.json

Best decision threshold by F1 (optional cell).

predict_cli.py

CLI script for predictions from JSON or key=value pairs.

gradio_app.py

Optional Gradio UI for interactive predictions.

shap_summary.png

SHAP beeswarm plot of feature attributions.

shap_bar.png

SHAP mean

value

bar plot.

 Confusion Matrix (Result)

Below is the confusion matrix rendered from your saved result. Make sure confusion_matrix.png exists in the same folder as this README or adjust the path accordingly.



Interpretation:

True Negatives (top‑left): correctly predicted Fail

False Positives (top‑right): predicted Pass but actually Fail

False Negatives (bottom‑left): predicted Fail but actually Pass

True Positives (bottom‑right): correctly predicted Pass

 Quick Predict (Two Options)

1) Python helper (inside notebook)

example = {
    "school":"GP","sex":"F","address":"U","famsize":"GT3","Pstatus":"T",
    "schoolsup":"no","famsup":"yes","paid":"no","activities":"yes","nursery":"yes","higher":"yes","internet":"yes","romantic":"no",
    "age":17,"Medu":3,"Fedu":2,"traveltime":1,"studytime":2,"failures":0,
    "famrel":4,"freetime":3,"goout":3,"Dalc":1,"Walc":1,"health":5,"absences":4
}
print(predict_from_dict(example))

2) CLI (no notebook needed)

From a terminal:

cd "C:\Users\sagni\Downloads\Dynamic Curriculum Designer"
python predict_cli.py --json sample_input.json
# or
python predict_cli.py --kv school=GP --kv sex=F --kv address=U --kv age=17 --kv Medu=3 --kv Fedu=2 --kv traveltime=1 --kv studytime=2 --kv failures=0 --kv famrel=4 --kv freetime=3 --kv goout=3 --kv Dalc=1 --kv Walc=1 --kv health=5 --kv absences=4

The CLI automatically loads threshold.json (if present) to apply the best decision cut‑off instead of 0.5.

 Explainability (SHAP)

Run the SHAP cell to generate:

shap_summary.png — beeswarm view of per‑feature contributions per sample

shap_bar.png — global importance (mean |SHAP|) per feature

If you hit shape mismatches, the README’s SHAP cell includes robust handling for output shapes and feature‑name alignment.

 Notes & Troubleshooting

CSV vs Excel: Some provided .csv files are actually Excel workbooks. The loader checks for the ZIP signature (PK) and uses openpyxl automatically.

Encoding/Delimiter: The loader tries multiple encodings (utf‑8, cp1252, latin1) and delimiters (;, ,, \t, |).

Leakage: Make sure G1, G2, G3 are not fed as features if your goal is early/structural prediction.

Native Save: We save both model.keras and model.h5 (legacy). Prefer model.keras going forward.

Reproducible Split: The training split uses random_state=42. Reuse it to reproduce metrics.

Gradio not installed: Skip the UI cell or pip install gradio.

 Roadmap (Next Steps)

Add a knowledge‑tracing model (RNN/Transformer) using the optional 2012–2013 interactions file.

Implement curriculum policy learning (RL) to recommend next content block.

Track fairness metrics across subgroups (e.g., by prior support programs).

Package as a pip module or FastAPI microservice.
Author
Sagnik Patra
