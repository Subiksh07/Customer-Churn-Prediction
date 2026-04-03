import pandas as pd
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import pickle
import os
import json

# ===============================
# Load & Preprocess
# ===============================
df = pd.read_csv("data/churn.csv")
df = df.drop("CustomerID", axis=1)

df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df["Contract"] = df["Contract"].map({
    "Month-to-month": 0, "One year": 1, "Two year": 2
})
df["PaymentMethod"] = df["PaymentMethod"].map({
    "Electronic check": 0, "Mailed check": 1,
    "Bank transfer": 2,   "Credit card": 3
})
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df = df.fillna(0)

X = df.drop("Churn", axis=1)
y = df["Churn"]

# ===============================
# Train / Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# SMOTE (only on training data)
# ===============================
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# ===============================
# Train XGBoost Model
# ===============================
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    random_state=42
)
model.fit(X_train_res, y_train_res)

# ===============================
# Evaluate & Save Metrics
# ===============================
y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy":  round(accuracy_score(y_test, y_pred)  * 100, 2),
    "precision": round(precision_score(y_test, y_pred) * 100, 2),
    "recall":    round(recall_score(y_test, y_pred)    * 100, 2),
    "f1":        round(f1_score(y_test, y_pred)        * 100, 2),
    "roc_auc":   round(roc_auc_score(y_test, y_pred_prob) * 100, 2),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
}

os.makedirs("model", exist_ok=True)

with open("model/metrics.json", "w") as f:
    json.dump(metrics, f)

# ===============================
# Save Model & Columns
# ===============================
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/columns.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

print("=" * 40)
print("  Model trained and saved!")
print("=" * 40)
print(f"  Accuracy  : {metrics['accuracy']}%")
print(f"  Precision : {metrics['precision']}%")
print(f"  Recall    : {metrics['recall']}%")
print(f"  F1 Score  : {metrics['f1']}%")
print(f"  ROC-AUC   : {metrics['roc_auc']}%")
print("=" * 40)
print("  Files saved in /model folder")
print("=" * 40)