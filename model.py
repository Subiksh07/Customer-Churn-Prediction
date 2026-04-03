import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ===============================
# 📥 Load dataset
# ===============================
df = pd.read_csv("data/churn.csv")

# ===============================
# 🧹 Drop unnecessary column
# ===============================
df = df.drop("CustomerID", axis=1)

# ===============================
# 🔄 Encoding
# ===============================
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df["Contract"] = df["Contract"].map({
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
})
df["PaymentMethod"] = df["PaymentMethod"].map({
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer": 2,
    "Credit card": 3
})
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ===============================
# ⚠️ Handle Missing Values (IMPORTANT FIX)
# ===============================
print("Missing values BEFORE fix:\n", df.isnull().sum())

df = df.fillna(0)

print("\nMissing values AFTER fix:\n", df.isnull().sum())

# ===============================
# 🎯 Features & Target
# ===============================
X = df.drop("Churn", axis=1)
y = df["Churn"]

# ===============================
# ⚖️ SMOTE (balance data)
# ===============================
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# ===============================
# ✂️ Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# ===============================
# 🌳 Random Forest
# ===============================
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

print("\n===== Random Forest =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ===============================
# 🚀 XGBoost
# ===============================
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

print("\n===== XGBoost =====")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))

# ===============================
# 📊 Feature Importance
# ===============================
importance = xgb_model.feature_importances_

plt.barh(X.columns, importance)
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Feature Importance (XGBoost)")
plt.show()