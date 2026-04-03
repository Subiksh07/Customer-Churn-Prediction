import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import os
from sklearn.metrics import roc_curve, auc

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ===============================
# Load Model (cached)
# ===============================
@st.cache_resource
def load_model():
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/columns.pkl", "rb") as f:
        columns = pickle.load(f)
    with open("model/metrics.json", "r") as f:
        metrics = json.load(f)
    return model, columns, metrics

try:
    model, feature_columns, metrics = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.error("Model not found. Please run train_model.py first.")
    st.code("python train_model.py", language="bash")
    st.stop()

# ===============================
# Sidebar Inputs
# ===============================
st.sidebar.title("Customer Info")
st.sidebar.markdown("---")

age             = st.sidebar.slider("Age", 18, 100, 30)
gender_input    = st.sidebar.selectbox("Gender", ["Male", "Female"])
tenure          = st.sidebar.slider("Tenure (months)", 0, 100, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 0.0, 500.0, 50.0)
contract_input  = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment_input   = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer", "Credit card"
])
total_charges   = st.sidebar.number_input("Total Charges ($)", min_value=0.0, value=float(tenure * monthly_charges))

st.sidebar.markdown("---")
st.sidebar.caption("Fill in the customer details and click Predict.")

# ===============================
# Encode Inputs
# ===============================
gender   = 1 if gender_input == "Male" else 0
contract = {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract_input]
payment  = {"Electronic check": 0, "Mailed check": 1, "Bank transfer": 2, "Credit card": 3}[payment_input]

input_data = pd.DataFrame(
    [[age, gender, tenure, monthly_charges, contract, payment, total_charges]],
    columns=feature_columns
)

# ===============================
# Main Header
# ===============================
st.title("Customer Churn Prediction Dashboard")
st.caption("XGBoost model trained on Telco Customer Churn dataset with SMOTE balancing.")
st.markdown("---")

# ===============================
# Tabs
# ===============================
tab1, tab2, tab3, tab4 = st.tabs([
    "Predict", "Model Performance", "Analytics", "Bulk Predict"
])

# =========================================================
# TAB 1 — PREDICT
# =========================================================
with tab1:
    st.subheader("Single Customer Prediction")
    st.write("Adjust the customer details in the sidebar, then click Predict.")

    if st.button("Predict Churn", use_container_width=True, type="primary"):

        prediction  = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Result")
            if prediction == 1:
                st.error("High Risk of Churn")
                st.metric("Churn Probability", f"{probability * 100:.1f}%")
            else:
                st.success("Customer Likely to Stay")
                st.metric("Churn Probability", f"{probability * 100:.1f}%")

            # Risk badge
            if probability < 0.3:
                st.info("Low Risk — No immediate action needed.")
            elif probability < 0.7:
                st.warning("Medium Risk — Consider a retention offer.")
            else:
                st.error("High Risk — Immediate action recommended.")

            # Retention recommendation
            st.markdown("#### Retention Suggestions")
            if prediction == 1:
                if contract == 0:
                    st.write("- Offer a discount to upgrade to a 1 or 2-year contract")
                if payment == 0:
                    st.write("- Encourage switching from electronic check to auto-pay")
                if monthly_charges > 70:
                    st.write("- Offer a lower-cost bundle or loyalty discount")
                if tenure < 12:
                    st.write("- Send a welcome loyalty reward for staying past 12 months")
            else:
                st.write("- Customer is stable. Keep engagement with loyalty rewards.")

        with col2:
            st.markdown("#### Feature Importance")
            importance = model.feature_importances_
            fig, ax = plt.subplots(figsize=(5, 3))
            colors = ["#E24B4A" if i == importance.argmax() else "#378ADD"
                      for i in range(len(feature_columns))]
            ax.barh(feature_columns, importance, color=colors)
            ax.set_xlabel("Importance Score")
            ax.set_title("What drives this prediction")
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)

        # Save prediction
        new_row = input_data.copy()
        new_row["Prediction"]  = prediction
        new_row["Probability"] = round(probability * 100, 2)
        path = "predictions.csv"
        if os.path.exists(path):
            new_row.to_csv(path, mode="a", header=False, index=False)
        else:
            new_row.to_csv(path, index=False)
        st.caption("Prediction saved to predictions.csv")

# =========================================================
# TAB 2 — MODEL PERFORMANCE
# =========================================================
with tab2:
    st.subheader("Model Evaluation Metrics")
    st.caption("Evaluated on 20% held-out test data.")

    # Metric cards
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy",  f"{metrics['accuracy']}%")
    m2.metric("Precision", f"{metrics['precision']}%")
    m3.metric("Recall",    f"{metrics['recall']}%")
    m4.metric("F1 Score",  f"{metrics['f1']}%")
    m5.metric("ROC-AUC",   f"{metrics['roc_auc']}%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Confusion Matrix")
        cm = np.array(metrics["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(4, 3))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["No Churn", "Churn"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["No Churn", "Churn"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=14, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("#### ROC Curve")
        st.caption("Approximate ROC curve based on saved AUC score.")
        # Draw approximate ROC curve using saved AUC
        auc_val = metrics["roc_auc"] / 100
        fpr = np.linspace(0, 1, 100)
        tpr = np.clip(fpr + (auc_val - 0.5) * 2 * (1 - fpr) * fpr ** 0.3, 0, 1)
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.plot(fpr, tpr, color="#378ADD", lw=2,
                 label=f"AUC = {auc_val:.2f}")
        ax2.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title("ROC Curve")
        ax2.legend(loc="lower right")
        ax2.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2)

    st.markdown("---")
    st.markdown("#### Feature Importance (All Features)")
    importance = model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": importance
    }).sort_values("Importance", ascending=False)

    fig3, ax3 = plt.subplots(figsize=(8, 3))
    ax3.bar(feat_df["Feature"], feat_df["Importance"], color="#378ADD")
    ax3.set_ylabel("Importance Score")
    ax3.set_title("XGBoost Feature Importance")
    ax3.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig3)

# =========================================================
# TAB 3 — ANALYTICS DASHBOARD
# =========================================================
with tab3:
    st.subheader("Prediction Analytics Dashboard")

    if os.path.exists("predictions.csv"):
        data = pd.read_csv("predictions.csv")
        data["Label"] = data["Prediction"].map({0: "No Churn", 1: "Churn"})

        total     = len(data)
        churned   = data["Prediction"].sum()
        churn_pct = round((churned / total) * 100, 1) if total > 0 else 0
        avg_prob  = round(data["Probability"].mean(), 1) if total > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Predictions", total)
        c2.metric("Churned",           int(churned))
        c3.metric("Churn Rate",        f"{churn_pct}%")
        c4.metric("Avg Probability",   f"{avg_prob}%")

        st.markdown("---")
        colA, colB, colC = st.columns(3)

        with colA:
            st.markdown("#### Churn Distribution")
            counts = data["Label"].value_counts()
            fig4, ax4 = plt.subplots(figsize=(3.5, 3))
            ax4.bar(counts.index, counts.values,
                    color=["#1D9E75", "#E24B4A"])
            ax4.set_title("Churn vs No Churn")
            ax4.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig4)

        with colB:
            st.markdown("#### Churn Share")
            fig5, ax5 = plt.subplots(figsize=(3.5, 3))
            ax5.pie(counts.values, labels=counts.index,
                    autopct="%1.1f%%", startangle=90,
                    colors=["#1D9E75", "#E24B4A"])
            ax5.axis("equal")
            plt.tight_layout()
            st.pyplot(fig5)

        with colC:
            st.markdown("#### Probability Distribution")
            fig6, ax6 = plt.subplots(figsize=(3.5, 3))
            ax6.hist(data["Probability"], bins=10,
                     color="#378ADD", edgecolor="white")
            ax6.set_xlabel("Churn Probability (%)")
            ax6.set_ylabel("Count")
            ax6.set_title("Probability Spread")
            ax6.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig6)

        st.markdown("---")
        st.markdown("#### Recent Predictions")
        st.dataframe(
            data.tail(10).reset_index(drop=True),
            use_container_width=True
        )

        # Download predictions
        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download All Predictions as CSV",
            data=csv,
            file_name="all_predictions.csv",
            mime="text/csv"
        )

    else:
        st.info("No predictions yet. Go to the Predict tab and make some predictions first.")

# =========================================================
# TAB 4 — BULK PREDICT
# =========================================================
with tab4:
    st.subheader("Bulk Prediction via CSV Upload")
    st.write("Upload a CSV file with multiple customers to predict churn for all of them at once.")

    st.markdown("#### Required columns")
    st.code(", ".join(feature_columns))

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        try:
            bulk_df = pd.read_csv(uploaded)

            # Encode if raw string columns exist
            if "Gender" in bulk_df.columns and bulk_df["Gender"].dtype == object:
                bulk_df["Gender"] = bulk_df["Gender"].map({"Male": 1, "Female": 0})
            if "Contract" in bulk_df.columns and bulk_df["Contract"].dtype == object:
                bulk_df["Contract"] = bulk_df["Contract"].map({
                    "Month-to-month": 0, "One year": 1, "Two year": 2
                })
            if "PaymentMethod" in bulk_df.columns and bulk_df["PaymentMethod"].dtype == object:
                bulk_df["PaymentMethod"] = bulk_df["PaymentMethod"].map({
                    "Electronic check": 0, "Mailed check": 1,
                    "Bank transfer": 2, "Credit card": 3
                })

            bulk_df = bulk_df.fillna(0)
            X_bulk  = bulk_df[feature_columns]

            preds  = model.predict(X_bulk)
            probs  = model.predict_proba(X_bulk)[:, 1]

            bulk_df["Prediction"]        = preds
            bulk_df["Churn Probability"] = (probs * 100).round(1)
            bulk_df["Risk Level"]        = pd.cut(
                probs,
                bins=[0, 0.3, 0.7, 1.0],
                labels=["Low", "Medium", "High"]
            )

            total_b   = len(bulk_df)
            churned_b = int(preds.sum())
            st.success(f"Predicted {total_b} customers — {churned_b} likely to churn ({round(churned_b/total_b*100, 1)}%)")

            b1, b2, b3 = st.columns(3)
            b1.metric("Total Customers", total_b)
            b2.metric("Likely to Churn", churned_b)
            b3.metric("Churn Rate",      f"{round(churned_b/total_b*100,1)}%")

            st.markdown("#### Results")
            st.dataframe(bulk_df, use_container_width=True)

            csv_out = bulk_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Predictions as CSV",
                data=csv_out,
                file_name="bulk_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.info("Make sure your CSV has the correct column names listed above.")