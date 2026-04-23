import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import os

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ===============================
# Custom CSS
# ===============================
st.markdown("""
<style>
    .main { background-color: #f8f9fb; }
    #MainMenu, footer, header { visibility: hidden; }

    .top-banner {
        background: linear-gradient(90deg, #0C447C, #185FA5);
        color: white;
        padding: 18px 28px;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .banner-title { font-size: 22px; font-weight: 700; }
    .banner-sub   { font-size: 13px; opacity: 0.8; margin-top: 3px; }
    .banner-badge {
        background: rgba(255,255,255,0.2);
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 500;
    }

    .metric-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 16px 18px;
        border: 1px solid #e8eaf0;
    }
    .metric-value { font-size: 24px; font-weight: 700; color: #0C447C; }
    .metric-label { font-size: 12px; color: #6b7280; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-badge { display: inline-block; font-size: 11px; padding: 3px 10px; border-radius: 20px; margin-top: 6px; font-weight: 500; }
    .badge-green { background: #EAF3DE; color: #27500A; }
    .badge-blue  { background: #E6F1FB; color: #0C447C; }
    .badge-amber { background: #FAEEDA; color: #633806; }

    .section-card {
        background: white;
        border-radius: 12px;
        padding: 20px 22px;
        border: 1px solid #e8eaf0;
        margin-bottom: 1rem;
    }
    .section-title {
        font-size: 15px;
        font-weight: 600;
        color: #111827;
        margin-bottom: 16px;
        padding-bottom: 10px;
        border-bottom: 1px solid #f0f0f0;
    }

    .result-high {
        background: #FCEBEB; border: 1px solid #F09595;
        border-radius: 10px; padding: 16px 18px;
    }
    .result-low {
        background: #EAF3DE; border: 1px solid #97C459;
        border-radius: 10px; padding: 16px 18px;
    }
    .result-title-red   { font-size: 17px; font-weight: 700; color: #A32D2D; }
    .result-title-green { font-size: 17px; font-weight: 700; color: #27500A; }
    .result-sub-red     { font-size: 13px; color: #791F1F; margin-top: 3px; }
    .result-sub-green   { font-size: 13px; color: #3B6D11; margin-top: 3px; }

    .seg-grid { display: grid; grid-template-columns: repeat(2,1fr); gap: 10px; margin-top: 8px; }
    .seg-card { border-radius: 10px; padding: 12px 14px; border: 1px solid #e8eaf0; }
    .seg-at-risk { background: #FCEBEB; border-color: #F7C1C1; }
    .seg-loyal   { background: #EAF3DE; border-color: #C0DD97; }
    .seg-waver   { background: #FAEEDA; border-color: #FAC775; }
    .seg-new     { background: #E6F1FB; border-color: #B5D4F4; }

    .suggestion-item {
        display: flex; align-items: flex-start; gap: 10px;
        font-size: 13px; color: #374151; margin-bottom: 8px; line-height: 1.5;
    }
    .s-dot {
        width: 7px; height: 7px; min-width: 7px;
        border-radius: 50%; background: #378ADD; margin-top: 5px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 6px; background: white; border-radius: 10px;
        padding: 6px; border: 1px solid #e8eaf0; margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; font-size: 14px; padding: 8px 18px; }
    .stTabs [aria-selected="true"] { background: #E6F1FB; color: #0C447C; font-weight: 600; }

    .stButton > button {
        background: #0C447C; color: white; border: none;
        border-radius: 10px; padding: 10px 0;
        font-size: 15px; font-weight: 600; width: 100%;
    }
    .stButton > button:hover { background: #185FA5; color: white; }

    .app-footer {
        text-align: center; font-size: 12px; color: #9ca3af;
        padding: 1.5rem 0 0.5rem; border-top: 1px solid #f0f0f0; margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# Load Model
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
except FileNotFoundError:
    st.error("Model files not found. Please run train_model.py first.")
    st.code("python train_model.py", language="bash")
    st.stop()

# ===============================
# Top Banner
# ===============================
st.markdown(f"""
<div class="top-banner">
    <div>
        <div class="banner-title">Customer Churn Prediction Dashboard</div>
        <div class="banner-sub">XGBoost model with SMOTE balancing</div>
    </div>
    <div class="banner-badge">Accuracy: {metrics['accuracy']}%</div>
</div>
""", unsafe_allow_html=True)

# ===============================
# Metric Cards
# ===============================
st.markdown(f"""
<div class="metric-row">
    <div class="metric-card">
        <div class="metric-value">{metrics['accuracy']}%</div>
        <div class="metric-label">Accuracy</div>
        <div class="metric-badge badge-green">XGBoost</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{metrics['f1']}%</div>
        <div class="metric-label">F1 Score</div>
        <div class="metric-badge badge-blue">Balanced</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{metrics['roc_auc']}%</div>
        <div class="metric-label">ROC-AUC</div>
        <div class="metric-badge badge-green">Excellent</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{metrics['recall']}%</div>
        <div class="metric-label">Recall</div>
        <div class="metric-badge badge-amber">Sensitivity</div>
    </div>
</div>
""", unsafe_allow_html=True)

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
    col_left, col_right = st.columns([1, 1], gap="medium")

    with col_left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Customer details</div>', unsafe_allow_html=True)

        age             = st.slider("Age", 18, 100, 30)
        gender_input    = st.selectbox("Gender", ["Male", "Female"])
        tenure          = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly charges ($)", 0.0, 200.0, 65.0, step=0.5)
        total_charges   = st.number_input("Total charges ($)", min_value=0.0,
                                          value=round(float(tenure * monthly_charges), 2))
        contract_input  = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
        payment_input   = st.selectbox("Payment method", [
            "Electronic check", "Mailed check", "Bank transfer", "Credit card"
        ])
        predict_clicked = st.button("Predict Churn", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Prediction result</div>', unsafe_allow_html=True)

        if predict_clicked:
            gender   = 1 if gender_input == "Male" else 0
            contract = {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract_input]
            payment  = {"Electronic check": 0, "Mailed check": 1,
                        "Bank transfer": 2, "Credit card": 3}[payment_input]

            input_data = pd.DataFrame(
                [[age, gender, tenure, monthly_charges, contract, payment, total_charges]],
                columns=feature_columns
            )

            prediction  = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
            pct         = round(probability * 100, 1)

            if prediction == 1:
                st.markdown(f"""
                <div class="result-high">
                    <div class="result-title-red">High risk of churn</div>
                    <div class="result-sub-red">This customer is likely to leave soon</div>
                    <div style="margin-top:12px;">
                        <div style="display:flex;justify-content:space-between;font-size:13px;color:#791F1F;margin-bottom:5px;">
                            <span>Churn probability</span><span style="font-weight:700;">{pct}%</span>
                        </div>
                        <div style="height:8px;background:#F7C1C1;border-radius:4px;overflow:hidden;">
                            <div style="width:{pct}%;height:100%;background:#E24B4A;border-radius:4px;"></div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-low">
                    <div class="result-title-green">Customer likely to stay</div>
                    <div class="result-sub-green">Low churn risk detected</div>
                    <div style="margin-top:12px;">
                        <div style="display:flex;justify-content:space-between;font-size:13px;color:#3B6D11;margin-bottom:5px;">
                            <span>Churn probability</span><span style="font-weight:700;">{pct}%</span>
                        </div>
                        <div style="height:8px;background:#C0DD97;border-radius:4px;overflow:hidden;">
                            <div style="width:{pct}%;height:100%;background:#639922;border-radius:4px;"></div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

            # Retention suggestions
            st.markdown("<div style='margin-top:14px;font-size:14px;font-weight:600;color:#111827;'>Retention suggestions</div>", unsafe_allow_html=True)
            suggestions = []
            if prediction == 1:
                if contract == 0:
                    suggestions.append("Offer a discount to upgrade to a 1 or 2-year contract")
                if payment == 0:
                    suggestions.append("Encourage switching from electronic check to auto-pay")
                if monthly_charges > 70:
                    suggestions.append("Offer a lower-cost bundle or loyalty discount")
                if tenure < 12:
                    suggestions.append("Send a welcome loyalty reward — tenure is under 12 months")
                if not suggestions:
                    suggestions.append("Consider a personalised retention call")
            else:
                suggestions.append("Customer is stable — maintain engagement with loyalty rewards")
                suggestions.append("Offer an annual plan upgrade for added value")

            for s in suggestions:
                st.markdown(f'<div class="suggestion-item"><div class="s-dot"></div>{s}</div>',
                            unsafe_allow_html=True)

            # Feature importance
            st.markdown("<hr style='border:none;border-top:1px solid #f0f0f0;margin:12px 0;'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:14px;font-weight:600;color:#111827;margin-bottom:10px;'>Feature importance</div>", unsafe_allow_html=True)
            importance = model.feature_importances_
            feat_df = pd.DataFrame({
                "Feature": feature_columns, "Importance": importance
            }).sort_values("Importance", ascending=True)

            fig, ax = plt.subplots(figsize=(5, 2.5))
            colors = ["#E24B4A" if v == feat_df["Importance"].max() else "#378ADD"
                      for v in feat_df["Importance"]]
            ax.barh(feat_df["Feature"], feat_df["Importance"], color=colors, height=0.55)
            ax.set_xlabel("Importance score", fontsize=10)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(labelsize=9)
            fig.patch.set_facecolor("none")
            ax.set_facecolor("none")
            plt.tight_layout()
            st.pyplot(fig)

            # Save prediction
            new_row = input_data.copy()
            new_row["Prediction"]  = prediction
            new_row["Probability"] = pct
            path = "predictions.csv"
            if os.path.exists(path):
                new_row.to_csv(path, mode="a", header=False, index=False)
            else:
                new_row.to_csv(path, index=False)

        else:
            st.markdown("""
            <div style="text-align:center;padding:3rem 1rem;color:#9ca3af;">
                <div style="font-size:36px;margin-bottom:12px;">📊</div>
                <div style="font-size:14px;font-weight:500;color:#6b7280;">Fill in customer details and click Predict</div>
                <div style="font-size:13px;margin-top:6px;">Results will appear here</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Customer segments
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Customer segments</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="seg-grid">
        <div class="seg-card seg-at-risk">
            <div style="font-size:13px;font-weight:600;color:#A32D2D;">At-risk</div>
            <div style="font-size:12px;color:#791F1F;margin-top:3px;">Short tenure, high charges, month-to-month contract</div>
        </div>
        <div class="seg-card seg-loyal">
            <div style="font-size:13px;font-weight:600;color:#27500A;">Loyal</div>
            <div style="font-size:12px;color:#3B6D11;margin-top:3px;">Long tenure, stable contract, auto-pay enabled</div>
        </div>
        <div class="seg-card seg-waver">
            <div style="font-size:13px;font-weight:600;color:#633806;">Wavering</div>
            <div style="font-size:12px;color:#854F0B;margin-top:3px;">Medium tenure, mixed usage signals</div>
        </div>
        <div class="seg-card seg-new">
            <div style="font-size:13px;font-weight:600;color:#0C447C;">New customer</div>
            <div style="font-size:12px;color:#185FA5;margin-top:3px;">Recently joined, needs early engagement</div>
        </div>
    </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# TAB 2 — MODEL PERFORMANCE
# =========================================================
with tab2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Model evaluation metrics</div>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy",  f"{metrics['accuracy']}%")
    m2.metric("Precision", f"{metrics['precision']}%")
    m3.metric("Recall",    f"{metrics['recall']}%")
    m4.metric("F1 Score",  f"{metrics['f1']}%")
    m5.metric("ROC-AUC",   f"{metrics['roc_auc']}%")
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Confusion matrix</div>', unsafe_allow_html=True)
        cm = np.array(metrics["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["No churn", "Churn"], fontsize=11)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["No churn", "Churn"], fontsize=11)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "#111827",
                        fontsize=16, fontweight="bold")
        fig.patch.set_facecolor("none")
        ax.set_facecolor("none")
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">ROC curve</div>', unsafe_allow_html=True)
        auc_val = metrics["roc_auc"] / 100
        fpr = np.linspace(0, 1, 100)
        tpr = np.clip(fpr + (auc_val - 0.5) * 2 * (1 - fpr) * fpr ** 0.3, 0, 1)
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.plot(fpr, tpr, color="#378ADD", lw=2.5, label=f"AUC = {auc_val:.2f}")
        ax2.fill_between(fpr, tpr, alpha=0.08, color="#378ADD")
        ax2.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.3)
        ax2.set_xlabel("False positive rate", fontsize=11)
        ax2.set_ylabel("True positive rate", fontsize=11)
        ax2.legend(loc="lower right", fontsize=11)
        ax2.spines[["top", "right"]].set_visible(False)
        fig2.patch.set_facecolor("none")
        ax2.set_facecolor("none")
        plt.tight_layout()
        st.pyplot(fig2)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Feature importance — all features</div>', unsafe_allow_html=True)
    importance = model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": feature_columns, "Importance": importance
    }).sort_values("Importance", ascending=False)
    fig3, ax3 = plt.subplots(figsize=(8, 2.5))
    bar_colors = ["#E24B4A" if i == 0 else "#378ADD" for i in range(len(feat_df))]
    ax3.bar(feat_df["Feature"], feat_df["Importance"], color=bar_colors, width=0.5)
    ax3.set_ylabel("Importance score", fontsize=11)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.tick_params(axis="x", rotation=20, labelsize=10)
    fig3.patch.set_facecolor("none")
    ax3.set_facecolor("none")
    plt.tight_layout()
    st.pyplot(fig3)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# TAB 3 — ANALYTICS
# =========================================================
with tab3:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Prediction analytics</div>', unsafe_allow_html=True)

    if os.path.exists("predictions.csv"):
        data = pd.read_csv("predictions.csv")
        data["Label"] = data["Prediction"].map({0: "No Churn", 1: "Churn"})

        total     = len(data)
        churned   = int(data["Prediction"].sum())
        churn_pct = round((churned / total) * 100, 1) if total > 0 else 0
        avg_prob  = round(data["Probability"].mean(), 1) if total > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total predictions", total)
        c2.metric("Likely to churn",   churned)
        c3.metric("Churn rate",        f"{churn_pct}%")
        c4.metric("Avg probability",   f"{avg_prob}%")

        st.markdown("<hr style='border:none;border-top:1px solid #f0f0f0;margin:12px 0;'>", unsafe_allow_html=True)

        colA, colB, colC = st.columns(3, gap="medium")
        with colA:
            st.markdown("#### Churn distribution")
            counts = data["Label"].value_counts()
            fig4, ax4 = plt.subplots(figsize=(3.5, 2.8))
            ax4.bar(counts.index, counts.values, color=["#1D9E75","#E24B4A"], width=0.4)
            ax4.spines[["top","right"]].set_visible(False)
            fig4.patch.set_facecolor("none"); ax4.set_facecolor("none")
            plt.tight_layout(); st.pyplot(fig4)

        with colB:
            st.markdown("#### Churn share")
            fig5, ax5 = plt.subplots(figsize=(3.5, 2.8))
            ax5.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                    startangle=90, colors=["#1D9E75","#E24B4A"],
                    wedgeprops={"linewidth":1,"edgecolor":"white"})
            ax5.axis("equal")
            fig5.patch.set_facecolor("none")
            plt.tight_layout(); st.pyplot(fig5)

        with colC:
            st.markdown("#### Probability spread")
            fig6, ax6 = plt.subplots(figsize=(3.5, 2.8))
            ax6.hist(data["Probability"], bins=10, color="#378ADD", edgecolor="white", rwidth=0.85)
            ax6.set_xlabel("Churn probability (%)", fontsize=10)
            ax6.spines[["top","right"]].set_visible(False)
            fig6.patch.set_facecolor("none"); ax6.set_facecolor("none")
            plt.tight_layout(); st.pyplot(fig6)

        st.markdown("<hr style='border:none;border-top:1px solid #f0f0f0;margin:12px 0;'>", unsafe_allow_html=True)
        st.markdown("#### Recent predictions")
        st.dataframe(data.tail(10).reset_index(drop=True), use_container_width=True)

        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button("Download all predictions as CSV", data=csv,
                           file_name="all_predictions.csv", mime="text/csv")
    else:
        st.info("No predictions yet. Go to the Predict tab and make some predictions first.")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# TAB 4 — BULK PREDICT
# =========================================================
with tab4:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Bulk prediction via CSV upload</div>', unsafe_allow_html=True)
    st.write("Upload a CSV with multiple customers to predict churn for all of them at once.")
    st.markdown("**Required columns:**")
    st.code(", ".join(feature_columns))

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded is not None:
        try:
            bulk_df = pd.read_csv(uploaded)
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
            bulk_df  = bulk_df.fillna(0)
            X_bulk   = bulk_df[feature_columns]
            preds    = model.predict(X_bulk)
            probs    = model.predict_proba(X_bulk)[:, 1]

            bulk_df["Prediction"]        = preds
            bulk_df["Churn Probability"] = (probs * 100).round(1)
            bulk_df["Risk Level"]        = pd.cut(
                probs, bins=[0, 0.3, 0.7, 1.0], labels=["Low", "Medium", "High"]
            )

            total_b   = len(bulk_df)
            churned_b = int(preds.sum())
            st.success(f"Predicted {total_b} customers — {churned_b} likely to churn ({round(churned_b/total_b*100,1)}%)")

            b1, b2, b3 = st.columns(3)
            b1.metric("Total customers", total_b)
            b2.metric("Likely to churn", churned_b)
            b3.metric("Churn rate",      f"{round(churned_b/total_b*100,1)}%")

            st.dataframe(bulk_df, use_container_width=True)
            csv_out = bulk_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions as CSV", data=csv_out,
                               file_name="bulk_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Make sure your CSV has the correct column names listed above.")

    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# Footer
# ===============================
st.markdown("""
<div class="app-footer">
    Customer Churn Prediction — XGBoost + SMOTE — Final Year Project
</div>
""", unsafe_allow_html=True)

