# bank_customer_dashboard.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(page_title="Bank Customer Intelligence Dashboard", layout="wide")

# --- Load Data ---
@st.cache_data
def load_data():
    return pd.read_csv("Customer_Churn_CLV_Segmentation.csv")

df = load_data()

# --- Simulated Model Metrics ---
model_metrics = {
    "Random Forest": {
        "Accuracy": 0.683,
        "Precision": 0.746,
        "Recall": 0.855,
        "F1 Score": 0.796,
        "ROC-AUC": 0.583
    },
    "XGBoost": {
        "Accuracy": 0.699,
        "Precision": 0.758,
        "Recall": 0.859,
        "F1 Score": 0.806,
        "ROC-AUC": 0.581
    }
}

# --- Title ---
st.title("🏦 Bank Customer Intelligence System")
st.markdown("Analyze **Customer Churn** and **Lifetime Value (CLV)**, and compare churn prediction models.")

# --- Tabs ---
tab1, tab2 = st.tabs(["📊 Segmentation Insights", "🤖 Churn Model Metrics"])

# ========== TAB 1 ==========
with tab1:
    st.header("📊 CLV + Churn Segmentation")

    # KPIs
    total_customers = df['CustomerID'].nunique()
    churn_rate = df['Churn'].mean() * 100
    high_value_churners = df.query("CustomerGroup == 'High Value - Churn'").shape[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("🧍 Total Customers", f"{total_customers:,}")
    col2.metric("📉 Churn Rate", f"{churn_rate:.2f}%")
    col3.metric("🔥 High Value Churners", f"{high_value_churners:,}")

    # Customer Group Chart
    st.markdown("### 📦 Customer Groups")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.countplot(data=df, x="CustomerGroup", order=df['CustomerGroup'].value_counts().index, palette="Set2", ax=ax1)
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    # CLV Distribution
    st.markdown("### 💰 Estimated CLV Distribution")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.histplot(df['EstimatedCLV'], bins=30, kde=True, color="teal", ax=ax2)
    st.pyplot(fig2)

    # Segment Filter
    st.markdown("### 🔍 Explore Customers by Segment")
    segment_filter = st.selectbox("Select Segment", df['CustomerGroup'].unique())
    filtered_df = df[df['CustomerGroup'] == segment_filter]
    st.dataframe(filtered_df[['CustomerID', 'EstimatedCLV', 'CLVSegment', 'Churn', 'CustomerGroup']].head(50))


# ========== TAB 2 ==========
with tab2:
    st.header("🤖 Churn Prediction Model Comparison")

    for model_name, metrics in model_metrics.items():
        st.subheader(f"🔍 {model_name}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
        col2.metric("Precision", f"{metrics['Precision']:.3f}")
        col3.metric("Recall", f"{metrics['Recall']:.3f}")

        col4, col5 = st.columns(2)
        col4.metric("F1 Score", f"{metrics['F1 Score']:.3f}")
        col5.metric("ROC-AUC", f"{metrics['ROC-AUC']:.3f}")

        st.markdown("---")

    st.markdown("✅ **Conclusion**: While both models perform well, **XGBoost** slightly outperforms Random Forest on F1 and accuracy, though ROC-AUC remains modest for both due to class imbalance.")

