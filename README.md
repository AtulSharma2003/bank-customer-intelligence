# 🏦 Bank Customer Intelligence Dashboard

An end-to-end Data Science project that predicts customer churn and segments customers based on Recency, Frequency, Monetary (RFM) values and Customer Lifetime Value (CLV). Includes interactive dashboard built using **Streamlit**.

### 🔗 Live Demo:
[atul-bank-dashboard.streamlit.app](https://atul-bank-dashboard.streamlit.app/)

---

## 🚀 Features

- 🔍 **Churn Prediction** using Random Forest & XGBoost
- 📊 **Customer Segmentation** using RFM & CLV modeling
- 🧠 ML performance comparison (Accuracy, F1 Score, ROC-AUC)
- 🌐 **Deployed dashboard** with interactive tabs and filters

---

## 📁 Project Structure
├── bank_customer_dashboard.py # Streamlit dashboard code
├── Customer_Churn_CLV_Segmentation.csv # Cleaned data for dashboard
├── project_bank_churn_analysis.ipynb # Jupyter analysis notebook
├── requirements.txt # Python dependencies
└── README.md # Project overview


---

## 💡 Technologies Used

- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Seaborn, Matplotlib
- Streamlit (for deployment)
- Jupyter Notebook (for development)

---

## 📊 Model Results

| Model         | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---------------|----------|-----------|--------|----------|----------|
| Random Forest | 0.683    | 0.746     | 0.855  | 0.796    | 0.583    |
| XGBoost       | 0.699    | 0.758     | 0.859  | 0.806    | 0.581    |

---

## 📦 How to Run Locally

```bash
git clone https://github.com/yourusername/bank-customer-intelligence
cd bank-customer-intelligence
pip install -r requirements.txt
streamlit run bank_customer_dashboard.py


