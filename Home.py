"""
Fraud Detection & Risk Analytics Dashboard
Author: Professional Data Science / Engineering Team
Description: An interactive Streamlit application for analyzing banking transactions,
monitoring risk metrics, and exploring fraud patterns.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# Page Configuration
st.set_page_config(
    page_title="Fraud Detection & Risk Analytics",
    page_icon="🛡️",
    layout="wide",
)

# Load Data
@st.cache_data
def load_data():
  return pd.read_parquet("Cleaned_Data_Fraud_Trans.parquet")


df = load_data()
st.image("Unlocking-the-Power-of-Fraud-Analytics-in-Banking_11536x699.jpg", caption="Banking Fraud Analytics", use_container_width=True)
# Main Dashboard Content (Home Page with Data Overview - No Filters)
st.title("🏦 Banking Fraud Analytics Dashboard")
st.markdown(
    "Monitor and analyze fraudulent transactions across different payment"
    " channels."
)

# Summary Metrics (Using full dataframe)
col1, col2, col3 = st.columns(3)
total_transactions = len(df)
total_fraud = int(df["fraud_flag"].sum()) if "fraud_flag" in df.columns else 0
fraud_rate = (total_fraud / total_transactions) * 100 if total_transactions > 0 else 0.0

with col1:
  st.metric(label="Total Transactions", value=f"{total_transactions:,}")
with col2:
  st.metric(label="Total Fraud Cases", value=f"{total_fraud:,}")
with col3:
  st.metric(label="Fraud Rate (%)", value=f"{fraud_rate:.2f}%")

st.markdown("---")

# Data Overview Section
st.header("📋 Data Overview")
tab_preview, tab_stats, tab_missing, tab_duplicates = st.tabs(
    ["Dataset Sample", "Descriptive Statistics", "Missing Values Audit", "Duplicates Audit",]
)

with tab_preview:
  st.subheader("Raw Data Preview")
  st.markdown("Showing the first 10 rows of the transaction dataset:")

  # إنشاء نسخة مؤقتة للعرض فقط لتحويل الـ boolean إلى نص (True/False)
  preview_df = df.head(10).copy()
  if "fraud_flag" in preview_df.columns:
    preview_df["fraud_flag"] = preview_df["fraud_flag"].astype(str)

  st.dataframe(preview_df, use_container_width=True)

with tab_stats:
  st.subheader("Statistical Summary (Numerical Features)")
  numeric_df = df.select_dtypes(include=[np.number])
  if not numeric_df.empty:
    st.dataframe(numeric_df.describe().round(2), use_container_width=True)
  else:
    st.info("No numerical features available.")

with tab_missing:
  st.subheader("Missing Values Percentage (%)")
  missing_data = (df.isna().mean() * 100).round(2)
  missing_df = pd.DataFrame({"Missing Percentage (%)": missing_data})
  st.dataframe(
      missing_df[missing_df["Missing Percentage (%)"] > 0],
      use_container_width=True,
  )
  if missing_df["Missing Percentage (%)"].max() == 0:
    st.success("No missing values found!")

# كود التبويب الجديد الخاص بفحص التكرار
with tab_duplicates:
  st.subheader("Duplicates Check")
  duplicate_count = df.duplicated().sum()

  if duplicate_count > 0:
    st.warning(f"Found {duplicate_count:,} duplicate rows in the dataset.")
    st.dataframe(df[df.duplicated(keep=False)], use_container_width=True)
  else:
    st.success("No duplicate rows found! The dataset is completely clean.")