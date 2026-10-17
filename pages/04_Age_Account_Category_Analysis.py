"""
Fraud Detection & Risk Analytics Dashboard - Account Age Group Analysis
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------
# 1. Page Configuration (يجب أن يكون أول أمر Streamlit حصرياً)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Account Age Group Analysis",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main { background-color: #f8f9fa; }
        .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 2. Data Loading & Caching Pipeline
# ---------------------------------------------------------
@st.cache_data
def load_data():
  return pd.read_parquet("Cleaned_Data_Fraud_Trans.parquet")


df = load_data()

# ---------------------------------------------------------
# 3. Sidebar Controls (Selectbox for Account Age Group with ALL)
# ---------------------------------------------------------
st.sidebar.header("Filter & View Options")

if "Account_Age_Group" in df.columns:
  # الترتيب المنطقي لفئات عمر الحساب (يمكن تعديله حسب قيم الداتا لديك)
  logical_order = ["New", "Medium", "Established", "Old", "< 1 Year", "1-3 Years", "3+ Years"]
  unique_ages = df["Account_Age_Group"].dropna().unique().tolist()
  
  # مطابقة الترتيب المنطقي مع الموجود في البيانات، أو استخراج الفريد منها مباشرة
  categories = [cat for cat in logical_order if cat in unique_ages]
  # إذا كانت القيم لا تطابق القائمة الملقاة، نأخذ الفريدة مباشرة
  if not categories:
    categories = unique_ages

  categories.insert(0, "ALL")  # إضافة خيار ALL في البداية

  selected_age_group = st.sidebar.selectbox(
      "Select Account Age Group:", options=categories
  )

  # تصفية البيانات بناءً على اختيار المستخدم
  if selected_age_group == "ALL":
    filtered_df = df.copy()
  else:
    filtered_df = df[df["Account_Age_Group"] == selected_age_group]
else:
  st.sidebar.warning("Column 'Account_Age_Group' not found in dataset.")
  filtered_df = df.copy()

# ---------------------------------------------------------
# 4. Main Dashboard Content
# ---------------------------------------------------------
st.title("👥 Fraud Analysis by Account Age Group")
st.markdown(
    "Analyze and compare fraud distributions, total transactions, and fraud percentages across different account age groups."
)

st.write(
    f"**Selected Account Age Group:** {selected_age_group} | **Total Transactions (Filtered):** {len(filtered_df):,}"
)
st.markdown("---")

# ---------------------------------------------------------
# 5. Visualizations & Tables Logic
# ---------------------------------------------------------
if not filtered_df.empty and "Account_Age_Group" in filtered_df.columns:

  # --- Part 1: Fraud Count Analysis ---
  fraud_by_Account_Age_Group = (
      filtered_df.groupby("Account_Age_Group")["fraud_flag"]
      .agg(["count", "sum"])
      .reset_index()
  )
  fraud_by_Account_Age_Group.columns = [
      "Account_Age_Group",
      "Total_Transactions",
      "Fraud_Count",
  ]

  fraud_by_Account_Age_Group = fraud_by_Account_Age_Group.sort_values(
      by="Fraud_Count", ascending=False
  ).reset_index(drop=True)

  st.subheader("Fraud Summary Table by Account Age Group")
  st.dataframe(fraud_by_Account_Age_Group, use_container_width=True)

  # Chart 1: Fraud Count Bar Chart
  fig_3 = px.bar(
      fraud_by_Account_Age_Group,
      x="Account_Age_Group",
      y="Fraud_Count",
      title="Fraud Count by Account Age Group",
      text_auto=True,
      color="Account_Age_Group",
      color_discrete_sequence=[
          "#9A260F",
          "#e9701a",
          "#ebd613",
          "#2b5c8f",
          "#27ae60",
      ],
  )

  fig_3.update_layout(
      xaxis_title="Account Age Group",
      yaxis_title="Fraud Count Transactions",
  )
  st.plotly_chart(fig_3, use_container_width=True)

  st.markdown("---")

  # --- Part 2: Total Transactions vs. Fraud Count (Grouped & Percentage) ---
  fraud_by_Account_Age_Group["Fraud_Percentage"] = (
      (
          fraud_by_Account_Age_Group["Fraud_Count"]
          / fraud_by_Account_Age_Group["Total_Transactions"]
      )
      * 100
  ).round(2)

  fraud_by_Account_Age_Group = fraud_by_Account_Age_Group.sort_values(
      by="Fraud_Percentage", ascending=False
  ).reset_index(drop=True)

  fraud_melted = fraud_by_Account_Age_Group.melt(
      id_vars=["Account_Age_Group", "Fraud_Percentage"],
      value_vars=["Total_Transactions", "Fraud_Count"],
      var_name="Metric_Type",
      value_name="Count",
  )

  st.subheader("Comparison Table: Total Transactions vs. Fraud Count")
  st.dataframe(fraud_by_Account_Age_Group, use_container_width=True)

  # Chart 2: Grouped Bar Chart (Melted)
  fig = px.bar(
      fraud_melted,
      x="Account_Age_Group",
      y="Count",
      color="Metric_Type",
      barmode="group",
      text_auto=True,
      title="Total Transactions vs. Fraud Count per Account Age Group",
      hover_data=["Fraud_Percentage"],
  )

  fig.update_layout(
      xaxis_title="Account Age Group",
      yaxis_title="Count of The Transactions",
  )
  st.plotly_chart(fig, use_container_width=True)

else:
  st.warning("No data available for the selected account age group.")