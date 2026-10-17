"""
Fraud Detection & Risk Analytics Dashboard - Amount Category Analysis (Combined Views)
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------
# 1. Page Configuration (يجب أن يكون أول أمر Streamlit حصرياً)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Amount Category Analysis",
    page_icon="💳",
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
# 3. Sidebar Controls (Selectbox & Chart Type)
# ---------------------------------------------------------
st.sidebar.header("Filter & View Options")

# أداة اختيار نوع الرسم البياني (Bar أو Scatter) لتنطبق على التحليلين أو التحكم بهما
chart_type = st.sidebar.selectbox(
    "Select Chart Type:", options=["Bar Chart", "Scatter (Bubble) Chart"]
)

# أداة القائمة المنسدلة لاختيار فئة المعاملة أو ALL مع ترتيب منطقي
if "Amount_Category" in df.columns:
  logical_order = ["Micro", "Low", "Medium", "High", "Very High"]
  unique_cats = df["Amount_Category"].dropna().unique().tolist()
  categories = [cat for cat in logical_order if cat in unique_cats]
  categories.insert(0, "ALL")  # إضافة خيار ALL في البداية

  selected_category = st.sidebar.selectbox(
      "Select Amount Category:", options=categories
  )

  # تصفية البيانات بناءً على اختيار المستخدم
  if selected_category == "ALL":
    filtered_df = df.copy()
  else:
    filtered_df = df[df["Amount_Category"] == selected_category]
else:
  st.sidebar.warning("Column 'Amount_Category' not found in dataset.")
  filtered_df = df.copy()

# ---------------------------------------------------------
# 4. Main Dashboard Content Header
# ---------------------------------------------------------
st.title("💳 Comprehensive Amount Category Analysis")
st.markdown(
    "Analyze both **Fraud Counts** and **Fraud Rates (%)** across different"
    " transaction amount categories."
)

st.write(
    f"**Selected Category:** {selected_category} | **Total Transactions"
    f" (Filtered):** {len(filtered_df):,}"
)
st.markdown("---")

# ---------------------------------------------------------
# Part 1: Fraud Count & Total Transactions Analysis
# ---------------------------------------------------------
st.header("1. Fraud Distribution by Count (Total vs. Fraud)")

if not filtered_df.empty and "Amount_Category" in filtered_df.columns:

  # التجميع بناءً على Total_Transactions و Fraud_Count
  fraud_by_count = (
      filtered_df.groupby("Amount_Category")["fraud_flag"]
      .agg(Total_Transactions="count", Fraud_Count="sum")
      .reset_index()
  )

  fraud_by_count = fraud_by_count.sort_values(
      by="Fraud_Count", ascending=False
  ).reset_index(drop=True)

  st.subheader("Fraud Summary Table (Count View)")
  st.dataframe(fraud_by_count, use_container_width=True)

  if chart_type == "Bar Chart":
    fig_bar_count = px.bar(
        fraud_by_count,
        x="Amount_Category",
        y="Fraud_Count",
        title="Fraud Distribution by Amount Category (Fraud Count)",
        text_auto=True,
        color="Fraud_Count",
        hover_data=["Total_Transactions"],
        color_continuous_scale="Reds",
        category_orders={
            "Amount_Category": fraud_by_count["Amount_Category"].tolist()
        },
    )
    fig_bar_count.update_layout(
        xaxis_title="Amount of The Transaction Category",
        yaxis_title="Fraud Count Transactions",
    )
    st.plotly_chart(fig_bar_count, use_container_width=True)

  else:  # Scatter / Bubble Chart
    fig_scatter_count = px.scatter(
        fraud_by_count,
        x="Amount_Category",
        y="Fraud_Count",
        title="Fraud Distribution by Amount Category (Bubble Count)",
        text="Fraud_Count",
        size="Total_Transactions",
        color="Fraud_Count",
        hover_data=["Total_Transactions"],
        color_continuous_scale="Reds",
        category_orders={
            "Amount_Category": fraud_by_count["Amount_Category"].tolist()
        },
    )
    fig_scatter_count.update_traces(textposition="top center")
    fig_scatter_count.update_layout(
        xaxis_title="Amount of The Transaction Category",
        yaxis_title="Fraud Count Transactions",
    )
    st.plotly_chart(fig_scatter_count, use_container_width=True)

else:
  st.warning("No data available for Count Analysis.")

st.markdown("---")

# ---------------------------------------------------------
# Part 2: Fraud Rate (%) Analysis
# ---------------------------------------------------------
st.header("2. Fraud Rate Trend Analysis (%)")

if not filtered_df.empty and "Amount_Category" in filtered_df.columns:

  # التجميع الحسابي للنسبة المئوية Fraud_Rate
  fraud_by_rate = (
      filtered_df.groupby("Amount_Category")["fraud_flag"]
      .agg(
          Total_Transactions="count",
          Fraud_Count="sum",
          Fraud_Rate=lambda x: (x.mean() * 100).round(2),
      )
      .reset_index()
  )

  fraud_by_rate = fraud_by_rate.sort_values(
      by="Fraud_Rate", ascending=False
  ).reset_index(drop=True)

  st.subheader("Fraud Rate Summary Table (Percentage View)")
  st.dataframe(fraud_by_rate, use_container_width=True)

  if chart_type == "Bar Chart":
    fig_bar_rate = px.bar(
        fraud_by_rate,
        x="Amount_Category",
        y="Fraud_Rate",
        title="Fraud Rate (%) by Transaction Amount Category",
        text="Fraud_Rate",
        color="Fraud_Rate",
        hover_data=["Total_Transactions", "Fraud_Count"],
        color_continuous_scale="Reds",
        category_orders={
            "Amount_Category": fraud_by_rate["Amount_Category"].tolist()
        },
    )
    fig_bar_rate.update_traces(textposition="outside")
    fig_bar_rate.update_layout(
        xaxis_title="Transaction Amount Category",
        yaxis_title="Fraud Rate (%)",
    )
    st.plotly_chart(fig_bar_rate, use_container_width=True)

  else:  # Scatter / Bubble Chart
    fig_scatter_rate = px.scatter(
        fraud_by_rate,
        x="Amount_Category",
        y="Fraud_Rate",
        title="Fraud Rate by Amount Category (Bubble Rate)",
        text="Fraud_Rate",
        size="Total_Transactions",
        color="Fraud_Rate",
        hover_data=["Fraud_Count", "Total_Transactions"],
        color_continuous_scale="Reds",
        category_orders={
            "Amount_Category": fraud_by_rate["Amount_Category"].tolist()
        },
    )
    fig_scatter_rate.update_traces(textposition="top center")
    fig_scatter_rate.update_layout(
        xaxis_title="Transaction Amount Category",
        yaxis_title="Fraud Rate (%)",
    )
    st.plotly_chart(fig_scatter_rate, use_container_width=True)

else:
  st.warning("No data available for Rate Analysis.")