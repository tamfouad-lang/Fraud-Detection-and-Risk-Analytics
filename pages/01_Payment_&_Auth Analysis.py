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

# ---------------------------------------------------------
# 1. Page Configuration & Global Styling (يجب أن يكون أول أمر Streamlit)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Univariate Analysis",
    page_icon="📊",
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
# 3. Sidebar Filters (Combined Filters)
# ---------------------------------------------------------
st.sidebar.header("Filter Options")

# --- Filter 1: Payment Channel ---
if "payment_channel" in df.columns:
  unique_channels = df["payment_channel"].dropna().unique().tolist()
  selected_channels = st.sidebar.multiselect(
      "Select Payment Channel(s):",
      options=unique_channels,
      default=unique_channels,
  )
  filtered_df = (
      df[df["payment_channel"].isin(selected_channels)]
      if selected_channels
      else df.iloc[0:0]
  )
else:
  filtered_df = df.copy()

# --- Filter 2: Authentication Type (بناءً على البيانات المصفاة مسبقاً) ---
if "authentication_type" in filtered_df.columns:
  unique_auth = filtered_df["authentication_type"].dropna().unique().tolist()
  selected_auth = st.sidebar.multiselect(
      "Select Authentication Type(s):",
      options=unique_auth,
      default=unique_auth,
  )
  filtered_df = (
      filtered_df[filtered_df["authentication_type"].isin(selected_auth)]
      if selected_auth
      else filtered_df.iloc[0:0]
  )
else:
  st.sidebar.warning("Column 'authentication_type' not found in dataset.")

# ---------------------------------------------------------
# 4. Main Dashboard Content
# ---------------------------------------------------------
st.title("📊 Fraud Risk Analytics")
st.markdown(
    "Monitor and analyze fraudulent transactions across payment channels and"
    " authentication types."
)

st.write(f"**Total Transactions after filtering:** {len(filtered_df):,}")
st.markdown("---")

# ---------------------------------------------------------
# Part 1: Payment Channel Analysis
# ---------------------------------------------------------
st.header("1. Analysis by Payment Channel")

if not filtered_df.empty and "payment_channel" in filtered_df.columns:
  fraud_by_channel = (
      filtered_df.groupby("payment_channel")["fraud_flag"]
      .agg(["count", "sum"])
      .reset_index()
  )
  fraud_by_channel.columns = [
      "Payment_Channel",
      "Total_Transactions",
      "Fraud_Count",
  ]
  fraud_by_channel = fraud_by_channel.sort_values(
      by="Fraud_Count", ascending=False
  ).reset_index(drop=True)

  st.subheader("Fraud Summary by Payment Channel")
  st.dataframe(fraud_by_channel, use_container_width=True)

  # Chart 1: Bar Chart
  fig1 = px.bar(
      fraud_by_channel,
      x="Payment_Channel",
      y="Fraud_Count",
      title="Fraud Count by Payment Channel",
      text_auto=True,
      color="Payment_Channel",
  )
  st.plotly_chart(fig1, use_container_width=True)

  # Calculate Percentage & Grouped Bar Chart for Channels
  fraud_by_channel["Fraud_Percentage"] = (
      (
          fraud_by_channel["Fraud_Count"]
          / fraud_by_channel["Total_Transactions"]
      )
      * 100
  ).round(2)
  fraud_by_channel = fraud_by_channel.sort_values(
      by="Fraud_Percentage", ascending=False
  ).reset_index(drop=True)

  fraud_melted_channel = fraud_by_channel.melt(
      id_vars=["Payment_Channel", "Fraud_Percentage"],
      value_vars=["Total_Transactions", "Fraud_Count"],
      var_name="Metric_Type",
      value_name="Count",
  )

  fig2 = px.bar(
      fraud_melted_channel,
      x="Payment_Channel",
      y="Count",
      color="Metric_Type",
      barmode="group",
      title=(
          "Total Transactions vs. Fraud Count per Payment Channel (with %"
          " Rate)"
      ),
      text_auto=True,
      hover_data=["Fraud_Percentage"],
  )
  st.plotly_chart(fig2, use_container_width=True)

else:
  st.warning("No data available for Payment Channel with current filters.")

st.markdown("---")

# ---------------------------------------------------------
# Part 2: Authentication Type Analysis
# ---------------------------------------------------------
st.header("2. Analysis by Authentication Type")

if not filtered_df.empty and "authentication_type" in filtered_df.columns:
  fraud_by_auth = (
      filtered_df.groupby("authentication_type")["fraud_flag"]
      .agg(["count", "sum"])
      .reset_index()
  )
  fraud_by_auth.columns = [
      "authentication_type",
      "Total_Transactions",
      "Fraud_Count",
  ]
  fraud_by_auth = fraud_by_auth.sort_values(
      by="Fraud_Count", ascending=False
  ).reset_index(drop=True)

  st.subheader("Fraud Summary by Authentication Type")
  st.dataframe(fraud_by_auth, use_container_width=True)

  # Chart 3: Authentication Fraud Count
  fig_auth_1 = px.bar(
      fraud_by_auth,
      x="authentication_type",
      y="Fraud_Count",
      title="Fraud Count by Authentication Type",
      text_auto=True,
      color="Fraud_Count",
      color_continuous_scale="Reds",
  )
  st.plotly_chart(fig_auth_1, use_container_width=True)

  # Calculate Percentage & Grouped Bar Chart for Authentication
  fraud_by_auth["Fraud_Percentage"] = (
      (fraud_by_auth["Fraud_Count"] / fraud_by_auth["Total_Transactions"]) * 100
  ).round(2)
  fraud_by_auth = fraud_by_auth.sort_values(
      by="Fraud_Percentage", ascending=False
  ).reset_index(drop=True)

  fraud_melted_auth = fraud_by_auth.melt(
      id_vars=["authentication_type", "Fraud_Percentage"],
      value_vars=["Total_Transactions", "Fraud_Count"],
      var_name="Metric_Type",
      value_name="Count",
  )

  fig_auth_2 = px.bar(
      fraud_melted_auth,
      x="authentication_type",
      y="Count",
      color="Metric_Type",
      barmode="group",
      title=(
          "Total Transactions vs. Fraud Count per Authentication Type (with %"
          " Rate)"
      ),
      text_auto=True,
      hover_data=["Fraud_Percentage"],
      color_discrete_sequence=["#2b5c8f", "#e67e22"],
  )
  st.plotly_chart(fig_auth_2, use_container_width=True)

else:
  st.warning("No data available for Authentication Type with current filters.")

  # ---------------------------------------------------------
# Part 3: Fraud Distribution Share (Pie Charts inside Tabs)
# ---------------------------------------------------------
st.header("3. Fraud Distribution Share (Pie Charts)")
st.markdown("Compare the proportions of fraud across different dimensions.")

# إنشاء التابات (Tabs) بجانب بعضها
tab_pie1, tab_pie2 = st.tabs(
    ["💳 Fraud by Payment Channel", "🔐 Fraud by Authentication Type"]
)

# Tab 1: Payment Channel Pie Chart
with tab_pie1:
  if not filtered_df.empty and "payment_channel" in filtered_df.columns:
    fig_pie = px.pie(
        fraud_by_channel,
        names="Payment_Channel",
        values="Fraud_Count",
        title="Fraud Distribution by Payment Channel",
        hover_data=["Total_Transactions", "Fraud_Percentage"],
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)
  else:
    st.info("No data available for Payment Channel.")

# Tab 2: Authentication Type Donut/Pie Chart
with tab_pie2:
  if not filtered_df.empty and "authentication_type" in filtered_df.columns:
    fig_pie_2 = px.pie(
        fraud_by_auth,
        names="authentication_type",
        values="Fraud_Count",
        title="Fraud Distribution by Authentication Type",
        hover_data=["Total_Transactions", "Fraud_Percentage"],
        hole=0.5,  # تحويلها لـ Donut Chart شكلها احترافي جداً
    )
    fig_pie_2.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_pie_2, use_container_width=True)
  else:
    st.info("No data available for Authentication Type.")