"""
Fraud Detection & Risk Analytics Dashboard - Hourly Trend Analysis
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------
# 1. Page Configuration (يجب أن يكون أول أمر Streamlit)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Hourly Fraud Trend Analysis",
    page_icon="⏰",
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
# 3. Sidebar Scale / Slider for Hours (تحديد نطاق الساعة على Scale)
# ---------------------------------------------------------
st.sidebar.header("Filter Options")

if "transaction_time_hour" in df.columns:
  min_hour = int(df["transaction_time_hour"].min())
  max_hour = int(df["transaction_time_hour"].max())

  # إضافة Slider (Scale) لتحديد الساعات
  selected_hours = st.sidebar.slider(
      "Select Hour Range:",
      min_value=min_hour,
      max_value=max_hour,
      value=(min_hour, max_hour),  # النطاق الافتراضي من البداية للنهاية
  )

  # تصفية البيانات بناءً على نطاق الساعات المختار وتعريف filtered_df بضمان
  filtered_df = df[
      (df["transaction_time_hour"] >= selected_hours[0])
      & (df["transaction_time_hour"] <= selected_hours[1])
  ]
else:
  st.sidebar.warning("Column 'transaction_time_hour' not found in dataset.")
  filtered_df = df.copy()
  selected_hours = (0, 23)

# ---------------------------------------------------------
# 4. Main Dashboard Content
# ---------------------------------------------------------
st.title("⏰ Fraud Rate Trend by Hour of Day")
st.markdown(
    "Analyze how fraud rates fluctuate across different hours of the day within"
    " the selected time scale."
)

st.write(
    f"**Selected Hour Range:** {selected_hours[0]}:00 to"
    f" {selected_hours[1]}:00 | **Total Transactions:** {len(filtered_df):,}"
)
st.markdown("---")

# ---------------------------------------------------------
# 5. Hourly Fraud Rate Calculation & Line Chart
# ---------------------------------------------------------
if not filtered_df.empty and "transaction_time_hour" in filtered_df.columns:

  # حساب معدل الاحتيال لكل ساعة
  fraud_rate_by_hour = (
      (filtered_df.groupby("transaction_time_hour")["fraud_flag"].mean() * 100)
      .sort_values(ascending=False)
      .reset_index()
      .round(2)
  )

  # تغيير أسماء الأعمدة في الـ DataFrame مباشرة لتظهر في الجدول والرسم البياني
  fraud_rate_by_hour = fraud_rate_by_hour.rename(
      columns={
          "transaction_time_hour": "Transaction Time Per Hour",
          "fraud_flag": "Fraud Rate (%)",
      }
  )

  # إعادة الترتيب بناءً على العمود الجديد لعرض الخط بشكل متسلسل
  df_sorted = fraud_rate_by_hour.sort_values(
      by="Transaction Time Per Hour", ascending=True
  )

  # عرض الجدول التحليلي بالعناوين الجديدة مباشرة
  st.subheader("Hourly Fraud Rate Summary Table")
  st.dataframe(df_sorted, use_container_width=True)

  # إنشاء الرسم البياني الخطي باستخدام الأسماء الجديدة للـ DataFrame
  fig = px.line(
      data_frame=df_sorted,
      x="Transaction Time Per Hour",
      y="Fraud Rate (%)",
      markers=True,
      title="Fraud Rate Trend by Hour of Day",
      text="Fraud Rate (%)",
  )

  fig.update_traces(
      textposition="bottom right", line=dict(color="#d9534f", width=2)
  )

  fig.update_layout(
      xaxis_title="Transaction Time Per Hour",
      yaxis_title="Fraud Rate (%)",
  )

  st.plotly_chart(fig, use_container_width=True)

else:
  st.warning(
      "No data available for the selected hour range. Please adjust the"
      " slider."
  )