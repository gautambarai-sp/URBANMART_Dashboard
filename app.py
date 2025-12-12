import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import List, Dict, Tuple

st.set_page_config(page_title="UrbanMart — Advanced Sales Dashboard", layout="wide")

# -------------------------------------------------------------------
# Load Data (CSV upload option removed)
# -------------------------------------------------------------------
@st.cache_data
def load_data(path: str = "data/urbanmart_sales.csv") -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)

try:
    df = load_data()
except:
    st.error("❌ File not found. Please place 'urbanmart_sales.csv' inside data/.")
    st.stop()

# -------------------------------------------------------------------
# Parse dates & compute revenue
# -------------------------------------------------------------------
def try_parse_dates(df):
    date_cols = []
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")
            if df[col].notna().sum() > 0:
                date_cols.append(col)
    return df, date_cols

df, date_cols = try_parse_dates(df)

def compute_line_revenue(df):
    if "line_revenue" not in df.columns:
        if {"quantity", "unit_price"}.issubset(df.columns):
            df["discount_applied"] = df.get("discount_applied", 0)
            df["line_revenue"] = (
                df["quantity"].astype(float) * df["unit_price"].astype(float)
                - df["discount_applied"].astype(float)
            )
        else:
            df["line_revenue"] = 0
    return df

df = compute_line_revenue(df)

# -------------------------------------------------------------------
# Excel-style Slicers
# -------------------------------------------------------------------
st.sidebar.title("🔎 Quick Slicers")

if date_cols:
    date_col = date_cols[0]
    df["Year"] = df[date_col].dt.year
    df["Month"] = df[date_col].dt.month_name()

    year_selected = st.sidebar.multiselect("Year", sorted(df["Year"].dropna().unique()))
    if year_selected:
        df = df[df["Year"].isin(year_selected)]

    month_selected = st.sidebar.multiselect("Month", df["Month"].unique())
    if month_selected:
        df = df[df["Month"].isin(month_selected)]

if "channel" in df.columns:
    channels = df["channel"].dropna().unique().tolist()
    active_channels = st.sidebar.pills("Channel Slicer", channels, selection_mode="multi")
    if len(active_channels) > 0:
        df = df[df["channel"].isin(active_channels)]

# -------------------------------------------------------------------
# KPIs
# -------------------------------------------------------------------
st.title("UrbanMart — Advanced Sales KPI Dashboard")

total_revenue = df["line_revenue"].sum()
units_sold = df["quantity"].sum() if "quantity" in df.columns else df.shape[0]
unique_bills = df["bill_id"].nunique() if "bill_id" in df.columns else df.shape[0]
aov = total_revenue / unique_bills

k1,k2,k3,k4 = st.columns(4)
k1.metric("Total Revenue", f"₹{total_revenue:,.0f}")
k2.metric("Units Sold", f"{units_sold:,}")
k3.metric("Unique Bills", unique_bills)
k4.metric("Avg Order Value", f"₹{aov:,.2f}")

st.markdown("---")

# -------------------------------------------------------------------
# Advanced Visualizations
# -------------------------------------------------------------------

# 1. Monthly Revenue Trend
if date_cols:
    monthly = df.groupby(df[date_col].dt.to_period("M"))["line_revenue"].sum().reset_index()
    monthly[date_col] = monthly[date_col].astype(str)

    fig = px.line(monthly, x=date_col, y="line_revenue", title="📈 Monthly Revenue Trend")
    st.plotly_chart(fig, use_container_width=True)

# 2. Category Bubble Chart
if "product_category" in df.columns:
    cat = df.groupby("product_category").agg({
        "line_revenue": "sum",
        "quantity": "sum"
    }).reset_index()

    fig2 = px.scatter(
        cat,
        x="quantity",
        y="line_revenue",
        size="line_revenue",
        color="product_category",
        title="🟣 Category Performance Bubble Chart",
        hover_name="product_category"
    )
    st.plotly_chart(fig2, use_container_width=True)

# 3. Revenue Distribution Histogram
fig3 = px.histogram(df, x="line_revenue", nbins=40, title="📊 Revenue Distribution")
st.plotly_chart(fig3, use_container_width=True)

# 4. Heatmap (day-of-week vs hour-of-day)
if date_cols:
    df["Day"] = df[date_col].dt.day_name()
    df["Hour"] = df[date_col].dt.hour

    heat = df.pivot_table(values="line_revenue", index="Day", columns="Hour", aggfunc="sum").fillna(0)
    fig4 = px.imshow(heat, aspect="auto", title="🔥 Sales Heatmap (Day vs Hour)")
    st.plotly_chart(fig4, use_container_width=True)

# -------------------------------------------------------------------
# Detailed Insights Section
# -------------------------------------------------------------------
st.markdown("## 🧠 Detailed Business Insights")

ins = []

# highest revenue category
if "product_category" in df.columns:
    cat_rev = df.groupby("product_category")["line_revenue"].sum()
    top_cat = cat_rev.idxmax()
    ins.append(f"⭐ **Top revenue category:** {top_cat} (₹{cat_rev.max():,.0f})")

# best performing store
if "store_location" in df.columns:
    store_rev = df.groupby("store_location")["line_revenue"].sum()
    ins.append(f"🏪 **Best performing store:** {store_rev.idxmax()}")

# weekday insights
if date_cols:
    day_rev = df.groupby(df[date_col].dt.day_name())["line_revenue"].sum().sort_values(ascending=False)
    ins.append(f"🗓 **Busiest day:** {day_rev.index[0]}")

for i in ins:
    st.write("- " + i)

# -------------------------------------------------------------------
# Review Questions Section
# -------------------------------------------------------------------
st.markdown("---")
st.header("📝 Review Questions")

q1 = st.radio("1. Which category generated the highest revenue?",
              df["product_category"].unique() if "product_category" in df.columns else ["N/A"])

if "product_category" in df.columns:
    correct = df.groupby("product_category")["line_revenue"].sum().idxmax()
    if q1 == correct:
        st.success("Correct!")
    else:
        st.error(f"Incorrect — Correct answer: {correct}")

q2 = st.radio("2. Which day of the week had the most sales?",
              ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"] if date_cols else ["N/A"])

if date_cols:
    correct_day = df.groupby(df[date_col].dt.day_name())["line_revenue"].sum().idxmax()
    if q2 == correct_day:
        st.success("Correct!")
    else:
        st.error(f"Incorrect — Correct answer: {correct_day}")

st.markdown("---")
st.caption("Enhanced UrbanMart Dashboard — with slicers, insights, and review questions.")
