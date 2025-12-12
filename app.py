# app.py
"""
Enhanced Streamlit dashboard for UrbanMart sales dataset.
Features: Advanced visualizations, business insights, slicers, and review questions.
"""

from typing import Dict, List, Tuple
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="UrbanMart – Advanced Analytics", layout="wide")

# -------------------------
# Helpers & caching
# -------------------------
@st.cache_data
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and transform uploaded or default data"""

    # Parse date
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df[df['date'].notna()].copy()

    # Ensure numeric columns
    numeric_cols = ['quantity', 'unit_price', 'discount_applied']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Calculate revenue
    if 'quantity' in df.columns and 'unit_price' in df.columns:
        df['discount_applied'] = df.get('discount_applied', 0)
        df['line_revenue'] = (df['quantity'] * df['unit_price']) - df['discount_applied']

    # Date breakdown columns
    if 'date' in df.columns:
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['month_name'] = df['date'].dt.strftime('%B')
        df['quarter'] = df['date'].dt.quarter
        df['day_of_week'] = df['date'].dt.day_name()
        df['week'] = df['date'].dt.isocalendar().week

    # Profit (assumed)
    df['profit'] = df['line_revenue'] * 0.4

    return df


# -------------------------
# Slicers
# -------------------------
def create_slicers(df: pd.DataFrame) -> Dict:
    st.sidebar.title("🎯 Slicers & Filters")
    st.sidebar.markdown("---")
    filters = {}

    # Date range slicer
    if 'date' in df.columns:
        min_date, max_date = df['date'].min(), df['date'].max()
        date_range = st.sidebar.date_input(
            "📅 Select Period",
            (min_date.date(), max_date.date())
        )
        if isinstance(date_range, tuple):
            filters['date'] = (pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))

    # Store location slicer
    if 'store_location' in df.columns:
        stores = sorted(df['store_location'].dropna().unique())
        filters['store_location'] = st.sidebar.multiselect("🏪 Store Location", stores, stores)

    # Category slicer
    if 'product_category' in df.columns:
        cats = sorted(df['product_category'].dropna().unique())
        filters['product_category'] = st.sidebar.multiselect("📦 Product Category", cats, cats)

    # Channel slicer
    if 'channel' in df.columns:
        channels = sorted(df['channel'].dropna().unique())
        filters['channel'] = st.sidebar.multiselect("🛒 Channel", channels, channels)

    # Payment method slicer
    if 'payment_method' in df.columns:
        methods = sorted(df['payment_method'].dropna().unique())
        filters['payment_method'] = st.sidebar.multiselect("💳 Payment Method", methods, methods)

    # Revenue range slicer
    if 'line_revenue' in df.columns:
        lo, hi = float(df['line_revenue'].min()), float(df['line_revenue'].max())
        filters['line_revenue'] = st.sidebar.slider("💰 Revenue Range", lo, hi, (lo, hi))

    if st.sidebar.button("🔄 Reset Filters"):
        st.rerun()

    return filters


# -------------------------
# Apply Filters
# -------------------------
def apply_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    df2 = df.copy()

    for k, v in filters.items():
        if k == 'date':
            df2 = df2[(df2['date'] >= v[0]) & (df2['date'] <= v[1])]
        elif k == 'line_revenue':
            df2 = df2[(df2['line_revenue'] >= v[0]) & (df2['line_revenue'] <= v[1])]
        elif isinstance(v, list):
            df2 = df2[df2[k].isin(v)]

    return df2


# -------------------------
# KPI Metrics
# -------------------------
def create_kpi_metrics(df: pd.DataFrame):
    col1, col2, col3, col4, col5 = st.columns(5)

    total_rev = df['line_revenue'].sum()
    total_profit = df['profit'].sum()
    transactions = df['transaction_id'].nunique() if 'transaction_id' in df.columns else len(df)
    bills = df['bill_id'].nunique() if 'bill_id' in df.columns else len(df)
    aov = total_rev / bills if bills > 0 else 0

    col1.metric("💰 Total Revenue", f"₹{total_rev:,.2f}")
    col2.metric("📈 Total Profit", f"₹{total_profit:,.2f}")
    col3.metric("🛒 Transactions", f"{transactions:,}")
    col4.metric("🧾 Bills", f"{bills:,}")
    col5.metric("💵 Avg Order Value", f"₹{aov:,.2f}")


# -------------------------
# Visualizations
# -------------------------
def create_advanced_visualizations(df: pd.DataFrame):
    st.markdown("---")
    st.header("📊 Advanced Visualizations")

    # Revenue Trend
    if 'date' in df.columns:
        st.subheader("📈 Revenue Trend Over Time")
        rev = df.groupby('date')['line_revenue'].sum().reset_index()
        fig = px.line(rev, x='date', y='line_revenue')
        st.plotly_chart(fig, use_container_width=True)

    # Category revenue
    if 'product_category' in df.columns:
        st.subheader("📦 Revenue by Category")
        cat = df.groupby('product_category')['line_revenue'].sum().reset_index()
        fig = px.bar(cat, x='product_category', y='line_revenue')
        st.plotly_chart(fig, use_container_width=True)

    # Top products
    if 'product_name' in df.columns:
        st.subheader("🏆 Top 10 Products")
        prod = df.groupby('product_name')['line_revenue'].sum().nlargest(10).reset_index()
        fig = px.bar(prod, y='product_name', x='line_revenue', orientation='h')
        st.plotly_chart(fig, use_container_width=True)


# -------------------------
# Business Insights
# -------------------------
def generate_business_insights(df: pd.DataFrame):
    st.markdown("---")
    st.header("🎯 Business Insights")

    insights = []

    total_rev = df['line_revenue'].sum()
    insights.append(f"• Total revenue: **₹{total_rev:,.2f}**")

    if 'product_category' in df.columns:
        top_cat = df.groupby('product_category')['line_revenue'].sum().idxmax()
        insights.append(f"• Top category: **{top_cat}**")

    if 'day_of_week' in df.columns:
        best_day = df.groupby('day_of_week')['line_revenue'].sum().idxmax()
        insights.append(f"• Busiest day: **{best_day}**")

    for i in insights:
        st.write(i)


# -------------------------
# Review Questions
# -------------------------
def create_review_questions():
    st.markdown("---")
    st.header("❓ Review Questions")

    st.write("""
    1. Which category performs best?
    2. Which store location generates highest revenue?
    3. What is the highest-selling product?
    4. Which channel is most profitable?
    5. Which day of week shows highest sales?
    """)


# -------------------------
# MAIN APPLICATION
# -------------------------
def main():
    st.title("🏪 UrbanMart Advanced Sales Dashboard")
    st.markdown("Comprehensive analytics for data-driven business decisions")

    # -------------------------
    # FILE UPLOAD (FIXED)
    # -------------------------
    st.sidebar.header("📁 Upload CSV File")
    uploaded_file = st.sidebar.file_uploader("Upload your sales CSV file", type=["csv"])

    if uploaded_file:
        df_raw = pd.read_csv(uploaded_file)
        df = clean_data(df_raw)
        st.success(f"📄 Loaded uploaded file: {len(df):,} rows")
    else:
        try:
            df_raw = pd.read_csv("data/urbanmart_sales.csv")
            df = clean_data(df_raw)
            st.success(f"📄 Loaded default file: {len(df):,} rows")
        except:
            st.error("❌ No file uploaded and 'data/urbanmart_sales.csv' not found.")
            st.stop()

    # Preview
    with st.expander("📋 Data Preview"):
        st.dataframe(df.head(10), use_container_width=True)

    # Filters
    filters = create_slicers(df)
    df_filtered = apply_filters(df, filters)

    st.info(f"📊 Showing **{len(df_filtered):,}** filtered records")

    # KPIs
    create_kpi_metrics(df_filtered)

    # Visuals
    create_advanced_visualizations(df_filtered)

    # Insights
    generate_business_insights(df_filtered)

    # Review Questions
    create_review_questions()

    st.markdown("---")
    st.caption("UrbanMart Dashboard • Built with Streamlit")


if __name__ == "__main__":
    main()
