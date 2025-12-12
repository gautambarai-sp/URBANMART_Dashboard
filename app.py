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
def load_data(path: str = "data/urbanmart_sales.csv") -> pd.DataFrame:
    """Load CSV and do comprehensive cleaning"""
    df = pd.read_csv(path, low_memory=False)
    
    # Parse date column with multiple format attempts
    if 'date' in df.columns:
        # Try different date formats
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%y', errors='coerce')
        
        # If parsing failed, try alternative formats
        if df['date'].isna().all():
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Drop rows with invalid dates
        df = df[df['date'].notna()].copy()
    
    # Ensure numeric columns
    numeric_cols = ['quantity', 'unit_price', 'discount_applied']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Calculate line revenue
    if 'quantity' in df.columns and 'unit_price' in df.columns:
        df['line_revenue'] = (df['quantity'] * df['unit_price']) - df.get('discount_applied', 0)
    
    # Add derived columns for analysis
    if 'date' in df.columns and df['date'].notna().any():
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['month_name'] = df['date'].dt.strftime('%B')
        df['quarter'] = df['date'].dt.quarter
        df['day_of_week'] = df['date'].dt.day_name()
        df['week'] = df['date'].dt.isocalendar().week
    
    # Calculate profit margin (assuming 40% margin as default)
    df['profit'] = df['line_revenue'] * 0.4
    
    return df

def create_slicers(df: pd.DataFrame) -> Dict:
    """Create interactive slicers (filters) in sidebar"""
    st.sidebar.title("🎯 Slicers & Filters")
    st.sidebar.markdown("---")
    
    filters = {}
    
    # Date range slicer
    if 'date' in df.columns and df['date'].notna().any():
        st.sidebar.subheader("📅 Date Range")
        min_date = df['date'].min()
        max_date = df['date'].max()
        
        # Convert to date objects (not Timestamp)
        if pd.notna(min_date) and pd.notna(max_date):
            min_date = min_date.date()
            max_date = max_date.date()
            
            date_range = st.sidebar.date_input(
                "Select Period",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            if isinstance(date_range, tuple) and len(date_range) == 2:
                filters['date'] = (pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))
            st.sidebar.markdown("---")
    
    # Store Location slicer
    if 'store_location' in df.columns:
        st.sidebar.subheader("🏪 Store Location")
        locations = sorted(df['store_location'].dropna().unique())
        selected_locations = st.sidebar.multiselect(
            "Select Locations",
            options=locations,
            default=locations
        )
        filters['store_location'] = selected_locations
        st.sidebar.markdown("---")
    
    # Product Category slicer
    if 'product_category' in df.columns:
        st.sidebar.subheader("📦 Product Category")
        categories = sorted(df['product_category'].dropna().unique())
        selected_categories = st.sidebar.multiselect(
            "Select Categories",
            options=categories,
            default=categories
        )
        filters['product_category'] = selected_categories
        st.sidebar.markdown("---")
    
    # Customer Segment slicer
    if 'customer_segment' in df.columns:
        st.sidebar.subheader("👥 Customer Segment")
        segments = sorted(df['customer_segment'].dropna().unique())
        selected_segments = st.sidebar.multiselect(
            "Select Segments",
            options=segments,
            default=segments
        )
        filters['customer_segment'] = selected_segments
        st.sidebar.markdown("---")
    
    # Channel slicer
    if 'channel' in df.columns:
        st.sidebar.subheader("🛒 Sales Channel")
        channels = sorted(df['channel'].dropna().unique())
        selected_channels = st.sidebar.multiselect(
            "Select Channels",
            options=channels,
            default=channels
        )
        filters['channel'] = selected_channels
        st.sidebar.markdown("---")
    
    # Payment Method slicer
    if 'payment_method' in df.columns:
        st.sidebar.subheader("💳 Payment Method")
        payments = sorted(df['payment_method'].dropna().unique())
        selected_payments = st.sidebar.multiselect(
            "Select Payment Methods",
            options=payments,
            default=payments
        )
        filters['payment_method'] = selected_payments
        st.sidebar.markdown("---")
    
    # Revenue range slicer
    if 'line_revenue' in df.columns:
        st.sidebar.subheader("💰 Revenue Range")
        min_rev = float(df['line_revenue'].min())
        max_rev = float(df['line_revenue'].max())
        rev_range = st.sidebar.slider(
            "Filter by Revenue",
            min_value=min_rev,
            max_value=max_rev,
            value=(min_rev, max_rev),
            format="₹%.2f"
        )
        filters['line_revenue'] = rev_range
    
    # Reset button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
        st.rerun()
    
    return filters

def apply_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """Apply all selected filters to dataframe"""
    df_filtered = df.copy()
    
    for key, value in filters.items():
        if key == 'date' and isinstance(value, tuple):
            start, end = value
            df_filtered = df_filtered[(df_filtered['date'] >= start) & (df_filtered['date'] <= end)]
        elif key == 'line_revenue' and isinstance(value, tuple):
            min_val, max_val = value
            df_filtered = df_filtered[(df_filtered['line_revenue'] >= min_val) & 
                                     (df_filtered['line_revenue'] <= max_val)]
        elif isinstance(value, list) and len(value) > 0:
            df_filtered = df_filtered[df_filtered[key].isin(value)]
    
    return df_filtered

def create_kpi_metrics(df: pd.DataFrame):
    """Display key performance indicators"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_revenue = df['line_revenue'].sum() if 'line_revenue' in df.columns else 0
    total_profit = df['profit'].sum() if 'profit' in df.columns else 0
    total_transactions = df['transaction_id'].nunique() if 'transaction_id' in df.columns else len(df)
    total_bills = df['bill_id'].nunique() if 'bill_id' in df.columns else len(df)
    avg_order_value = total_revenue / total_bills if total_bills > 0 else 0
    
    col1.metric("💰 Total Revenue", f"₹{total_revenue:,.2f}")
    col2.metric("📈 Total Profit", f"₹{total_profit:,.2f}")
    col3.metric("🛒 Total Transactions", f"{total_transactions:,}")
    col4.metric("🧾 Total Bills", f"{total_bills:,}")
    col5.metric("💵 Avg Order Value", f"₹{avg_order_value:,.2f}")

def create_advanced_visualizations(df: pd.DataFrame):
    """Create 5+ advanced business visualizations"""
    
    st.markdown("---")
    st.header("📊 Advanced Business Visualizations")
    
    # Visualization 1: Revenue Trend Analysis (Time Series)
    st.subheader("1️⃣ Revenue Trend Over Time")
    if 'date' in df.columns and df['date'].notna().any():
        daily_revenue = df.groupby('date')['line_revenue'].sum().reset_index()
        fig1 = px.line(daily_revenue, x='date', y='line_revenue',
                      title='Daily Revenue Trend',
                      labels={'line_revenue': 'Revenue (₹)', 'date': 'Date'})
        fig1.add_scatter(x=daily_revenue['date'], y=daily_revenue['line_revenue'],
                        mode='markers', name='Daily Sales', marker=dict(size=4))
        fig1.update_layout(height=400, hovermode='x unified')
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("Date information not available for time series analysis")
    
    # Visualization 2: Category Performance Matrix
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("2️⃣ Category Revenue Distribution")
        if 'product_category' in df.columns:
            cat_revenue = df.groupby('product_category')['line_revenue'].sum().reset_index()
            cat_revenue = cat_revenue.sort_values('line_revenue', ascending=False)
            fig2 = px.bar(cat_revenue, x='product_category', y='line_revenue',
                         title='Revenue by Product Category',
                         labels={'line_revenue': 'Revenue (₹)', 'product_category': 'Category'},
                         color='line_revenue',
                         color_continuous_scale='Blues')
            fig2.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.subheader("3️⃣ Channel Performance")
        if 'channel' in df.columns:
            channel_data = df.groupby('channel')['line_revenue'].sum().reset_index()
            fig3 = px.pie(channel_data, values='line_revenue', names='channel',
                         title='Revenue Distribution by Sales Channel',
                         hole=0.4)
            fig3.update_traces(textposition='inside', textinfo='percent+label')
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)
    
    # Visualization 4: Store Location Heatmap
    st.subheader("4️⃣ Store Location Performance Heatmap")
    if 'store_location' in df.columns and 'product_category' in df.columns:
        heatmap_data = df.pivot_table(
            values='line_revenue',
            index='store_location',
            columns='product_category',
            aggfunc='sum',
            fill_value=0
        )
        fig4 = px.imshow(heatmap_data,
                        labels=dict(x="Product Category", y="Store Location", color="Revenue (₹)"),
                        title="Revenue Heatmap: Store Location × Product Category",
                        color_continuous_scale='RdYlGn',
                        aspect="auto")
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, use_container_width=True)
    
    # Visualization 5: Customer Segment Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("5️⃣ Customer Segment Revenue")
        if 'customer_segment' in df.columns:
            segment_data = df.groupby('customer_segment').agg({
                'line_revenue': 'sum',
                'bill_id': 'nunique'
            }).reset_index()
            segment_data.columns = ['customer_segment', 'total_revenue', 'total_orders']
            segment_data['avg_order_value'] = segment_data['total_revenue'] / segment_data['total_orders']
            
            fig5 = go.Figure()
            fig5.add_trace(go.Bar(
                x=segment_data['customer_segment'],
                y=segment_data['total_revenue'],
                name='Total Revenue',
                marker_color='lightblue'
            ))
            fig5.update_layout(
                title='Revenue by Customer Segment',
                xaxis_title='Customer Segment',
                yaxis_title='Revenue (₹)',
                height=400
            )
            st.plotly_chart(fig5, use_container_width=True)
    
    with col2:
        st.subheader("6️⃣ Payment Method Preferences")
        if 'payment_method' in df.columns:
            payment_data = df.groupby('payment_method').agg({
                'transaction_id': 'count',
                'line_revenue': 'sum'
            }).reset_index()
            payment_data.columns = ['payment_method', 'transaction_count', 'total_revenue']
            
            fig6 = px.scatter(payment_data, x='transaction_count', y='total_revenue',
                            size='total_revenue', color='payment_method',
                            title='Payment Method Analysis',
                            labels={'transaction_count': 'Number of Transactions',
                                   'total_revenue': 'Total Revenue (₹)'},
                            hover_data=['payment_method'])
            fig6.update_layout(height=400)
            st.plotly_chart(fig6, use_container_width=True)
    
    # Visualization 7: Top Products Performance
    st.subheader("7️⃣ Top 10 Products by Revenue")
    if 'product_name' in df.columns:
        product_performance = df.groupby('product_name').agg({
            'line_revenue': 'sum',
            'quantity': 'sum'
        }).reset_index()
        product_performance = product_performance.sort_values('line_revenue', ascending=False).head(10)
        
        fig7 = px.bar(product_performance, x='line_revenue', y='product_name',
                     orientation='h',
                     title='Top 10 Products by Revenue',
                     labels={'line_revenue': 'Revenue (₹)', 'product_name': 'Product'},
                     color='line_revenue',
                     color_continuous_scale='Viridis')
        fig7.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig7, use_container_width=True)
    
    # Visualization 8: Day of Week Analysis
    st.subheader("8️⃣ Sales Pattern by Day of Week")
    if 'day_of_week' in df.columns:
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_data = df.groupby('day_of_week')['line_revenue'].sum().reset_index()
        dow_data['day_of_week'] = pd.Categorical(dow_data['day_of_week'], categories=day_order, ordered=True)
        dow_data = dow_data.sort_values('day_of_week')
        
        fig8 = px.bar(dow_data, x='day_of_week', y='line_revenue',
                     title='Revenue by Day of Week',
                     labels={'line_revenue': 'Revenue (₹)', 'day_of_week': 'Day'},
                     color='line_revenue',
                     color_continuous_scale='Plasma')
        fig8.update_layout(height=400)
        st.plotly_chart(fig8, use_container_width=True)

def generate_business_insights(df: pd.DataFrame):
    """Generate detailed business insights"""
    st.markdown("---")
    st.header("🎯 Detailed Business Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📌 Key Findings")
        insights = []
        
        # Revenue insights
        total_revenue = df['line_revenue'].sum()
        insights.append(f"**Total Revenue Generated:** ₹{total_revenue:,.2f}")
        
        # Best performing category
        if 'product_category' in df.columns:
            top_category = df.groupby('product_category')['line_revenue'].sum().idxmax()
            top_cat_revenue = df.groupby('product_category')['line_revenue'].sum().max()
            top_cat_pct = (top_cat_revenue / total_revenue) * 100
            insights.append(f"**Top Category:** {top_category} (₹{top_cat_revenue:,.2f}, {top_cat_pct:.1f}% of total)")
        
        # Best performing location
        if 'store_location' in df.columns:
            top_location = df.groupby('store_location')['line_revenue'].sum().idxmax()
            top_loc_revenue = df.groupby('store_location')['line_revenue'].sum().max()
            insights.append(f"**Best Store Location:** {top_location} (₹{top_loc_revenue:,.2f})")
        
        # Customer segment analysis
        if 'customer_segment' in df.columns:
            top_segment = df.groupby('customer_segment')['line_revenue'].sum().idxmax()
            top_seg_revenue = df.groupby('customer_segment')['line_revenue'].sum().max()
            insights.append(f"**Most Valuable Segment:** {top_segment} (₹{top_seg_revenue:,.2f})")
        
        # Channel preference
        if 'channel' in df.columns:
            top_channel = df.groupby('channel')['line_revenue'].sum().idxmax()
            top_ch_pct = (df.groupby('channel')['line_revenue'].sum().max() / total_revenue) * 100
            insights.append(f"**Preferred Channel:** {top_channel} ({top_ch_pct:.1f}% of sales)")
        
        for insight in insights:
            st.markdown(f"• {insight}")
    
    with col2:
        st.subheader("💡 Recommendations")
        recommendations = [
            "**Inventory Management:** Focus on high-performing categories and products",
            "**Store Optimization:** Allocate more resources to top-performing locations",
            "**Customer Retention:** Develop loyalty programs for Regular and New segments",
            "**Channel Strategy:** Optimize both Online and In-store experiences",
            "**Pricing Strategy:** Analyze discount effectiveness on overall profitability",
            "**Product Mix:** Consider expanding product lines in high-revenue categories"
        ]
        
        for rec in recommendations:
            st.markdown(f"• {rec}")

def create_review_questions():
    """Create interactive review questions section"""
    st.markdown("---")
    st.header("❓ Data Review Questions")
    st.markdown("*Use these questions to guide your analysis and decision-making*")
    
    with st.expander("📊 Sales Performance Questions", expanded=False):
        st.markdown("""
        1. **Which product category generates the highest revenue and why?**
        2. **What is the trend in sales over the analyzed period? Is it growing or declining?**
        3. **Which store location has the best performance? What makes it successful?**
        4. **What is the average order value, and how does it vary across channels?**
        5. **Which day of the week shows the highest sales? Can we optimize staffing accordingly?**
        """)
    
    with st.expander("👥 Customer Behavior Questions", expanded=False):
        st.markdown("""
        1. **Which customer segment (Loyal, Regular, New) contributes most to revenue?**
        2. **What are the purchasing patterns of different customer segments?**
        3. **How does customer segment affect average order value?**
        4. **What is the relationship between customer segment and product category preferences?**
        5. **Are loyal customers more likely to use specific payment methods or channels?**
        """)
    
    with st.expander("🛒 Channel & Payment Questions", expanded=False):
        st.markdown("""
        1. **What is the revenue split between Online and In-store channels?**
        2. **Which payment method is most popular, and does it vary by channel?**
        3. **Is there a difference in average order value between Online and In-store?**
        4. **How does channel preference vary across different store locations?**
        5. **What is the correlation between payment method and transaction size?**
        """)
    
    with st.expander("📦 Product & Inventory Questions", expanded=False):
        st.markdown("""
        1. **Which are the top 10 products by revenue and quantity sold?**
        2. **What is the average quantity per transaction for each product category?**
        3. **Which products have the highest and lowest profit margins?**
        4. **Are there any seasonal patterns in product sales?**
        5. **Which product combinations are frequently purchased together?**
        """)
    
    with st.expander("💰 Pricing & Discount Questions", expanded=False):
        st.markdown("""
        1. **What is the average discount applied per transaction?**
        2. **How do discounts affect overall revenue and profitability?**
        3. **Which product categories receive the most discounts?**
        4. **Is there a correlation between discount amount and order value?**
        5. **Are discounts more common in specific channels or locations?**
        """)
    
    with st.expander("🎯 Strategic Questions", expanded=False):
        st.markdown("""
        1. **What opportunities exist for revenue growth in underperforming categories?**
        2. **Should we expand or optimize our store locations based on performance?**
        3. **How can we improve customer retention and move customers to Loyal segment?**
        4. **What marketing strategies should we implement for different segments?**
        5. **How should we allocate inventory across different store locations?**
        """)

# -------------------------
# Main Application
# -------------------------
def main():
    st.title("🏪 UrbanMart Advanced Sales Analytics Dashboard")
    st.markdown("*Comprehensive business intelligence for data-driven decision making*")
    
    # Load data
    try:
        df = load_data("data/urbanmart_sales.csv")
        st.success(f"✅ Data loaded successfully: {len(df):,} transactions")
    except FileNotFoundError:
        st.error("❌ Error: Please place your CSV file at 'data/urbanmart_sales.csv'")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()
    
    # Display data overview
    with st.expander("📋 Data Preview & Summary", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", f"{df.shape[0]:,}")
        col2.metric("Total Columns", f"{df.shape[1]}")
        
        if 'date' in df.columns and df['date'].notna().any():
            col3.metric("Date Range", f"{df['date'].min().date()} to {df['date'].max().date()}")
        else:
            col3.metric("Date Range", "N/A")
            
        if 'product_name' in df.columns:
            col4.metric("Unique Products", f"{df['product_name'].nunique():,}")
        else:
            col4.metric("Unique Products", "N/A")
        
        st.dataframe(df.head(10), use_container_width=True)
    
    # Create slicers and apply filters
    filters = create_slicers(df)
    df_filtered = apply_filters(df, filters)
    
    # Show filtered data count
    st.info(f"📊 Showing data for **{len(df_filtered):,}** transactions (filtered from {len(df):,} total)")
    
    # Display KPI metrics
    create_kpi_metrics(df_filtered)
    
    # Create advanced visualizations
    create_advanced_visualizations(df_filtered)
    
    # Generate business insights
    generate_business_insights(df_filtered)
    
    # Review questions
    create_review_questions()
    
    # Footer
    st.markdown("---")
    st.markdown("*Dashboard created with Streamlit | Data-driven insights for UrbanMart*")

if __name__ == "__main__":
    main()
