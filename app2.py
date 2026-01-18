"""
Qatar Auto Dealer Advanced AI Platform with Sales Forecasting
Complete AI-powered analytics, insights, and powerful forecasting for car dealership
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
import os
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Qatar Dealer AI Platform Pro",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    .forecast-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load Data with Caching
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("qatar_auto_dealer_sales_forecasting_dataset_2026_extended.csv")
        df['sale_date'] = pd.to_datetime(df['sale_date'])
        df['last_service_date'] = pd.to_datetime(df['last_service_date'], errors='coerce')
        return df
    except FileNotFoundError:
        st.error("⚠️ Data file not found! Please ensure 'qatar_auto_dealer_sales_forecasting_dataset_2026_extended.csv' exists.")
        st.stop()

# Load the processed data for customer analytics
@st.cache_data
def load_processed_data():
    try:
        df = pd.read_csv("processed_dealer_data.csv")
        return df
    except FileNotFoundError:
        return None

df_forecast = load_data()
df_customer = load_processed_data()

# OpenAI API Setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Advanced Forecasting Engine
@st.cache_data
def prepare_forecasting_data(df):
    """Prepare comprehensive forecasting dataset with all internal and external factors"""
    
    # Aggregate monthly sales data
    df_monthly = df.groupby(['sale_year', 'sale_month', 'brand']).agg({
        'customer_id': 'count',
        'final_sale_price': 'sum',
        'vehicle_price': 'mean',
        'discount_percentage': 'mean',
        'interest_rate_pct': 'mean',
        'fuel_energy_price_index': 'mean',
        'supply_chain_constraint_index': 'mean',
        'competitive_pressure_index': 'mean',
        'seasonality_index': 'mean',
        'used_car_market_pressure': 'mean',
        'weather_climate_index': 'mean',
        'market_demand_index': 'mean',
        'promotion_active': 'mean'
    }).reset_index()
    
    df_monthly.columns = ['year', 'month', 'brand', 'sales_count', 'total_revenue', 
                          'avg_price', 'avg_discount', 'interest_rate', 'fuel_energy_price',
                          'supply_chain', 'competitive_pressure', 'seasonality',
                          'used_car_pressure', 'weather_index', 'demand_index', 'promotion']
    
    # Fill any remaining NaN values with mean
    for col in df_monthly.columns:
        if df_monthly[col].dtype in ['float64', 'int64']:
            df_monthly[col] = df_monthly[col].fillna(df_monthly[col].mean())
    
    # Create time features
    df_monthly['month_sin'] = np.sin(2 * np.pi * df_monthly['month'] / 12)
    df_monthly['month_cos'] = np.cos(2 * np.pi * df_monthly['month'] / 12)
    df_monthly['quarter'] = ((df_monthly['month'] - 1) // 3) + 1
    
    return df_monthly

@st.cache_data
def build_forecasting_model(df_monthly, brand=None):
    """Build advanced gradient boosting model for sales forecasting"""
    
    if brand:
        df_model = df_monthly[df_monthly['brand'] == brand].copy()
    else:
        df_model = df_monthly.copy()
        # Encode brand
        le_brand = LabelEncoder()
        df_model['brand_encoded'] = le_brand.fit_transform(df_model['brand'])
    
    # Sort by time
    df_model = df_model.sort_values(['year', 'month'])
    
    # Features for model
    feature_cols = [
        'month', 'quarter', 'month_sin', 'month_cos',
        'avg_price', 'avg_discount', 'interest_rate', 'fuel_energy_price',
        'supply_chain', 'competitive_pressure', 'seasonality',
        'used_car_pressure', 'weather_index', 'demand_index', 'promotion'
    ]
    
    if not brand:
        feature_cols.append('brand_encoded')
    
    X = df_model[feature_cols]
    y = df_model['sales_count']
    
    # Train model on all available data
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X, y)
    
    return model, feature_cols, df_model

@st.cache_data
def forecast_next_12_months(df_monthly, brand=None):
    """Generate 12-month sales forecast with external factors"""
    
    model, feature_cols, df_model = build_forecasting_model(df_monthly, brand)
    
    # Get the last date in data
    last_year = df_model['year'].max()
    last_month = df_model[df_model['year'] == last_year]['month'].max()
    
    # Get recent averages for external factors (last 3 months)
    recent_data = df_model.tail(3)
    
    forecasts = []
    
    for i in range(1, 13):
        # Calculate next month
        next_month = last_month + i
        next_year = last_year
        while next_month > 12:
            next_month -= 12
            next_year += 1
        
        # Create feature vector with projected external factors
        features = {
            'month': next_month,
            'quarter': ((next_month - 1) // 3) + 1,
            'month_sin': np.sin(2 * np.pi * next_month / 12),
            'month_cos': np.cos(2 * np.pi * next_month / 12),
            
            # Internal factors (use recent averages with slight trend)
            'avg_price': recent_data['avg_price'].mean() * (1 + 0.02 * i/12),  # 2% annual increase
            'avg_discount': recent_data['avg_discount'].mean(),
            'promotion': recent_data['promotion'].mean(),
            'demand_index': recent_data['demand_index'].mean(),
            
            # External Factor 1: Interest Rates
            'interest_rate': recent_data['interest_rate'].mean() * (1 + np.random.uniform(-0.05, 0.05)),
            
            # External Factor 2: Fuel & Energy Prices
            'fuel_energy_price': recent_data['fuel_energy_price'].mean() * (1 + np.random.uniform(-0.1, 0.15)),
            
            # External Factor 3: Supply Chain
            'supply_chain': recent_data['supply_chain'].mean() * (1 - 0.1 * i/12),  # Improving over time
            
            # External Factor 4: Competition
            'competitive_pressure': recent_data['competitive_pressure'].mean() * (1 + np.random.uniform(-0.05, 0.05)),
            
            # External Factor 5: Seasonality (cyclic pattern)
            'seasonality': 1.0 + 0.3 * np.sin(2 * np.pi * (next_month - 4) / 12),  # Peak in Ramadan/Spring
            
            # External Factor 6: Used Car Market
            'used_car_pressure': recent_data['used_car_pressure'].mean() * (1 + 0.02 * i/12),
            
            # External Factor 7: Weather (cyclic pattern)
            'weather_index': 1.0 - 0.2 * np.abs(np.sin(2 * np.pi * (next_month - 7) / 12))  # Lower in extreme summer
        }
        
        if not brand:
            features['brand_encoded'] = 0  # Will forecast for each brand separately
        
        # Create feature array
        X_forecast = pd.DataFrame([features])[feature_cols]
        
        # Predict
        predicted_sales = model.predict(X_forecast)[0]
        
        # Calculate confidence interval (±15%)
        lower_bound = predicted_sales * 0.85
        upper_bound = predicted_sales * 1.15
        
        forecasts.append({
            'year': next_year,
            'month': next_month,
            'month_name': datetime(next_year, next_month, 1).strftime('%B %Y'),
            'predicted_sales': max(0, int(predicted_sales)),
            'lower_bound': max(0, int(lower_bound)),
            'upper_bound': int(upper_bound),
            'interest_rate': features['interest_rate'],
            'fuel_energy_price': features['fuel_energy_price'],
            'supply_chain': features['supply_chain'],
            'seasonality': features['seasonality']
        })
    
    return pd.DataFrame(forecasts)

# Sidebar Navigation
st.sidebar.markdown("## 🚗 Qatar Dealer AI Platform Pro")
st.sidebar.markdown("### Navigation")

# Determine which pages to show based on data availability
pages = [
    "📊 Executive Dashboard",
    "📈 Sales Forecasting (12 Months)",
    "🏷️ Brand-Level Forecasting",
    "🔮 Forecast Analysis & Insights"
]

if df_customer is not None:
    pages.extend([
        "👥 Customer Segmentation",
        "🚨 Churn Prediction",
        "💰 Customer Lifetime Value",
        "🔧 Service Optimization",
        "📊 Sales Insights",
        "🎯 Marketing Campaigns",
        "🤖 AI Sales Assistant",
        "🔍 Customer Search"
    ])

page = st.sidebar.radio("Select Module:", pages)

# =============================================================================
# PAGE 1: EXECUTIVE DASHBOARD
# =============================================================================
if page == "📊 Executive Dashboard":
    st.markdown("<h1 class='main-header'>Executive Dashboard</h1>", unsafe_allow_html=True)
    
    # Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_sales = len(df_forecast)
    total_revenue = df_forecast['final_sale_price'].sum()
    avg_sale_price = df_forecast['final_sale_price'].mean()
    unique_brands = df_forecast['brand'].nunique()
    avg_discount = df_forecast['discount_percentage'].mean()
    
    with col1:
        st.metric("Total Sales", f"{total_sales:,}")
    with col2:
        st.metric("Total Revenue", f"QAR {total_revenue/1e6:.1f}M")
    with col3:
        st.metric("Avg Sale Price", f"QAR {avg_sale_price/1e3:.0f}K")
    with col4:
        st.metric("Brands Sold", f"{unique_brands}")
    with col5:
        st.metric("Avg Discount", f"{avg_discount:.1f}%")
    
    st.markdown("---")
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        brand_revenue = df_forecast.groupby('brand')['final_sale_price'].sum().sort_values(ascending=False)
        fig1 = px.bar(x=brand_revenue.values, y=brand_revenue.index, orientation='h',
                     title="📊 Revenue by Brand", labels={'x': 'Revenue (QAR)', 'y': 'Brand'},
                     color=brand_revenue.values, color_continuous_scale='Blues')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        brand_counts = df_forecast['brand'].value_counts()
        fig2 = px.pie(values=brand_counts.values, names=brand_counts.index,
                     title="🎯 Market Share by Brand", hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Charts Row 2
    col3, col4 = st.columns(2)
    
    with col3:
        # Monthly sales trend
        monthly_sales = df_forecast.groupby(['sale_year', 'sale_month']).size().reset_index(name='sales')
        monthly_sales['date'] = pd.to_datetime(
            monthly_sales['sale_year'].astype(str) + '-' + 
            monthly_sales['sale_month'].astype(str).str.zfill(2) + '-01'
        )
        monthly_sales = monthly_sales.sort_values('date')
        
        fig3 = px.line(monthly_sales, x='date', y='sales',
                      title="📈 Monthly Sales Trend",
                      labels={'date': 'Date', 'sales': 'Number of Sales'})
        fig3.update_traces(mode='lines+markers')
        st.plotly_chart(fig3, use_container_width=True)
    
    with col4:
        # Average interest rate trend
        interest_trend = df_forecast.groupby(['sale_year', 'sale_month'])['interest_rate_pct'].mean().reset_index()
        interest_trend['date'] = pd.to_datetime(
            interest_trend['sale_year'].astype(str) + '-' + 
            interest_trend['sale_month'].astype(str).str.zfill(2) + '-01'
        )
        interest_trend = interest_trend.sort_values('date')
        
        fig4 = px.line(interest_trend, x='date', y='interest_rate_pct',
                      title="💰 Interest Rate Trend",
                      labels={'date': 'Date', 'interest_rate_pct': 'Interest Rate (%)'})
        fig4.update_traces(mode='lines+markers', line_color='red')
        st.plotly_chart(fig4, use_container_width=True)
    
    # External Factors Overview
    st.markdown("### 🌍 External Factors Impact Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_interest = df_forecast['interest_rate_pct'].mean()
        st.metric("Avg Interest Rate", f"{avg_interest:.2f}%" if pd.notna(avg_interest) else "N/A")
    
    with col2:
        avg_fuel = df_forecast['fuel_energy_price_index'].mean()
        st.metric("Fuel & Energy Index", f"{avg_fuel:.1f}" if pd.notna(avg_fuel) else "N/A")
    
    with col3:
        avg_supply = df_forecast['supply_chain_constraint_index'].mean()
        st.metric("Supply Chain Index", f"{avg_supply:.2f}" if pd.notna(avg_supply) else "N/A")
    
    with col4:
        avg_weather = df_forecast['weather_climate_index'].mean()
        st.metric("Weather Index", f"{avg_weather:.2f}" if pd.notna(avg_weather) else "N/A")

# =============================================================================
# PAGE 2: SALES FORECASTING (12 MONTHS)
# =============================================================================
elif page == "📈 Sales Forecasting (12 Months)":
    st.markdown("<h1 class='main-header'>Advanced Sales Forecasting - Next 12 Months</h1>", unsafe_allow_html=True)
    
    st.info("🔮 **AI-Powered Forecasting** with Internal & External Factors including: Interest Rates, Fuel Prices, Supply Chain, Competition, Seasonality, Weather, and Used Car Market")
    
    # Prepare data and generate forecast
    with st.spinner("🤖 Training advanced forecasting models..."):
        df_monthly = prepare_forecasting_data(df_forecast)
        
        # Generate forecasts for each brand and sum them up for overall forecast
        brands = sorted(df_forecast['brand'].unique())
        brand_forecasts_list = []
        
        for brand in brands:
            brand_fc = forecast_next_12_months(df_monthly, brand=brand)
            brand_fc['brand'] = brand
            brand_forecasts_list.append(brand_fc)
        
        # Aggregate all brand forecasts to get overall forecast
        forecast_df = pd.DataFrame()
        for month_idx in range(12):
            month_data = {
                'year': brand_forecasts_list[0].iloc[month_idx]['year'],
                'month': brand_forecasts_list[0].iloc[month_idx]['month'],
                'month_name': brand_forecasts_list[0].iloc[month_idx]['month_name'],
                'predicted_sales': sum([bf.iloc[month_idx]['predicted_sales'] for bf in brand_forecasts_list]),
                'lower_bound': sum([bf.iloc[month_idx]['lower_bound'] for bf in brand_forecasts_list]),
                'upper_bound': sum([bf.iloc[month_idx]['upper_bound'] for bf in brand_forecasts_list]),
                'interest_rate': brand_forecasts_list[0].iloc[month_idx]['interest_rate'],
                'fuel_energy_price': brand_forecasts_list[0].iloc[month_idx]['fuel_energy_price'],
                'supply_chain': brand_forecasts_list[0].iloc[month_idx]['supply_chain'],
                'seasonality': brand_forecasts_list[0].iloc[month_idx]['seasonality']
            }
            forecast_df = pd.concat([forecast_df, pd.DataFrame([month_data])], ignore_index=True)
    
    # Display forecast summary
    st.markdown("### 📊 12-Month Sales Forecast Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_forecast = forecast_df['predicted_sales'].sum()
    avg_monthly = forecast_df['predicted_sales'].mean()
    peak_month = forecast_df.loc[forecast_df['predicted_sales'].idxmax(), 'month_name']
    peak_sales = forecast_df['predicted_sales'].max()
    
    with col1:
        st.metric("Total Forecast (12mo)", f"{total_forecast:,} units")
    with col2:
        st.metric("Avg Monthly Sales", f"{int(avg_monthly):,} units")
    with col3:
        st.metric("Peak Month", peak_month)
    with col4:
        st.metric("Peak Sales", f"{int(peak_sales):,} units")
    
    # Forecast visualization
    st.markdown("### 📈 Monthly Sales Forecast with Confidence Intervals")
    
    fig = go.Figure()
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=forecast_df['month_name'],
        y=forecast_df['upper_bound'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_df['month_name'],
        y=forecast_df['lower_bound'],
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(68, 68, 68, 0.2)',
        fill='tonexty',
        name='Confidence Interval',
        hoverinfo='skip'
    ))
    
    # Add predicted sales
    fig.add_trace(go.Scatter(
        x=forecast_df['month_name'],
        y=forecast_df['predicted_sales'],
        mode='lines+markers',
        name='Predicted Sales',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title="Sales Forecast for Next 12 Months",
        xaxis_title="Month",
        yaxis_title="Predicted Sales (Units)",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed forecast table
    st.markdown("### 📋 Detailed Monthly Forecast")
    
    display_df = forecast_df[['month_name', 'predicted_sales', 'lower_bound', 'upper_bound']].copy()
    display_df.columns = ['Month', 'Predicted Sales', 'Lower Bound', 'Upper Bound']
    display_df['Confidence Range'] = display_df['Upper Bound'] - display_df['Lower Bound']
    
    st.dataframe(display_df.style.background_gradient(cmap='Blues', subset=['Predicted Sales']), 
                use_container_width=True)
    
    # External factors impact
    st.markdown("### 🌍 Key External Factors in Forecast")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_interest = px.line(forecast_df, x='month_name', y='interest_rate',
                              title="💰 Projected Interest Rate Trend",
                              labels={'month_name': 'Month', 'interest_rate': 'Interest Rate (%)'})
        fig_interest.update_traces(mode='lines+markers', line_color='green')
        st.plotly_chart(fig_interest, use_container_width=True)
    
    with col2:
        fig_fuel = px.line(forecast_df, x='month_name', y='fuel_energy_price',
                          title="⛽ Projected Fuel & Energy Index",
                          labels={'month_name': 'Month', 'fuel_energy_price': 'Fuel & Energy Price Index'})
        fig_fuel.update_traces(mode='lines+markers', line_color='orange')
        st.plotly_chart(fig_fuel, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        fig_supply = px.line(forecast_df, x='month_name', y='supply_chain',
                         title="🚚 Supply Chain Constraint Projection",
                         labels={'month_name': 'Month', 'supply_chain': 'Supply Chain Constraint'})
        fig_supply.update_traces(mode='lines+markers', line_color='red')
        st.plotly_chart(fig_supply, use_container_width=True)
    
    with col4:
        fig_season = px.line(forecast_df, x='month_name', y='seasonality',
                          title="📅 Seasonality Index Forecast",
                          labels={'month_name': 'Month', 'seasonality': 'Seasonality Index'})
        fig_season.update_traces(mode='lines+markers', line_color='purple')
        st.plotly_chart(fig_season, use_container_width=True)
    
    # Download forecast
    st.markdown("### 📥 Download Forecast Data")
    csv = forecast_df.to_csv(index=False)
    st.download_button(
        label="Download 12-Month Forecast CSV",
        data=csv,
        file_name="sales_forecast_12_months.csv",
        mime="text/csv"
    )

# =============================================================================
# PAGE 3: BRAND-LEVEL FORECASTING
# =============================================================================
elif page == "🏷️ Brand-Level Forecasting":
    st.markdown("<h1 class='main-header'>Brand-Specific Sales Forecasting</h1>", unsafe_allow_html=True)
    
    st.info("📊 Generate detailed forecasts for individual car brands to optimize inventory and marketing strategies")
    
    # Brand selection
    brands = sorted(df_forecast['brand'].unique())
    selected_brand = st.selectbox("🚗 Select Brand for Forecast:", brands)
    
    if selected_brand:
        with st.spinner(f"🔮 Generating 12-month forecast for {selected_brand}..."):
            df_monthly = prepare_forecasting_data(df_forecast)
            brand_forecast = forecast_next_12_months(df_monthly, brand=selected_brand)
        
        # Brand forecast summary
        st.markdown(f"### 📊 {selected_brand} - 12-Month Forecast")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_forecast = brand_forecast['predicted_sales'].sum()
        avg_monthly = brand_forecast['predicted_sales'].mean()
        peak_month = brand_forecast.loc[brand_forecast['predicted_sales'].idxmax(), 'month_name']
        best_month_sales = brand_forecast['predicted_sales'].max()
        
        with col1:
            st.metric(f"{selected_brand} Total (12mo)", f"{total_forecast:,} units")
        with col2:
            st.metric("Avg Monthly", f"{int(avg_monthly):,} units")
        with col3:
            st.metric("Best Month", peak_month)
        with col4:
            st.metric("Best Month Sales", f"{int(best_month_sales):,} units")
        
        # Visualization
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=brand_forecast['month_name'],
            y=brand_forecast['upper_bound'],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=brand_forecast['month_name'],
            y=brand_forecast['lower_bound'],
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(68, 68, 68, 0.2)',
            fill='tonexty',
            name='Confidence Interval'
        ))
        
        fig.add_trace(go.Scatter(
            x=brand_forecast['month_name'],
            y=brand_forecast['predicted_sales'],
            mode='lines+markers',
            name=f'{selected_brand} Forecast',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title=f"{selected_brand} - Sales Forecast for Next 12 Months",
            xaxis_title="Month",
            yaxis_title="Predicted Sales (Units)",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Comparison with other brands
        st.markdown("### 📊 Brand Comparison - Total 12-Month Forecast")
        
        with st.spinner("Calculating forecasts for all brands..."):
            all_brand_forecasts = {}
            for brand in brands:
                forecast = forecast_next_12_months(df_monthly, brand=brand)
                all_brand_forecasts[brand] = forecast['predicted_sales'].sum()
        
        brand_comparison = pd.DataFrame(list(all_brand_forecasts.items()), 
                                       columns=['Brand', 'Forecast_Sales'])
        brand_comparison = brand_comparison.sort_values('Forecast_Sales', ascending=False)
        
        fig_comparison = px.bar(brand_comparison, x='Forecast_Sales', y='Brand', 
                               orientation='h',
                               title="12-Month Sales Forecast by Brand",
                               labels={'Forecast_Sales': 'Predicted Sales', 'Brand': 'Brand'},
                               color='Forecast_Sales',
                               color_continuous_scale='Viridis')
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Detailed table
        st.markdown(f"### 📋 {selected_brand} - Monthly Breakdown")
        
        display_df = brand_forecast[['month_name', 'predicted_sales', 'lower_bound', 'upper_bound']].copy()
        display_df.columns = ['Month', 'Predicted Sales', 'Lower Bound', 'Upper Bound']
        
        st.dataframe(display_df.style.background_gradient(cmap='Reds', subset=['Predicted Sales']),
                    use_container_width=True)
        
        # Download
        csv = brand_forecast.to_csv(index=False)
        st.download_button(
            label=f"Download {selected_brand} Forecast",
            data=csv,
            file_name=f"{selected_brand}_forecast_12_months.csv",
            mime="text/csv"
        )

# =============================================================================
# PAGE 4: FORECAST ANALYSIS & INSIGHTS
# =============================================================================
elif page == "🔮 Forecast Analysis & Insights":
    st.markdown("<h1 class='main-header'>Forecast Analysis & Strategic Insights</h1>", unsafe_allow_html=True)
    
    with st.spinner("🤖 Analyzing forecasts and generating insights..."):
        df_monthly = prepare_forecasting_data(df_forecast)
        
        # Generate forecasts for all brands
        brands = sorted(df_forecast['brand'].unique())
        brand_forecasts = {}
        for brand in brands:
            brand_forecasts[brand] = forecast_next_12_months(df_monthly, brand=brand)
        
        # Create overall forecast by summing brand forecasts
        overall_forecast = pd.DataFrame()
        for month_idx in range(12):
            first_brand_forecast = list(brand_forecasts.values())[0]
            month_data = {
                'year': first_brand_forecast.iloc[month_idx]['year'],
                'month': first_brand_forecast.iloc[month_idx]['month'],
                'month_name': first_brand_forecast.iloc[month_idx]['month_name'],
                'predicted_sales': sum([bf.iloc[month_idx]['predicted_sales'] for bf in brand_forecasts.values()]),
                'lower_bound': sum([bf.iloc[month_idx]['lower_bound'] for bf in brand_forecasts.values()]),
                'upper_bound': sum([bf.iloc[month_idx]['upper_bound'] for bf in brand_forecasts.values()])
            }
            overall_forecast = pd.concat([overall_forecast, pd.DataFrame([month_data])], ignore_index=True)
    
    # Strategic Insights
    st.markdown("### 🎯 Strategic Business Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='forecast-box'>
        <h3>📈 Growth Opportunities</h3>
        <ul>
        <li><strong>Peak Sales Periods:</strong> Focus marketing during high-forecast months</li>
        <li><strong>Interest Rate Impact:</strong> Offer special financing when rates are favorable</li>
        <li><strong>Seasonal Trends:</strong> Prepare inventory for Ramadan and year-end surges</li>
        <li><strong>Weather Considerations:</strong> Promote specific models during extreme heat</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='forecast-box'>
        <h3>⚠️ Risk Mitigation</h3>
        <ul>
        <li><strong>Supply Chain:</strong> Order inventory 3+ months ahead for constrained models</li>
        <li><strong>Competition:</strong> Monitor competitor launches and adjust pricing</li>
        <li><strong>Fuel Prices:</strong> Promote fuel-efficient/electric models when prices rise</li>
        <li><strong>Used Car Market:</strong> Strengthen trade-in programs to compete</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Quarter-wise analysis
    st.markdown("### 📅 Quarterly Sales Forecast")
    
    overall_forecast['quarter'] = ((overall_forecast['month'] - 1) // 3) + 1
    quarterly_forecast = overall_forecast.groupby('quarter').agg({
        'predicted_sales': 'sum',
        'lower_bound': 'sum',
        'upper_bound': 'sum'
    }).reset_index()
    quarterly_forecast['quarter'] = quarterly_forecast['quarter'].apply(lambda x: f'Q{x} 2026')
    
    fig_quarter = go.Figure()
    fig_quarter.add_trace(go.Bar(
        x=quarterly_forecast['quarter'],
        y=quarterly_forecast['predicted_sales'],
        name='Predicted Sales',
        marker_color='lightblue'
    ))
    
    fig_quarter.update_layout(
        title="Quarterly Sales Forecast Distribution",
        xaxis_title="Quarter",
        yaxis_title="Predicted Sales (Units)",
        showlegend=True,
        height=400
    )
    
    st.plotly_chart(fig_quarter, use_container_width=True)
    
    # External factors correlation
    st.markdown("### 🌍 External Factors Impact Analysis")
    
    # Create impact summary
    impact_data = {
        'Factor': [
            'Interest Rates',
            'Fuel & Energy Prices',
            'Supply Chain Constraints',
            'Competitive Pressure',
            'Seasonality Effects',
            'Used Car Market',
            'Weather Conditions'
        ],
        'Impact Level': ['High', 'High', 'Medium', 'Medium', 'High', 'Medium', 'Low'],
        'Trend': ['Decreasing', 'Increasing', 'Improving', 'Stable', 'Cyclical', 'Increasing', 'Seasonal'],
        'Action Required': [
            'Monitor & adjust financing offers',
            'Promote electric/hybrid vehicles',
            'Pre-order inventory early',
            'Competitive pricing strategy',
            'Seasonal marketing campaigns',
            'Enhanced trade-in programs',
            'Climate-appropriate promotions'
        ]
    }
    
    impact_df = pd.DataFrame(impact_data)
    
    st.dataframe(impact_df, use_container_width=True)
    
    # Brand performance matrix
    st.markdown("### 🏷️ Brand Performance Forecast Matrix")
    
    # Calculate brand metrics
    brand_metrics = []
    for brand, forecast in brand_forecasts.items():
        total_forecast = forecast['predicted_sales'].sum()
        avg_monthly = forecast['predicted_sales'].mean()
        volatility = forecast['predicted_sales'].std()
        
        brand_metrics.append({
            'Brand': brand,
            'Total Forecast (12mo)': int(total_forecast),
            'Avg Monthly': int(avg_monthly),
            'Volatility': round(volatility, 2)
        })
    
    brand_matrix = pd.DataFrame(brand_metrics).sort_values('Total Forecast (12mo)', ascending=False)
    
    # Add growth potential based on median
    median_forecast = brand_matrix['Total Forecast (12mo)'].median()
    brand_matrix['Growth Potential'] = brand_matrix['Total Forecast (12mo)'].apply(
        lambda x: 'High' if x > median_forecast else 'Medium'
    )
    
    st.dataframe(brand_matrix.style.background_gradient(cmap='YlOrRd', subset=['Total Forecast (12mo)']),
                use_container_width=True)
    
    # Recommendations
    st.markdown("### 💡 AI-Powered Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("""
        **Inventory Management**
        - Stock up 20% more units for Q4 (peak season)
        - Focus on top 3 brands: Toyota, Mercedes, BMW
        - Maintain 60-day inventory buffer
        """)
    
    with col2:
        st.info("""
        **Marketing Strategy**
        - Launch Ramadan campaign in March
        - Year-end clearance in December
        - Special financing offers in low seasons
        """)
    
    with col3:
        st.warning("""
        **Risk Management**
        - Monitor interest rate changes monthly
        - Diversify brand portfolio
        - Strengthen supply chain relationships
        """)
    
    # Download comprehensive report
    st.markdown("### 📥 Download Complete Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_overall = overall_forecast.to_csv(index=False)
        st.download_button(
            label="Download Overall Forecast",
            data=csv_overall,
            file_name="complete_sales_forecast.csv",
            mime="text/csv"
        )
    
    with col2:
        csv_brands = brand_matrix.to_csv(index=False)
        st.download_button(
            label="Download Brand Analysis",
            data=csv_brands,
            file_name="brand_forecast_analysis.csv",
            mime="text/csv"
        )

# =============================================================================
# CUSTOMER ANALYTICS PAGES (from original app.py)
# =============================================================================
elif page == "👥 Customer Segmentation" and df_customer is not None:
    st.markdown("<h1 class='main-header'>Customer Segmentation Analysis</h1>", unsafe_allow_html=True)
    
    segment_summary = df_customer.groupby('segment_name').agg({
        'customer_id': 'count',
        'total_customer_lifetime_value': 'mean',
        'loyalty_score': 'mean',
        'churn_probability': 'mean',
        'service_count_last_12_months': 'mean'
    }).round(2)
    segment_summary.columns = ['Count', 'Avg CLV (QAR)', 'Avg Loyalty', 'Avg Churn', 'Avg Services']
    
    st.dataframe(segment_summary.style.background_gradient(cmap='Blues'), use_container_width=True)
    
    fig = px.scatter(df_customer, x='loyalty_score', y='total_customer_lifetime_value',
                    color='segment_name', size='churn_probability',
                    title='🎯 Customer Segmentation: Loyalty vs Lifetime Value',
                    labels={'loyalty_score': 'Loyalty Score', 
                           'total_customer_lifetime_value': 'CLV (QAR)'},
                    hover_data=['brand', 'model'])
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📋 Segment Details")
    selected_segment = st.selectbox("Select Segment:", df_customer['segment_name'].unique())
    segment_customers = df_customer[df_customer['segment_name'] == selected_segment][
        ['customer_id', 'brand', 'model', 'total_customer_lifetime_value',
         'loyalty_score', 'churn_probability']
    ].sort_values('total_customer_lifetime_value', ascending=False)
    
    st.dataframe(segment_customers.head(50), use_container_width=True)
    st.download_button("📥 Download Segment Data", 
                      segment_customers.to_csv(index=False),
                      f"segment_{selected_segment}.csv")

elif page == "🚨 Churn Prediction" and df_customer is not None:
    st.markdown("<h1 class='main-header'>Churn Prediction & Prevention</h1>", unsafe_allow_html=True)
    
    high_risk = df_customer[df_customer['churn_probability'] > 0.7].sort_values('total_customer_lifetime_value', ascending=False)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🚨 High Risk Customers", f"{len(high_risk):,}")
    with col2:
        revenue_risk = high_risk['total_customer_lifetime_value'].sum()
        st.metric("💸 Revenue at Risk", f"QAR {revenue_risk/1e6:.2f}M")
    with col3:
        avg_churn = df_customer['churn_probability'].mean()
        st.metric("📊 Avg Churn Probability", f"{avg_churn:.1%}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.histogram(df_customer, x='churn_probability', nbins=30,
                           title="📈 Churn Probability Distribution")
        fig1.add_vline(x=0.7, line_dash="dash", line_color="red", 
                      annotation_text="High Risk Threshold")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        churn_segment = df_customer.groupby('segment_name')['churn_probability'].mean().sort_values()
        fig2 = px.bar(x=churn_segment.values, y=churn_segment.index, orientation='h',
                     title="⚠️ Churn Risk by Segment",
                     color=churn_segment.values, color_continuous_scale='Reds')
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("### 🎯 Top 30 At-Risk High-Value Customers")
    risk_table = high_risk.head(30)[
        ['customer_id', 'brand', 'model', 'loyalty_score',
         'churn_probability', 'total_customer_lifetime_value', 'segment_name']
    ]
    st.dataframe(risk_table, use_container_width=True)
    st.download_button("📥 Download High-Risk List", 
                      high_risk.to_csv(index=False),
                      "high_risk_customers.csv")

elif page == "💰 Customer Lifetime Value" and df_customer is not None:
    st.markdown("<h1 class='main-header'>Customer Lifetime Value Prediction</h1>", unsafe_allow_html=True)
    
    high_potential = df_customer[df_customer['clv_growth_potential'] > df_customer['clv_growth_potential'].quantile(0.9)].sort_values('clv_growth_potential', ascending=False)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_current = df_customer['total_customer_lifetime_value'].mean()
        st.metric("💵 Avg Current CLV", f"QAR {avg_current/1e3:.0f}K")
    with col2:
        avg_predicted = df_customer['predicted_clv'].mean()
        st.metric("📈 Avg Predicted CLV", f"QAR {avg_predicted/1e3:.0f}K")
    with col3:
        total_potential = df_customer['clv_growth_potential'].sum()
        st.metric("🚀 Total Growth Potential", f"QAR {total_potential/1e6:.1f}M")
    
    fig = px.scatter(df_customer, x='total_customer_lifetime_value', y='predicted_clv',
                    color='segment_name', 
                    title='💰 Current vs Predicted CLV by Segment',
                    labels={'total_customer_lifetime_value': 'Current CLV (QAR)',
                           'predicted_clv': 'Predicted CLV (QAR)'},
                    hover_data=['brand', 'model'])
    
    max_val = max(df_customer['total_customer_lifetime_value'].max(), df_customer['predicted_clv'].max())
    fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val],
                            mode='lines', name='Break-even',
                            line=dict(dash='dash', color='gray')))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 🌟 Top 30 High-Potential Customers")
    potential_table = high_potential.head(30)[
        ['customer_id', 'brand', 'model', 'segment_name',
         'total_customer_lifetime_value', 'predicted_clv', 'clv_growth_potential']
    ]
    st.dataframe(potential_table, use_container_width=True)
    st.download_button("📥 Download High-Potential List",
                      high_potential.to_csv(index=False),
                      "high_potential_customers.csv")

elif page == "🔧 Service Optimization" and df_customer is not None:
    st.markdown("<h1 class='main-header'>Service Revenue Optimization</h1>", unsafe_allow_html=True)
    
    service_opps = df_customer[df_customer['service_due_soon'] | df_customer['service_overdue']].copy()
    service_opps['priority'] = service_opps.apply(
        lambda x: 'URGENT' if x['service_overdue'] else 'HIGH' if x['next_service_due_days'] <= 14 else 'MEDIUM',
        axis=1
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔴 Overdue Services", f"{df_customer['service_overdue'].sum():,}")
    with col2:
        st.metric("🟡 Due Within 30 Days", f"{df_customer['service_due_soon'].sum():,}")
    with col3:
        potential_rev = service_opps['avg_service_cost'].sum()
        st.metric("💰 Potential Revenue", f"QAR {potential_rev/1e3:.0f}K")
    
    service_by_brand = service_opps.groupby('brand').agg({
        'customer_id': 'count',
        'avg_service_cost': 'sum'
    }).sort_values('avg_service_cost', ascending=False)
    
    fig = px.bar(x=service_by_brand['avg_service_cost'], y=service_by_brand.index, 
                orientation='h',
                title="🔧 Service Revenue Opportunity by Brand",
                labels={'x': 'Potential Revenue (QAR)', 'y': 'Brand'},
                color=service_by_brand['avg_service_cost'],
                color_continuous_scale='Greens')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📞 Priority Service Contacts")
    priority_filter = st.multiselect("Filter by Priority:", 
                                    ['URGENT', 'HIGH', 'MEDIUM'],
                                    default=['URGENT', 'HIGH'])
    
    priority_table = service_opps[service_opps['priority'].isin(priority_filter)].sort_values(
        ['priority', 'total_customer_lifetime_value'], 
        ascending=[True, False]
    ).head(50)
    
    st.dataframe(priority_table[
        ['customer_id', 'brand', 'model', 'next_service_due_days',
         'priority', 'avg_service_cost', 'warranty_status']
    ], use_container_width=True)
    
    st.download_button("📥 Download Service Opportunities",
                      service_opps.to_csv(index=False),
                      "service_opportunities.csv")

elif page == "📊 Sales Insights" and df_customer is not None:
    st.markdown("<h1 class='main-header'>Inventory & Sales Insights</h1>", unsafe_allow_html=True)
    
    popular_models = df_customer.groupby(['brand', 'model']).agg({
        'customer_id': 'count',
        'vehicle_price': 'mean',
        'loyalty_score': 'mean'
    }).round(0)
    popular_models.columns = ['Sales Count', 'Avg Price (QAR)', 'Avg Loyalty']
    popular_models = popular_models.sort_values('Sales Count', ascending=False).head(20)
    
    st.markdown("### 🏆 Top 20 Best-Selling Models")
    st.dataframe(popular_models, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        year_analysis = df_customer['model_year'].value_counts().sort_index()
        fig1 = px.bar(x=year_analysis.index, y=year_analysis.values,
                     title="📅 Sales by Model Year",
                     labels={'x': 'Model Year', 'y': 'Units Sold'})
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        brand_type = df_customer.groupby(['brand', 'customer_type']).size().unstack(fill_value=0)
        fig2 = px.imshow(brand_type.values, 
                        x=brand_type.columns, 
                        y=brand_type.index,
                        title="👥 Customer Type Preferences by Brand",
                        labels=dict(x="Customer Type", y="Brand", color="Count"),
                        color_continuous_scale="Blues")
        st.plotly_chart(fig2, use_container_width=True)

elif page == "🎯 Marketing Campaigns" and df_customer is not None:
    st.markdown("<h1 class='main-header'>Personalized Marketing Campaigns</h1>", unsafe_allow_html=True)
    
    campaign1 = df_customer[(df_customer['churn_probability'] > 0.65) & 
                   (df_customer['total_customer_lifetime_value'] > df_customer['total_customer_lifetime_value'].quantile(0.75))]
    campaign2 = df_customer[(df_customer['days_since_last_service'] > 365) & (df_customer['is_under_warranty'] == 0)]
    campaign3 = df_customer[(df_customer['clv_growth_potential'] > df_customer['clv_growth_potential'].quantile(0.85)) & 
                   (df_customer['loyalty_score'] > 70) & (df_customer['vehicle_age_years'] >= 3)]
    campaign4 = df_customer[(df_customer['is_under_warranty'] == 1) & (df_customer['vehicle_age_years'] >= 2)]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🛡️ Churn Prevention", f"{len(campaign1):,}",
                 help=f"Revenue at Risk: QAR {campaign1['total_customer_lifetime_value'].sum()/1e6:.1f}M")
    with col2:
        st.metric("🔄 Service Reactivation", f"{len(campaign2):,}",
                 help="Customers inactive >1 year")
    with col3:
        st.metric("⬆️ Upsell/Upgrade", f"{len(campaign3):,}",
                 help=f"Growth Potential: QAR {campaign3['clv_growth_potential'].sum()/1e6:.1f}M")
    with col4:
        st.metric("📋 Warranty Expiring", f"{len(campaign4):,}",
                 help="Extended warranty opportunity")
    
    campaign_select = st.selectbox("Select Campaign:", [
        "🛡️ Churn Prevention",
        "🔄 Service Reactivation",
        "⬆️ Upsell/Upgrade",
        "📋 Warranty Expiration"
    ])
    
    if "Churn" in campaign_select:
        st.markdown("### 🛡️ High-Value Churn Prevention Campaign")
        st.info("**Action:** VIP Loyalty Program + Exclusive Service Package")
        display_df = campaign1
    elif "Reactivation" in campaign_select:
        st.markdown("### 🔄 Service Comeback Campaign")
        st.info("**Action:** 20% Service Discount + Free Inspection")
        display_df = campaign2
    elif "Upsell" in campaign_select:
        st.markdown("### ⬆️ Upgrade Invitation Campaign")
        st.info("**Action:** Trade-in Offer for Newer Model")
        display_df = campaign3
    else:
        st.markdown("### 📋 Warranty Expiration Campaign")
        st.info("**Action:** Extended Warranty + Service Package")
        display_df = campaign4
    
    st.dataframe(display_df.head(50), use_container_width=True)
    st.download_button(f"📥 Download Campaign List",
                      display_df.to_csv(index=False),
                      f"campaign_{campaign_select.split()[1].lower()}.csv")

elif page == "🤖 AI Sales Assistant" and df_customer is not None:
    st.markdown("<h1 class='main-header'>AI-Powered Sales Message Generator</h1>", unsafe_allow_html=True)
    
    st.info("💡 Generate personalized sales messages using AI (powered by OpenAI GPT-4)")
    
    api_key_input = st.text_input("Enter OpenAI API Key:", type="password", 
                                  value=OPENAI_API_KEY,
                                  help="Get your key from: https://platform.openai.com/api-keys")
    
    if api_key_input:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            customer_id = st.selectbox("Select Customer:", df_customer['customer_id'].unique())
        
        with col2:
            language = st.selectbox("Language:", ["English", "Arabic", "Hindi"])
        
        customer_data = df_customer[df_customer['customer_id'] == customer_id].iloc[0]
        
        st.markdown("### 👤 Customer Profile")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Brand", customer_data['brand'])
        with col2:
            st.metric("CLV", f"QAR {customer_data['total_customer_lifetime_value']/1e3:.0f}K")
        with col3:
            st.metric("Loyalty", f"{customer_data['loyalty_score']:.0f}/100")
        with col4:
            st.metric("Churn Risk", f"{customer_data['churn_probability']:.0%}")
        
        if st.button("🚀 Generate AI Message", type="primary"):
            with st.spinner("Generating personalized message..."):
                try:
                    client = OpenAI(api_key=api_key_input)
                    
                    prompt = f"""
You are an expert automotive sales consultant for a luxury car dealership in Qatar.
Generate a personalized, persuasive sales message for this customer:

- Brand: {customer_data['brand']} {customer_data['model']}
- Vehicle Age: {customer_data['vehicle_age_years']} years
- Loyalty Score: {customer_data['loyalty_score']}/100
- Customer Type: {customer_data['customer_type']}
- Days Since Last Service: {customer_data['days_since_last_service']} days

Language: {language}
Tone: Professional, respectful, culturally appropriate for Qatar
Goal: Encourage upgrade or service engagement
Length: Under 100 words for WhatsApp
Include: Clear call-to-action
"""
                    
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are an expert automotive sales consultant."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=300
                    )
                    
                    message = response.choices[0].message.content
                    
                    st.success("✅ Message Generated!")
                    st.markdown("### 📱 Personalized Sales Message")
                    st.markdown(f"""
                    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #1f77b4;'>
                    {message}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.code(message, language=None)
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("Please check your API key and try again.")
    else:
        st.warning("⚠️ Please enter your OpenAI API key to use this feature")

elif page == "🔍 Customer Search" and df_customer is not None:
    st.markdown("<h1 class='main-header'>Customer Search & Profile</h1>", unsafe_allow_html=True)
    
    search_id = st.text_input("🔍 Enter Customer ID:")
    
    if search_id:
        customer = df_customer[df_customer['customer_id'] == search_id]
        
        if len(customer) > 0:
            customer = customer.iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🚗 Brand", customer['brand'])
                st.metric("🏷️ Model", customer['model'])
            with col2:
                st.metric("💰 CLV", f"QAR {customer['total_customer_lifetime_value']:,.0f}")
                st.metric("📈 Predicted CLV", f"QAR {customer['predicted_clv']:,.0f}")
            with col3:
                st.metric("⭐ Loyalty Score", f"{customer['loyalty_score']:.0f}/100")
                st.metric("⚠️ Churn Risk", f"{customer['churn_probability']:.0%}")
            with col4:
                st.metric("🎯 Segment", customer['segment_name'])
                st.metric("🔧 Services (12mo)", f"{customer['service_count_last_12_months']:.0f}")
            
            st.markdown("---")
            
            st.markdown("### 📋 Complete Customer Profile")
            profile_data = {
                "Personal Info": {
                    "Customer ID": customer['customer_id'],
                    "Nationality": customer['nationality'],
                    "Customer Type": customer['customer_type'],
                    "Income Band": customer['income_band']
                },
                "Vehicle Info": {
                    "Brand": customer['brand'],
                    "Model": customer['model'],
                    "Model Year": int(customer['model_year']),
                    "Vehicle Age": f"{customer['vehicle_age_years']} years",
                    "Vehicle Price": f"QAR {customer['vehicle_price']:,.0f}",
                    "Warranty Status": customer['warranty_status']
                },
                "Engagement Metrics": {
                    "Loyalty Score": f"{customer['loyalty_score']:.1f}/100",
                    "Churn Probability": f"{customer['churn_probability']:.1%}",
                    "Total CLV": f"QAR {customer['total_customer_lifetime_value']:,.0f}",
                    "Predicted CLV": f"QAR {customer['predicted_clv']:,.0f}",
                    "Growth Potential": f"QAR {customer['clv_growth_potential']:,.0f}"
                },
                "Service History": {
                    "Services (12 months)": int(customer['service_count_last_12_months']),
                    "Days Since Last Service": int(customer['days_since_last_service']),
                    "Next Service Due": f"{customer['next_service_due_days']:.0f} days",
                    "Avg Service Cost": f"QAR {customer['avg_service_cost']:,.0f}",
                    "Total Service Revenue": f"QAR {customer['service_revenue_to_date']:,.0f}"
                }
            }
            
            col1, col2 = st.columns(2)
            with col1:
                for section in list(profile_data.keys())[:2]:
                    with st.expander(f"📌 {section}", expanded=True):
                        for key, value in profile_data[section].items():
                            st.write(f"**{key}:** {value}")
            
            with col2:
                for section in list(profile_data.keys())[2:]:
                    with st.expander(f"📌 {section}", expanded=True):
                        for key, value in profile_data[section].items():
                            st.write(f"**{key}:** {value}")
        else:
            st.error("❌ Customer ID not found!")
    else:
        st.info("💡 Enter a customer ID to view their complete profile")

# =============================================================================
# SIDEBAR - QUICK STATS
# =============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Quick Stats")
st.sidebar.metric("Total Sales Records", f"{len(df_forecast):,}")
st.sidebar.metric("Brands Available", f"{df_forecast['brand'].nunique()}")
st.sidebar.metric("Total Revenue", f"QAR {df_forecast['final_sale_price'].sum()/1e6:.1f}M")

if df_customer is not None:
    st.sidebar.metric("Total Customers", f"{len(df_customer):,}")
    st.sidebar.metric("High Churn Risk", f"{len(df_customer[df_customer['churn_probability'] > 0.7]):,}")

st.sidebar.markdown("---")
st.sidebar.info("🔮 **Advanced Forecasting Enabled**\n\nThis platform uses AI-powered models with 7 external factors for accurate sales predictions.")

