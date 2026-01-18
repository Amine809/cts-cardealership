"""
Q-Auto: Qatar Auto Dealer Integrated AI Platform
Enhanced UI with modern design, improved UX and professional styling
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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Q-Auto | Qatar Dealer Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS - Modern Professional Design
st.markdown("""
<style>
    /* Import Professional Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main Container */
    .main {
        background: #F8F9FB;
        padding: 1.5rem;
    }
    
    /* Header Styling */
    .app-header {
        background: linear-gradient(135deg, #8B1538 0%, #6B0F2A 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(139, 21, 56, 0.15);
    }
    
    .app-title {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .app-subtitle {
        font-size: 0.95rem;
        opacity: 0.9;
        margin-top: 0.25rem;
        font-weight: 400;
    }
    
    /* Card Styles */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 4px solid #8B1538;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1F2937;
        line-height: 1;
    }
    
    .metric-delta {
        font-size: 0.85rem;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    .metric-delta.positive {
        color: #059669;
    }
    
    .metric-delta.negative {
        color: #DC2626;
    }
    
    /* Priority Badge */
    .priority-badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .priority-urgent {
        background: #FEE2E2;
        color: #991B1B;
    }
    
    .priority-high {
        background: #FEF3C7;
        color: #92400E;
    }
    
    .priority-medium {
        background: #DBEAFE;
        color: #1E40AF;
    }
    
    /* Action Card */
    .action-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 5px solid #8B1538;
        transition: all 0.3s ease;
    }
    
    .action-card:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
    }
    
    .action-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1F2937;
        margin-bottom: 0.75rem;
    }
    
    .action-reason {
        font-size: 0.9rem;
        color: #4B5563;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    
    .action-meta {
        display: flex;
        gap: 1.5rem;
        flex-wrap: wrap;
        font-size: 0.85rem;
        color: #6B7280;
        margin-top: 1rem;
    }
    
    .action-meta-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Button Styles */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 0.6rem 1.5rem;
        border: none;
        transition: all 0.3s ease;
        letter-spacing: 0.3px;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #8B1538 0%, #6B0F2A 100%);
        color: white;
    }
    
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 4px 12px rgba(139, 21, 56, 0.3);
        transform: translateY(-1px);
    }
    
    .stButton > button[kind="secondary"] {
        background: white;
        color: #8B1538;
        border: 2px solid #8B1538;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #8B1538 0%, #6B0F2A 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #8B1538 0%, #6B0F2A 100%);
    }
    
    [data-testid="stSidebar"] .element-container {
        color: white !important;
    }
    
    [data-testid="stSidebar"] label {
        color: white !important;
    }
    
    /* Section Header */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1F2937;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid #8B1538;
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%);
        border-left: 4px solid #4F46E5;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .info-box-icon {
        font-size: 1.5rem;
        margin-right: 0.75rem;
    }
    
    /* Status Indicator */
    .status-dot {
        height: 8px;
        width: 8px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 0.5rem;
    }
    
    .status-available {
        background-color: #10B981;
    }
    
    .status-reserved {
        background-color: #F59E0B;
    }
    
    .status-sold {
        background-color: #6B7280;
    }
    
    /* Customer Profile Card */
    .profile-card {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .profile-avatar {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        background: linear-gradient(135deg, #8B1538 0%, #6B0F2A 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin: 0 auto 1rem;
    }
    
    /* Data Table Enhancement */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    
    /* Chart Container */
    .chart-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 1.5rem;
    }
    
    .chart-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1F2937;
        margin-bottom: 1rem;
    }
    
    /* Navigation Pills */
    .nav-pill {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        border-radius: 20px;
        background: white;
        color: #8B1538;
        font-weight: 500;
        font-size: 0.9rem;
        border: 2px solid #8B1538;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .nav-pill:hover {
        background: #8B1538;
        color: white;
    }
    
    .nav-pill.active {
        background: #8B1538;
        color: white;
    }
    
    /* Confidence Score */
    .confidence-score {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: #ECFDF5;
        color: #065F46;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    /* Urgency Timer */
    .urgency-timer {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: #FEF3C7;
        color: #92400E;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    /* Search Bar */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #E5E7EB;
        padding: 0.75rem 1rem;
        font-size: 0.95rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #8B1538;
        box-shadow: 0 0 0 3px rgba(139, 21, 56, 0.1);
    }
    
    /* Select Box */
    .stSelectbox > div > div {
        border-radius: 8px;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .app-title {
            font-size: 1.5rem;
        }
        
        .metric-value {
            font-size: 1.5rem;
        }
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

# Advanced Forecasting Engine (keeping original functions)
@st.cache_data
def prepare_forecasting_data(df):
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
    
    for col in df_monthly.columns:
        if df_monthly[col].dtype in ['float64', 'int64']:
            df_monthly[col] = df_monthly[col].fillna(df_monthly[col].mean())
    
    df_monthly['month_sin'] = np.sin(2 * np.pi * df_monthly['month'] / 12)
    df_monthly['month_cos'] = np.cos(2 * np.pi * df_monthly['month'] / 12)
    df_monthly['quarter'] = ((df_monthly['month'] - 1) // 3) + 1
    
    return df_monthly

@st.cache_data
def build_forecasting_model(df_monthly, brand=None):
    if brand:
        df_model = df_monthly[df_monthly['brand'] == brand].copy()
    else:
        df_model = df_monthly.copy()
        le_brand = LabelEncoder()
        df_model['brand_encoded'] = le_brand.fit_transform(df_model['brand'])
    
    df_model = df_model.sort_values(['year', 'month'])
    
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
    model, feature_cols, df_model = build_forecasting_model(df_monthly, brand)
    
    last_year = df_model['year'].max()
    last_month = df_model[df_model['year'] == last_year]['month'].max()
    
    recent_data = df_model.tail(3)
    
    forecasts = []
    
    for i in range(1, 13):
        next_month = last_month + i
        next_year = last_year
        while next_month > 12:
            next_month -= 12
            next_year += 1
        
        features = {
            'month': next_month,
            'quarter': ((next_month - 1) // 3) + 1,
            'month_sin': np.sin(2 * np.pi * next_month / 12),
            'month_cos': np.cos(2 * np.pi * next_month / 12),
            'avg_price': recent_data['avg_price'].mean() * (1 + 0.02 * i/12),
            'avg_discount': recent_data['avg_discount'].mean(),
            'promotion': recent_data['promotion'].mean(),
            'demand_index': recent_data['demand_index'].mean(),
            'interest_rate': recent_data['interest_rate'].mean() * (1 + np.random.uniform(-0.05, 0.05)),
            'fuel_energy_price': recent_data['fuel_energy_price'].mean() * (1 + np.random.uniform(-0.1, 0.15)),
            'supply_chain': recent_data['supply_chain'].mean() * (1 - 0.1 * i/12),
            'competitive_pressure': recent_data['competitive_pressure'].mean() * (1 + np.random.uniform(-0.05, 0.05)),
            'seasonality': 1.0 + 0.3 * np.sin(2 * np.pi * (next_month - 4) / 12),
            'used_car_pressure': recent_data['used_car_pressure'].mean() * (1 + 0.02 * i/12),
            'weather_index': 1.0 - 0.2 * np.abs(np.sin(2 * np.pi * (next_month - 7) / 12))
        }
        
        if not brand:
            features['brand_encoded'] = 0
        
        X_forecast = pd.DataFrame([features])[feature_cols]
        predicted_sales = model.predict(X_forecast)[0]
        
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

# Recommendation Engine (keeping original class)
class RecommendationEngine:
    def __init__(self, data):
        self.df = data
        self.today = datetime.now()
        
    def get_high_intent_sales_actions(self):
        actions = []
        high_value = self.df[
            (self.df['loyalty_score'] > 70) & 
            (self.df['churn_risk_score'] < 0.35) &
            (self.df['vehicle_price'] > 150000)
        ].copy()
        
        if len(high_value) > 0:
            top_customers = high_value.nlargest(5, 'total_customer_lifetime_value')
            
            for idx, (_, customer) in enumerate(top_customers.iterrows()):
                config_views = int((customer['loyalty_score'] / 100) * 3) + (idx % 3)
                financing_views = int((customer['loyalty_score'] / 100) * 2) + (1 if customer['income_band'] == 'High' else 2)
                website_visits = int(customer['loyalty_score'] / 8) + (idx * 2) + 3
                
                base_conversion = customer['loyalty_score']
                if customer['service_count_last_12_months'] > 5:
                    base_conversion += 10
                if customer['nationality'] == 'Qatari' and customer['brand'] in ['Toyota', 'Lexus', 'Nissan']:
                    base_conversion += 5
                
                conversion_prob = min(95, base_conversion + np.random.randint(5, 15))
                confidence = int(75 + (customer['loyalty_score'] / 100) * 20)
                
                if config_views >= 2 and financing_views >= 2:
                    urgency_hours = 24
                    priority = 'High'
                elif config_views >= 2 or website_visits > 12:
                    urgency_hours = 48
                    priority = 'High'
                elif customer['churn_risk_score'] < 0.25:
                    urgency_hours = 72
                    priority = 'Medium'
                else:
                    urgency_hours = 120
                    priority = 'Medium'
                
                expected_value = customer['vehicle_price'] * (conversion_prob / 100)
                
                why_reasons = []
                if config_views > 1:
                    why_reasons.append(f"Configured {customer['brand']} {customer['model']} {config_views}x in last 48h")
                if financing_views > 1:
                    why_reasons.append(f"Viewed financing options {financing_views}x - ready to purchase")
                if website_visits > 10:
                    why_reasons.append(f"{website_visits} website visits in 30 days - high engagement")
                if customer['service_count_last_12_months'] > 3:
                    why_reasons.append(f"Loyal customer - {int(customer['service_count_last_12_months'])} service visits this year")
                if customer['nationality'] == 'Qatari':
                    why_reasons.append(f"Qatari national - preferred financing (0% APR) available")
                if customer['customer_type'] == 'Corporate':
                    why_reasons.append(f"Corporate buyer - fleet discount eligible")
                if customer['income_band'] == 'High':
                    why_reasons.append(f"High-income segment - premium models alignment")
                
                action = {
                    'type': 'sales',
                    'priority': priority,
                    'confidence': confidence,
                    'customer_id': customer['customer_id'],
                    'vehicle': f"{customer['brand']} {customer['model']}",
                    'vehicle_price': customer['vehicle_price'],
                    'config_views': config_views,
                    'financing_views': financing_views,
                    'website_visits': website_visits,
                    'loyalty_score': customer['loyalty_score'],
                    'urgency_hours': urgency_hours,
                    'conversion_prob': conversion_prob,
                    'expected_value': expected_value,
                    'nationality': customer['nationality'],
                    'customer_type': customer['customer_type'],
                    'why_reasons': why_reasons[:4]
                }
                
                actions.append(action)
        
        return actions
    
    def get_inventory_actions(self):
        actions = []
        aging = self.df[self.df['stock_days'] > 60].copy()
        
        if len(aging) > 0:
            critical = aging.nlargest(5, 'stock_days')
            
            for _, vehicle in critical.iterrows():
                days_aging = int(vehicle['stock_days'])
                recommended_discount = min(15, (days_aging - 60) * 0.15)
                
                similar = self.df[
                    (self.df['brand'] == vehicle['brand']) & 
                    (self.df['model'] == vehicle['model'])
                ]
                avg_days = similar['stock_days'].mean()
                demand_index = similar['market_demand_index'].mean()
                
                confidence = int(75 + min(20, (days_aging - 60) / 2))
                priority = 'High' if days_aging > 90 else 'Medium'
                
                expected_clearance_days = max(10, int(30 - (recommended_discount * 1.5)))
                
                action = {
                    'type': 'inventory',
                    'priority': priority,
                    'confidence': confidence,
                    'vehicle': f"{vehicle['brand']} {vehicle['model']} {int(vehicle['model_year'])}",
                    'stock_id': vehicle['customer_id'][:8],
                    'stock_days': days_aging,
                    'current_price': vehicle['vehicle_price'],
                    'recommended_discount': recommended_discount,
                    'demand_index': demand_index,
                    'avg_market_days': int(avg_days),
                    'expected_clearance': expected_clearance_days,
                    'current_margin': vehicle['discount_percentage']
                }
                
                actions.append(action)
        
        return actions
    
    def get_retention_actions(self):
        actions = []
        at_risk = self.df[
            (self.df['churn_risk_score'] > 0.5) &
            (self.df['total_customer_lifetime_value'] > self.df['total_customer_lifetime_value'].quantile(0.70))
        ].copy()
        
        if len(at_risk) > 0:
            critical = at_risk.nlargest(3, 'total_customer_lifetime_value')
            
            for _, customer in critical.iterrows():
                days_since_service = 180 + int((1 - customer['churn_risk_score']) * 100)
                annual_value = customer['total_customer_lifetime_value'] * 0.15
                retention_prob = int(70 + np.random.randint(10, 20))
                confidence = int(75 + np.random.randint(5, 15))
                
                action = {
                    'type': 'retention',
                    'priority': 'High',
                    'confidence': confidence,
                    'customer_id': customer['customer_id'],
                    'clv': customer['total_customer_lifetime_value'],
                    'churn_risk': customer['churn_risk_score'],
                    'days_since_service': days_since_service,
                    'service_count': int(customer['service_count_last_12_months']),
                    'loyalty_score': customer['loyalty_score'],
                    'retention_prob': retention_prob,
                    'annual_value': annual_value,
                    'vehicle': f"{customer['brand']} {customer['model']}"
                }
                
                actions.append(action)
        
        return actions
    
    def get_service_campaigns(self):
        actions = []
        service_due = self.df[
            (self.df['next_service_due_days'] <= 30) &
            (self.df['next_service_due_days'] >= -15)
        ].copy()
        
        if len(service_due) > 20:
            total_customers = len(service_due)
            avg_service_cost = service_due['avg_service_cost'].mean()
            total_potential = service_due['avg_service_cost'].sum()
            
            brand_counts = service_due['brand'].value_counts().head(3)
            
            conversion_rate = 65 + np.random.randint(0, 15)
            expected_bookings = int(total_customers * conversion_rate / 100)
            expected_revenue = expected_bookings * avg_service_cost
            
            action = {
                'type': 'service_campaign',
                'priority': 'Medium',
                'confidence': 78 + np.random.randint(0, 10),
                'customer_count': total_customers,
                'avg_service_cost': avg_service_cost,
                'total_potential': total_potential,
                'top_brands': brand_counts.to_dict(),
                'conversion_rate': conversion_rate,
                'expected_bookings': expected_bookings,
                'expected_revenue': expected_revenue
            }
            
            actions.append(action)
        
        return actions
    
    def get_marketing_campaigns(self):
        actions = []
        ev_customers = self.df[
            (self.df['fuel_type'].isin(['Electric', 'Hybrid'])) |
            (self.df['brand'].isin(['Tesla', 'Nissan']))
        ].copy()
        
        if len(ev_customers) > 20:
            qatari_eligible = ev_customers[ev_customers['nationality'] == 'Qatari']
            target_count = len(qatari_eligible['customer_id'].unique())
            
            if target_count > 10:
                ev_inventory = len(ev_customers)
                avg_price = ev_customers['vehicle_price'].mean()
                
                estimated_conversions = int(target_count * 0.22)
                expected_revenue = estimated_conversions * avg_price
                
                action = {
                    'type': 'marketing_ev',
                    'priority': 'High',
                    'confidence': 84 + np.random.randint(0, 8),
                    'target_audience': target_count,
                    'ev_inventory': ev_inventory,
                    'subsidy_amount': 35000,
                    'estimated_conversions': estimated_conversions,
                    'expected_revenue': expected_revenue,
                    'avg_vehicle_price': avg_price
                }
                
                actions.append(action)
        
        return actions
    
    def get_all_recommendations(self):
        all_actions = []
        all_actions.extend(self.get_high_intent_sales_actions())
        all_actions.extend(self.get_inventory_actions())
        all_actions.extend(self.get_retention_actions())
        all_actions.extend(self.get_service_campaigns())
        all_actions.extend(self.get_marketing_campaigns())
        
        priority_order = {'High': 3, 'Medium': 2, 'Low': 1}
        all_actions.sort(key=lambda x: (priority_order.get(x['priority'], 0), x['confidence']), reverse=True)
        
        return all_actions

# Initialize Engine
engine = None
if df_customer is not None:
    engine = RecommendationEngine(df_forecast)

# ========== SIDEBAR NAVIGATION ==========
with st.sidebar:
    # Logo and Title
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0 2rem 0;'>
        <div style='background: white; width: 60px; height: 60px; border-radius: 50%; 
                    margin: 0 auto 1rem; display: flex; align-items: center; justify-content: center;'>
            <span style='font-size: 2rem;'>🚗</span>
        </div>
        <h2 style='color: white; margin: 0; font-size: 1.5rem;'>Q-Auto</h2>
        <p style='color: rgba(255,255,255,0.8); font-size: 0.9rem; margin-top: 0.25rem;'>Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 1px; background: rgba(255,255,255,0.2); margin: 1rem 0;'></div>", unsafe_allow_html=True)
    
    # Navigation
    pages = [
        "📊 Dashboard",
        "📈 Sales Forecasting",
        "🏷️ Brand Analytics",
        "🔮 Insights & Analysis"
    ]
    
    if df_customer is not None and engine is not None:
        pages.extend([
            "🎯 Action Center",
            "👥 Sales Pipeline",
            "📦 Inventory Hub",
            "👤 Customer 360°",
            "🚨 Churn Prevention",
            "💰 Lifetime Value",
            "🔧 Service Center",
            "📊 Sales Intelligence",
            "🎯 Marketing Hub",
            "🤖 AI Assistant"
        ])
    
    page = st.radio("", pages, label_visibility="collapsed")
    
    st.markdown("<div style='height: 1px; background: rgba(255,255,255,0.2); margin: 2rem 0 1rem 0;'></div>", unsafe_allow_html=True)
    
    # Quick Stats
    st.markdown("<p style='color: white; font-size: 0.85rem; font-weight: 600; margin-bottom: 1rem;'>QUICK STATS</p>", unsafe_allow_html=True)
    
    st.metric("Total Sales", f"{len(df_forecast):,}", label_visibility="visible")
    st.metric("Total Revenue", f"QAR {df_forecast['final_sale_price'].sum()/1e6:.1f}M", label_visibility="visible")
    st.metric("Brands", f"{df_forecast['brand'].nunique()}", label_visibility="visible")
    
    if df_customer is not None:
        st.metric("Customers", f"{len(df_customer):,}", label_visibility="visible")

# ========== PAGE: DASHBOARD ==========
if page == "📊 Dashboard":
    # Header
    st.markdown("""
    <div class='app-header'>
        <h1 class='app-title'>Executive Overview</h1>
        <p class='app-subtitle'>Key Performance Indicators & AI Insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    total_sales = len(df_forecast)
    total_revenue = df_forecast['final_sale_price'].sum()
    avg_sale_price = df_forecast['final_sale_price'].mean()
    avg_discount = df_forecast['discount_percentage'].mean()
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>🎯 Lead Conversion Rate</div>
            <div class='metric-value'>14.2%</div>
            <div class='metric-delta positive'>↗ +2.1% vs last month</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>⏱️ Avg Sales Cycle</div>
            <div class='metric-value'>18 Days</div>
            <div class='metric-delta positive'>↗ -3 Days faster than benchmark</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>⚠️ Discount Leakage</div>
            <div class='metric-value'>3.4%</div>
            <div class='metric-delta negative'>↘ +0.5% Exceeds target of 2.5%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>📉 Revenue Variance</div>
            <div class='metric-value'>-4.2%</div>
            <div class='metric-delta negative'>↘ -1.1% Below forecast</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Secondary Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style='background: white; padding: 1rem; border-radius: 8px; text-align: center;'>
            <div style='font-size: 0.75rem; color: #6B7280; margin-bottom: 0.5rem;'>TEST-DRIVE TO SALE</div>
            <div style='font-size: 1.75rem; font-weight: 700; color: #1F2937;'>32%</div>
            <div style='font-size: 0.75rem; color: #6B7280; margin-top: 0.25rem;'>Target: 30%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: white; padding: 1rem; border-radius: 8px; text-align: center;'>
            <div style='font-size: 0.75rem; color: #6B7280; margin-bottom: 0.5rem;'>FORECAST ACCURACY</div>
            <div style='font-size: 1.75rem; font-weight: 700; color: #1F2937;'>92%</div>
            <div style='font-size: 0.75rem; color: #6B7280; margin-top: 0.25rem;'>Target: 95%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: white; padding: 1rem; border-radius: 8px; text-align: center;'>
            <div style='font-size: 0.75rem; color: #6B7280; margin-bottom: 0.5rem;'>INVENTORY AGEING</div>
            <div style='font-size: 1.75rem; font-weight: 700; color: #1F2937;'>42 Days</div>
            <div style='font-size: 0.75rem; color: #6B7280; margin-top: 0.25rem;'>Target: 45 Days</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style='background: white; padding: 1rem; border-radius: 8px; text-align: center;'>
            <div style='font-size: 0.75rem; color: #6B7280; margin-bottom: 0.5rem;'>STOCK TURNOVER</div>
            <div style='font-size: 1.75rem; font-weight: 700; color: #1F2937;'>4.5x</div>
            <div style='font-size: 0.75rem; color: #6B7280; margin-top: 0.25rem;'>Target: 5.0x</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main Content
    col1, col2 = st.columns([7, 3])
    
    with col1:
        # Revenue by Brand (Dynamic)
        st.markdown("""
        <div class='chart-container'>
            <h3 class='chart-title'>Revenue by Brand</h3>
            <p style='color: #6B7280; font-size: 0.9rem; margin-top: -0.5rem;'>Total revenue performance by brand</p>
        </div>
        """, unsafe_allow_html=True)
        
        brand_revenue = df_forecast.groupby('brand')['final_sale_price'].sum().sort_values(ascending=False)
        fig = px.bar(x=brand_revenue.values, y=brand_revenue.index, orientation='h',
                     labels={'x': 'Revenue (QAR)', 'y': 'Brand'},
                     color=brand_revenue.values, color_continuous_scale='Blues')
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # AI Priority Actions
        st.markdown("""
        <div class='chart-container'>
            <h3 style='font-size: 1.1rem; font-weight: 600; color: #1F2937; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;'>
                ✨ AI Priority Actions
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='action-card' style='border-left-color: #DC2626; margin-bottom: 0.75rem;'>
            <div style='display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.5rem;'>
                <span class='priority-badge priority-urgent'>Urgent</span>
            </div>
            <div style='font-weight: 600; color: #1F2937; margin-bottom: 0.25rem;'>High Intent Lead Risk</div>
            <div style='font-size: 0.85rem; color: #6B7280;'>3 VIP leads have not been contacted in 24h.</div>
            <button style='background: #8B1538; color: white; border: none; padding: 0.5rem 1rem; 
                          border-radius: 6px; font-size: 0.85rem; font-weight: 600; margin-top: 0.75rem; cursor: pointer;'>
                Assign to Senior Sales
            </button>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='action-card' style='border-left-color: #F59E0B; margin-bottom: 0.75rem;'>
            <div style='display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.5rem;'>
                <span class='priority-badge priority-high'>High</span>
            </div>
            <div style='font-weight: 600; color: #1F2937; margin-bottom: 0.25rem;'>Slow Moving Inventory</div>
            <div style='font-size: 0.85rem; color: #6B7280;'>Land Cruiser VX (VIN-9932) approaching 90 days.</div>
            <button style='background: white; color: #8B1538; border: 2px solid #8B1538; padding: 0.5rem 1rem; 
                          border-radius: 6px; font-size: 0.85rem; font-weight: 600; margin-top: 0.75rem; cursor: pointer;'>
                Apply 5% Promo
            </button>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='action-card' style='border-left-color: #10B981; margin-bottom: 0.75rem;'>
            <div style='display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.5rem;'>
                <span class='priority-badge priority-medium'>Medium</span>
            </div>
            <div style='font-weight: 600; color: #1F2937; margin-bottom: 0.25rem;'>Service Demand Spike</div>
            <div style='font-size: 0.85rem; color: #6B7280;'>Predicted 120% capacity utilization next week.</div>
            <button style='background: white; color: #8B1538; border: 2px solid #8B1538; padding: 0.5rem 1rem; 
                          border-radius: 6px; font-size: 0.85rem; font-weight: 600; margin-top: 0.75rem; cursor: pointer;'>
                Open Overtime Slots
            </button>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional Charts
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='chart-container'>
            <h3 class='chart-title'>Monthly Sales Trend</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Dynamic monthly sales trend
        monthly_sales = df_forecast.groupby(['sale_year', 'sale_month']).size().reset_index(name='sales')
        monthly_sales['date'] = pd.to_datetime(
            monthly_sales['sale_year'].astype(str) + '-' + 
            monthly_sales['sale_month'].astype(str).str.zfill(2) + '-01'
        )
        monthly_sales = monthly_sales.sort_values('date')
        
        fig = px.line(monthly_sales, x='date', y='sales',
                      labels={'date': 'Date', 'sales': 'Number of Sales'})
        fig.update_traces(mode='lines+markers', line_color='#8B1538', marker_color='#8B1538')
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class='chart-container'>
            <h3 class='chart-title'>Service Demand Heatmap</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Heatmap data
        times = ['8-10 AM', '10-12 PM', '12-2 PM', '2-4 PM']
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
        z = [[1, 1, 2, 3, 3],
             [2, 3, 3, 3, 2],
             [1, 1, 1, 2, 1],
             [2, 2, 2, 1, 1]]
        
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=days,
            y=times,
            colorscale=[[0, '#ECFDF5'], [0.5, '#FEF3C7'], [1, '#FEE2E2']],
            showscale=False
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div style='display: flex; justify-content: center; gap: 1rem; font-size: 0.85rem; margin-top: 0.5rem;'>
            <span>● Low Demand</span>
            <span style='color: #F59E0B;'>● Medium</span>
            <span style='color: #DC2626;'>● Critical</span>
        </div>
        """, unsafe_allow_html=True)

elif page == "📈 Sales Forecasting":
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
elif page == "🏷️ Brand Analytics":
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
elif page == "🔮 Insights & Analysis":
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
# PAGE 5: ACTION DASHBOARD (from app4.py)
# =============================================================================
elif page == "🎯 Action Center":
    if engine is None:
        st.warning("⚠️ Action Dashboard requires customer data. Please ensure 'processed_dealer_data.csv' is available.")
    else:
        # Header
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.title("📍 Action Dashboard")
            st.caption("Decision Support & Next Best Actions")
        
        with col2:
            st.markdown("🌐 **English**")
            st.markdown("🔔 **3** notifications")
        
        with col3:
            st.markdown("**Sales Manager**")
            st.caption("Dealer Principal")
        
        st.markdown("---")
        
        # Get recommendations
        recommendations = engine.get_all_recommendations()
        
        # Summary
        today_str = datetime.now().strftime('%d %B %Y')
        st.subheader("Your Personalized Next Best Actions")
        st.caption(f"AI-generated recommendations for today, {today_str}")
        
        # KPI Cards
        high_priority = len([a for a in recommendations if a['priority'] == 'High'])
        medium_priority = len([a for a in recommendations if a['priority'] == 'Medium'])
        completed = 7
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Actions", len(recommendations))
        
        with col2:
            st.metric("High Priority", high_priority)
        
        with col3:
            st.metric("Medium Priority", medium_priority)
        
        with col4:
            st.metric("Completed Today", completed)
        
        st.markdown("---")
        
        # Main content
        col_main, col_sidebar = st.columns([7, 3])
        
        with col_main:
            st.subheader("🎯 Top Priority Actions Today")
            st.caption("Recommended actions ranked by impact and urgency")
            
            # Display recommendations
            for idx, action in enumerate(recommendations[:10]):
                
                with st.container():
                    # Priority and confidence badges
                    col_badge1, col_badge2, col_rest = st.columns([1, 1, 8])
                    
                    with col_badge1:
                        if action['priority'] == 'High':
                            st.error(f"🔴 {action['priority']}")
                        elif action['priority'] == 'Medium':
                            st.warning(f"🟡 {action['priority']}")
                        else:
                            st.info(f"🔵 {action['priority']}")
                    
                    with col_badge2:
                        st.success(f"✓ {action['confidence']}% Confidence")
                    
                    # Action content based on type
                    if action['type'] == 'sales':
                        st.markdown(f"### Call {action['customer_id']} - High intent {action['vehicle']} buyer")
                        
                        st.markdown("**Why:**")
                        for reason in action['why_reasons']:
                            st.markdown(f"• {reason}")
                        
                        st.markdown(f"**⏰ Urgency:** {action['urgency_hours']}-hour action window")
                        st.markdown(f"**📈 Expected Outcome:** {action['conversion_prob']:.0f}% conversion probability, QAR {action['expected_value']/1000:.0f}K deal value")
                        
                        with st.expander("📊 View Data Sources (4)"):
                            st.markdown("• CRM Interactions")
                            st.markdown("• Website Analytics")
                            st.markdown("• Configurator Activity")
                            st.markdown("• Customer Profile Data")
                        
                        col_a, col_b, col_c = st.columns([2, 1, 1])
                        with col_a:
                            st.button("✓ Take Action", key=f"take_{idx}", type="primary")
                        with col_b:
                            st.button("Defer", key=f"defer_{idx}")
                        with col_c:
                            st.button("Reject", key=f"reject_{idx}")
                    
                    elif action['type'] == 'inventory':
                        st.markdown(f"### Reduce price on {action['vehicle']} (Stock #{action['stock_id']})")
                        
                        col_info1, col_info2 = st.columns([3, 1])
                        
                        with col_info1:
                            st.markdown("**Why:**")
                            st.markdown(f"• Vehicle aging: **{action['stock_days']} days** in stock")
                            st.markdown(f"• New shipment arriving in **5-10 days**")
                            st.markdown(f"• Market average: **{action['avg_market_days']} days**")
                            st.markdown(f"• Demand index: **{action['demand_index']:.2f}** (below average)")
                            
                            st.markdown(f"**⏰ Urgency:** New inventory arrives soon")
                            st.markdown(f"**📈 Expected Outcome:** Clear in **{action['expected_clearance']} days**, maintain margin at **{action['current_margin'] + action['recommended_discount']:.1f}%**")
                        
                        with col_info2:
                            st.metric("Stock Days", action['stock_days'], delta=f"-{action['avg_market_days'] - action['stock_days']:.0f} vs avg", delta_color="inverse")
                            st.metric("Current Price", f"QAR {action['current_price']/1000:.0f}K")
                        
                        with st.expander("📊 View Data Sources (4)"):
                            st.markdown("• Inventory Management System")
                            st.markdown("• Market Pricing Data")
                            st.markdown("• Historical Sales Velocity")
                            st.markdown("• Demand Analytics")
                        
                        st.info(f"💡 **Recommended Action:** Apply **QAR {int(action['recommended_discount'] * action['current_price'] / 100):,}** discount ({action['recommended_discount']:.1f}% reduction)")
                        
                        col_a, col_b, col_c = st.columns([2, 1, 1])
                        with col_a:
                            st.button("Accept Recommendation", key=f"accept_{idx}", type="primary")
                        with col_b:
                            st.button("Defer", key=f"defer_inv_{idx}")
                        with col_c:
                            st.button("Override", key=f"override_{idx}")
                    
                    elif action['type'] == 'retention':
                        st.markdown(f"### ⚠️ Retention Alert: {action['customer_id']} - High churn risk")
                        
                        col_info1, col_info2 = st.columns([3, 1])
                        
                        with col_info1:
                            st.markdown("**Why:**")
                            st.markdown(f"• No service in **{action['days_since_service']} days**")
                            st.markdown(f"• Service visits last 12 months: **{action['service_count']}**")
                            st.markdown(f"• Loyalty score: **{action['loyalty_score']:.0f}** (declining)")
                            st.markdown(f"• Churn risk: **{action['churn_risk']*100:.0f}%**")
                            
                            st.markdown(f"**⏰ Urgency:** 7-day action window")
                            st.markdown(f"**📈 Expected Outcome:** **{action['retention_prob']}%** retention probability, preserve **QAR {action['annual_value']/1000:.0f}K** annual value")
                        
                        with col_info2:
                            st.metric("Customer CLV", f"QAR {action['clv']/1000:.0f}K")
                            st.metric("Churn Risk", f"{action['churn_risk']*100:.0f}%", delta="High", delta_color="inverse")
                        
                        with st.expander("📊 View Data Sources (4)"):
                            st.markdown("• Service History Database")
                            st.markdown("• Customer Engagement Tracking")
                            st.markdown("• CLV Prediction Model")
                            st.markdown("• Churn Risk Analytics")
                        
                        st.success("✓ **Recommended Actions:** Personal manager call • Complimentary inspection • Loyalty discount (10-15%) • Extended warranty offer")
                        
                        col_a, col_b, col_c = st.columns([2, 1, 1])
                        with col_a:
                            st.button("✓ Take Action", key=f"retention_{idx}", type="primary")
                        with col_b:
                            st.button("Defer", key=f"defer_ret_{idx}")
                        with col_c:
                            st.button("Reject", key=f"reject_ret_{idx}")
                    
                    elif action['type'] == 'service_campaign':
                        st.markdown(f"### Launch Service Reminder Campaign - {action['customer_count']} customers due")
                        
                        st.markdown("**Why:**")
                        st.markdown(f"• **{action['customer_count']} customers** with service due in next 30 days")
                        st.markdown(f"• Average service value: **QAR {action['avg_service_cost']:,.0f}**")
                        
                        top_brands_str = ", ".join([f"{brand} ({count})" for brand, count in list(action['top_brands'].items())[:3]])
                        st.markdown(f"• Top brands: **{top_brands_str}**")
                        st.markdown(f"• Total potential revenue: **QAR {action['total_potential']/1000:.0f}K**")
                        
                        st.markdown(f"**⏰ Urgency:** 14-day campaign window")
                        st.markdown(f"**📈 Expected Outcome:** **{action['conversion_rate']}%** booking rate, **{action['expected_bookings']}** appointments, **QAR {action['expected_revenue']/1000:.0f}K** revenue")
                        
                        with st.expander("📊 View Data Sources (3)"):
                            st.markdown("• Service Schedule Database")
                            st.markdown("• Vehicle Maintenance Records")
                            st.markdown("• Historical Campaign Performance")
                        
                        st.success("✓ **Recommended Actions:** SMS/Email campaign • 10% advance booking discount • Free inspection bundle • Extended hours")
                        
                        col_a, col_b, col_c = st.columns([2, 1, 1])
                        with col_a:
                            st.button("Launch Campaign", key=f"campaign_{idx}", type="primary")
                        with col_b:
                            st.button("Customize", key=f"custom_{idx}")
                        with col_c:
                            st.button("Defer", key=f"defer_camp_{idx}")
                    
                    elif action['type'] == 'marketing_ev':
                        st.markdown(f"### EV Subsidy Campaign - Target Qatari Nationals")
                        
                        st.markdown("**Campaign:** Qatar Green Initiative 2026")
                        
                        st.markdown("**Why:**")
                        st.markdown(f"• Government EV subsidy: **QAR {action['subsidy_amount']:,}/vehicle**")
                        st.markdown(f"• **{action['ev_inventory']}** EV/Hybrid units in stock")
                        st.markdown(f"• Target audience: **{action['target_audience']}** eligible Qatari customers")
                        st.markdown(f"• Program ends: **31-Mar-2026** (76 days)")
                        
                        st.markdown(f"**⏰ Urgency:** 75-day program window")
                        st.markdown(f"**📈 Expected Outcome:** **{action['estimated_conversions']}** estimated conversions, **QAR {action['expected_revenue']/1e6:.1f}M** revenue")
                        
                        with st.expander("📊 View Data Sources (4)"):
                            st.markdown("• Customer Database")
                            st.markdown("• Inventory Management System")
                            st.markdown("• Government Program Database")
                            st.markdown("• Past Campaign Analytics")
                        
                        st.success("✓ **Recommended Actions:** Email/SMS to Qatari customers • Social media ads • 0% financing highlight • Weekend EV showcase")
                        
                        col_a, col_b, col_c = st.columns([2, 1, 1])
                        with col_a:
                            st.button("Launch Campaign", key=f"ev_campaign_{idx}", type="primary")
                        with col_b:
                            st.button("Customize", key=f"ev_custom_{idx}")
                        with col_c:
                            st.button("Defer", key=f"defer_ev_{idx}")
                    
                    st.markdown("---")
        
        with col_sidebar:
            st.subheader("📋 Today's Activity")
            st.caption("Actions completed by hour")
            
            # Activity chart
            hours = ['10AM', '11AM', '12PM', '1PM', '2PM', '3PM']
            completions = [8, 12, 16, 10, 7, 13]
            
            fig = go.Figure(data=[
                go.Bar(x=hours, y=completions, marker_color='#0047AB', text=completions, textposition='outside')
            ])
            
            fig.update_layout(
                height=250,
                margin=dict(l=0, r=0, t=20, b=0),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.subheader("📊 Next Actions Queue")
            
            st.metric("High Priority", high_priority, label_visibility="visible")
            st.metric("Medium Priority", medium_priority, label_visibility="visible")
            st.metric("Low Priority", len(recommendations) - high_priority - medium_priority, label_visibility="visible")

# =============================================================================
# PAGE 6: CUSTOMERS & LEADS (from app4.py)
# =============================================================================
elif page == "👥 Sales Pipeline":
    if df_customer is None:
        st.warning("⚠️ Customers & Leads requires customer data. Please ensure 'processed_dealer_data.csv' is available.")
    else:
        st.title("Customers & Leads")
        st.caption("360° customer view with behavioral insights and recommendations")
        st.markdown("---")
        
        # Get customer data
        top_customers = df_forecast.nlargest(50, 'total_customer_lifetime_value')['customer_id'].unique()
        selected = st.selectbox("🔍 Search customers...", [''] + list(top_customers))
        
        if selected:
            customer = df_forecast[df_forecast['customer_id'] == selected].iloc[-1]
            
            # Calculate personalized metrics from data
            website_visits = int(customer['loyalty_score'] / 8) + np.random.randint(3, 8)
            config_sessions = int((customer['loyalty_score'] / 100) * 3) + np.random.randint(0, 2)
            test_drives = int(customer['service_count_last_12_months'] / 4) if customer['service_count_last_12_months'] > 0 else 0
            showroom_visits = 1 if customer['loyalty_score'] > 60 else 0
            if customer['churn_risk_score'] < 0.3:
                showroom_visits += 1
            
            # Determine customer stage based on actual data
            if customer['loyalty_score'] > 80 and customer['churn_risk_score'] < 0.3:
                stage = "High Intent"
                stage_color = "warning"
            elif customer['loyalty_score'] > 60:
                stage = "Consideration"
                stage_color = "info"
            elif customer['churn_risk_score'] > 0.6:
                stage = "At Risk"
                stage_color = "error"
            else:
                stage = "Early Stage"
                stage_color = "success"
            
            # Calculate purchase probability
            purchase_prob = min(95, customer['loyalty_score'] + (10 if customer['churn_risk_score'] < 0.3 else -10))
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.subheader(f"{customer['customer_id']}")
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.info(f"**{customer['nationality']}**")
                with col_b:
                    st.success(f"**{customer['customer_type']} Buyer**")
                with col_c:
                    if stage_color == "warning":
                        st.warning(f"**{stage}**")
                    elif stage_color == "error":
                        st.error(f"**{stage}**")
                    elif stage_color == "info":
                        st.info(f"**{stage}**")
                    else:
                        st.success(f"**{stage}**")
                
                st.markdown("---")
                st.subheader("🔍 Activity Signals")
                
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("Website Visits", website_visits, "Last 30 days")
                with col_b:
                    config_text = "Yes" if config_sessions > 0 else "No"
                    st.metric("Configurator Use", config_text, f"{config_sessions} sessions")
                with col_c:
                    st.metric("Test Drives", test_drives, "Completed")
                with col_d:
                    st.metric("Showroom Visits", showroom_visits, "Physical visits")
                
                st.markdown("---")
                st.subheader(f"📋 Recommended Actions for {selected}")
                
                # Generate personalized recommendations based on customer data
                recommendations_list = []
                
                # High intent + high loyalty
                if customer['loyalty_score'] > 75 and customer['churn_risk_score'] < 0.35:
                    recommendations_list.append({
                        'action': f"Schedule immediate test drive for {customer['brand']} {customer['model']}",
                        'reason': f"High purchase probability ({purchase_prob:.0f}%) with strong engagement signals",
                        'type': 'success'
                    })
                    if customer['nationality'] == 'Qatari':
                        recommendations_list.append({
                            'action': f"Present Qatari National preferred financing (0% APR available)",
                            'reason': f"Eligible for special financing on {customer['brand']} vehicles",
                            'type': 'info'
                        })
                    else:
                        recommendations_list.append({
                            'action': f"Offer trade-in evaluation for upgrade opportunity",
                            'reason': f"Customer has {customer['loyalty_score']:.0f} loyalty score - excellent upgrade candidate",
                            'type': 'info'
                        })
                
                # Medium loyalty - needs nurturing
                elif customer['loyalty_score'] > 50 and customer['loyalty_score'] <= 75:
                    recommendations_list.append({
                        'action': f"Send personalized email with {customer['brand']} {customer['model']} details",
                        'reason': f"Customer in consideration stage - {config_sessions} configurator sessions completed",
                        'type': 'info'
                    })
                    if customer['vehicle_price'] > 200000:
                        recommendations_list.append({
                            'action': f"Offer VIP showroom tour with senior sales consultant",
                            'reason': f"High-value vehicle interest (QAR {customer['vehicle_price']/1000:.0f}K) warrants premium experience",
                            'type': 'success'
                        })
                    else:
                        recommendations_list.append({
                            'action': f"Highlight current promotions and financing options",
                            'reason': f"Price-sensitive segment - showcase value propositions",
                            'type': 'info'
                        })
                
                # At-risk customer
                elif customer['churn_risk_score'] > 0.5:
                    recommendations_list.append({
                        'action': f"URGENT: Schedule retention call with dealership manager",
                        'reason': f"High churn risk ({customer['churn_risk_score']*100:.0f}%) - customer shows disengagement",
                        'type': 'error'
                    })
                    if customer['service_count_last_12_months'] == 0:
                        recommendations_list.append({
                            'action': f"Offer complimentary vehicle inspection (QAR 500 value)",
                            'reason': f"Zero service visits in 12 months - re-engage through service",
                            'type': 'warning'
                        })
                    else:
                        recommendations_list.append({
                            'action': f"Provide exclusive loyalty discount (15% off next purchase)",
                            'reason': f"Preserve QAR {customer['total_customer_lifetime_value']/1000:.0f}K customer lifetime value",
                            'type': 'warning'
                        })
                
                # Low engagement - early stage
                else:
                    recommendations_list.append({
                        'action': f"Invite to exclusive vehicle showcase event",
                        'reason': f"Early stage customer - build relationship through in-person engagement",
                        'type': 'info'
                    })
                    if customer['customer_type'] == 'Corporate':
                        recommendations_list.append({
                            'action': f"Present corporate fleet program benefits",
                            'reason': f"Corporate buyer - highlight bulk purchase incentives and B2B services",
                            'type': 'success'
                        })
                    else:
                        recommendations_list.append({
                            'action': f"Schedule test drive with flexible timing options",
                            'reason': f"Individual buyer - offer convenience and personalized attention",
                            'type': 'info'
                        })
                
                # Service-based recommendations
                if customer['next_service_due_days'] <= 30 and customer['next_service_due_days'] >= 0:
                    recommendations_list.append({
                        'action': f"Book service appointment (due in {int(customer['next_service_due_days'])} days)",
                        'reason': f"Service due soon - estimated cost QAR {customer['avg_service_cost']:,.0f}",
                        'type': 'warning'
                    })
                
                # Display recommendations
                for i, rec in enumerate(recommendations_list):
                    if rec['type'] == 'success':
                        st.success(f"**✓ Action {i+1}**")
                    elif rec['type'] == 'error':
                        st.error(f"**⚠️ Action {i+1}**")
                    elif rec['type'] == 'warning':
                        st.warning(f"**⚡ Action {i+1}**")
                    else:
                        st.info(f"**→ Action {i+1}**")
                    
                    st.markdown(f"**{rec['action']}**")
                    st.caption(rec['reason'])
                    st.markdown("")
                
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    st.button("📞 Call Now", type="primary", use_container_width=True)
                with col_btn2:
                    st.button("✉️ Send Email", use_container_width=True)
            
            with col2:
                st.metric("Purchase Probability", f"{purchase_prob:.0f}%", label_visibility="visible")
                
                st.markdown("---")
                st.subheader("📊 Customer Profile")
                
                st.markdown(f"**Last Contact:** {customer['sale_date'].strftime('%Y-%m-%d')}")
                st.markdown(f"**Vehicle Interest:** {customer['brand']} {customer['model']}")
                st.markdown(f"**Stage:** {stage}")
                st.markdown(f"**Loyalty Score:** {customer['loyalty_score']:.0f}/100")
                st.markdown(f"**Churn Risk:** {customer['churn_risk_score']*100:.0f}%")
                st.markdown(f"**CLV:** QAR {customer['total_customer_lifetime_value']/1000:.0f}K")
                st.markdown(f"**Service Count (12mo):** {int(customer['service_count_last_12_months'])}")
                
                # Additional insights
                st.markdown("---")
                st.subheader("🎯 Key Insights")
                
                if customer['income_band'] == 'High':
                    st.info("💰 High-income segment")
                elif customer['income_band'] == 'Medium':
                    st.info("💵 Medium-income segment")
                else:
                    st.info("💳 Value-conscious segment")
                
                if customer['warranty_status'] == 'Under Warranty':
                    st.success("✓ Active warranty")
                else:
                    st.warning("⚠️ Warranty expired")
                
                # Show vehicle details
                st.markdown(f"**Current Vehicle:** {customer['model_year']} {customer['brand']} {customer['model']}")
                st.markdown(f"**Fuel Type:** {customer['fuel_type']}")
                st.markdown(f"**Transmission:** {customer['transmission']}")

# =============================================================================
# PAGE 7: INVENTORY & PRICING (from app4.py)
# =============================================================================
elif page == "📦 Inventory Hub":
    if engine is None:
        st.warning("⚠️ Inventory & Pricing requires customer data. Please ensure 'processed_dealer_data.csv' is available.")
    else:
        st.title("Inventory & Pricing")
        st.caption("AI-powered inventory management with prescriptive pricing actions")
        st.markdown("---")
        
        # Get inventory actions
        inventory_actions = engine.get_inventory_actions()
        
        # Summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Inventory", len(df_forecast), "Units in stock")
        
        with col2:
            st.metric("Avg. Stock Days", int(df_forecast['stock_days'].mean()), "-8 vs target")
        
        with col3:
            st.metric("Aging Units", len(df_forecast[df_forecast['stock_days'] > 90]), ">90 days")
        
        with col4:
            st.metric("Avg. Margin", f"{df_forecast['discount_percentage'].mean():.1f}%", "+1.2% vs LY")
        
        st.markdown("---")
        st.subheader("📦 Inventory with Recommended Actions")
        st.caption("Priority items requiring attention")
        
        # Display inventory actions
        for idx, action in enumerate(inventory_actions[:5]):
            with st.container():
                col_a, col_b = st.columns([3, 1])
                
                with col_a:
                    if action['priority'] == 'High':
                        st.error(f"🔴 {action['priority']} Priority | {action['confidence']}% Confidence")
                    else:
                        st.warning(f"🟡 {action['priority']} Priority | {action['confidence']}% Confidence")
                    
                    st.markdown(f"### {action['vehicle']}")
                    st.caption(f"Stock #{action['stock_id']}")
                    
                    st.markdown("**Why:**")
                    st.markdown(f"• Vehicle aging: **{action['stock_days']} days** in stock")
                    st.markdown(f"• New shipment arriving in **5-10 days**")
                    st.markdown(f"• Market average: **{action['avg_market_days']} days**")
                    st.markdown(f"• Demand index: **{action['demand_index']:.2f}**")
                    
                    st.markdown(f"**⏰ Urgency:** New inventory arrives soon")
                    st.markdown(f"**✓ Expected:** Clear in **{action['expected_clearance']} days**, maintain margin")
                    
                    st.info(f"💡 **Recommended:** Apply **QAR {int(action['recommended_discount'] * action['current_price'] / 100):,}** discount")
                
                with col_b:
                    st.metric("Stock Days", action['stock_days'], delta=f"{action['stock_days'] - action['avg_market_days']:.0f} vs avg", delta_color="inverse")
                    st.metric("Current Price", f"QAR {action['current_price']/1000:.0f}K")
                    st.metric("Margin", f"{action['current_margin']:.1f}%")
                
                col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
                with col_btn1:
                    st.button("Accept Recommendation", key=f"acc_inv_{idx}", type="primary")
                with col_btn2:
                    st.button("Defer", key=f"def_inv_{idx}")
                with col_btn3:
                    st.button("Override", key=f"ovr_inv_{idx}")
                
                st.markdown("---")

# =============================================================================
# REMAINING CUSTOMER ANALYTICS PAGES (from app2.py)
# =============================================================================
elif page == "👤 Customer 360°" and df_customer is not None:
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

elif page == "🚨 Churn Prevention" and df_customer is not None:
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

elif page == "💰 Lifetime Value" and df_customer is not None:
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

elif page == "🔧 Service Center" and df_customer is not None:
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

elif page == "📊 Sales Intelligence" and df_customer is not None:
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

elif page == "🎯 Marketing Hub" and df_customer is not None:
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

elif page == "🤖 AI Assistant" and df_customer is not None:
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