"""
Qatar Auto Dealer AI Platform - Streamlit Dashboard
Complete AI-powered analytics and insights for car dealership
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
import os

# Page Configuration
st.set_page_config(
    page_title="Qatar Dealer AI Platform",
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
</style>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("processed_dealer_data.csv")
        return df
    except FileNotFoundError:
        st.error("⚠️ Data file not found! Please run 'python preprocess_data.py' first.")
        st.stop()

df = load_data()

# OpenAI API Setup (for AI features)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Sidebar Navigation
st.sidebar.markdown("## 🚗 Qatar Dealer AI Platform")
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio("Select Module:", [
    "📊 Executive Dashboard",
    "👥 Customer Segmentation",
    "🚨 Churn Prediction",
    "💰 Customer Lifetime Value",
    "🔧 Service Optimization",
    "📈 Sales Insights",
    "🎯 Marketing Campaigns",
    "🤖 AI Sales Assistant",
    "🔍 Customer Search"
])

# =============================================================================
# PAGE 1: EXECUTIVE DASHBOARD
# =============================================================================
if page == "📊 Executive Dashboard":
    st.markdown("<h1 class='main-header'>Executive Dashboard</h1>", unsafe_allow_html=True)
    
    # Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    with col2:
        total_rev = df['total_customer_lifetime_value'].sum()
        st.metric("Total Revenue", f"QAR {total_rev/1e6:.1f}M")
    with col3:
        avg_clv = df['total_customer_lifetime_value'].mean()
        st.metric("Avg CLV", f"QAR {avg_clv/1e3:.0f}K")
    with col4:
        high_churn = len(df[df['churn_probability'] > 0.7])
        st.metric("High Churn Risk", f"{high_churn:,}", delta=f"-{high_churn/len(df)*100:.1f}%", delta_color="inverse")
    with col5:
        avg_loyalty = df['loyalty_score'].mean()
        st.metric("Avg Loyalty Score", f"{avg_loyalty:.1f}/100")
    
    st.markdown("---")
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        brand_perf = df.groupby('brand')['total_customer_lifetime_value'].sum().sort_values(ascending=False)
        fig1 = px.bar(x=brand_perf.values, y=brand_perf.index, orientation='h',
                     title="📊 Revenue by Brand", labels={'x': 'Revenue (QAR)', 'y': 'Brand'},
                     color=brand_perf.values, color_continuous_scale='Blues')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        segment_counts = df['segment_name'].value_counts()
        fig2 = px.pie(values=segment_counts.values, names=segment_counts.index,
                     title="🎯 Customer Segments Distribution", hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Charts Row 2
    col3, col4 = st.columns(2)
    
    with col3:
        fig3 = px.histogram(df, x='total_customer_lifetime_value', nbins=50,
                           title="💰 Customer Lifetime Value Distribution",
                           labels={'total_customer_lifetime_value': 'CLV (QAR)'})
        fig3.add_vline(x=df['total_customer_lifetime_value'].median(), 
                      line_dash="dash", line_color="red", annotation_text="Median")
        st.plotly_chart(fig3, use_container_width=True)
    
    with col4:
        churn_by_brand = df.groupby('brand')['churn_probability'].mean().sort_values()
        fig4 = px.bar(x=churn_by_brand.values, y=churn_by_brand.index, orientation='h',
                     title="⚠️ Average Churn Probability by Brand",
                     labels={'x': 'Churn Probability', 'y': 'Brand'},
                     color=churn_by_brand.values, color_continuous_scale='Reds')
        fig4.add_vline(x=0.6, line_dash="dash", line_color="darkred", annotation_text="High Risk")
        st.plotly_chart(fig4, use_container_width=True)

# =============================================================================
# PAGE 2: CUSTOMER SEGMENTATION
# =============================================================================
elif page == "👥 Customer Segmentation":
    st.markdown("<h1 class='main-header'>Customer Segmentation Analysis</h1>", unsafe_allow_html=True)
    
    # Segment Summary Table
    segment_summary = df.groupby('segment_name').agg({
        'customer_id': 'count',
        'total_customer_lifetime_value': 'mean',
        'loyalty_score': 'mean',
        'churn_probability': 'mean',
        'service_count_last_12_months': 'mean'
    }).round(2)
    segment_summary.columns = ['Count', 'Avg CLV (QAR)', 'Avg Loyalty', 'Avg Churn', 'Avg Services']
    
    st.dataframe(segment_summary.style.background_gradient(cmap='Blues'), use_container_width=True)
    
    # Interactive Scatter Plot
    fig = px.scatter(df, x='loyalty_score', y='total_customer_lifetime_value',
                    color='segment_name', size='churn_probability',
                    title='🎯 Customer Segmentation: Loyalty vs Lifetime Value',
                    labels={'loyalty_score': 'Loyalty Score', 
                           'total_customer_lifetime_value': 'CLV (QAR)'},
                    hover_data=['brand', 'model'])
    st.plotly_chart(fig, use_container_width=True)
    
    # Segment Details
    st.markdown("### 📋 Segment Details")
    selected_segment = st.selectbox("Select Segment:", df['segment_name'].unique())
    segment_customers = df[df['segment_name'] == selected_segment][
        ['customer_id', 'brand', 'model', 'total_customer_lifetime_value',
         'loyalty_score', 'churn_probability']
    ].sort_values('total_customer_lifetime_value', ascending=False)
    
    st.dataframe(segment_customers.head(50), use_container_width=True)
    st.download_button("📥 Download Segment Data", 
                      segment_customers.to_csv(index=False),
                      f"segment_{selected_segment}.csv")

# =============================================================================
# PAGE 3: CHURN PREDICTION
# =============================================================================
elif page == "🚨 Churn Prediction":
    st.markdown("<h1 class='main-header'>Churn Prediction & Prevention</h1>", unsafe_allow_html=True)
    
    high_risk = df[df['churn_probability'] > 0.7].sort_values('total_customer_lifetime_value', ascending=False)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🚨 High Risk Customers", f"{len(high_risk):,}")
    with col2:
        revenue_risk = high_risk['total_customer_lifetime_value'].sum()
        st.metric("💸 Revenue at Risk", f"QAR {revenue_risk/1e6:.2f}M")
    with col3:
        avg_churn = df['churn_probability'].mean()
        st.metric("📊 Avg Churn Probability", f"{avg_churn:.1%}")
    
    # Risk Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.histogram(df, x='churn_probability', nbins=30,
                           title="📈 Churn Probability Distribution")
        fig1.add_vline(x=0.7, line_dash="dash", line_color="red", 
                      annotation_text="High Risk Threshold")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Churn by segment
        churn_segment = df.groupby('segment_name')['churn_probability'].mean().sort_values()
        fig2 = px.bar(x=churn_segment.values, y=churn_segment.index, orientation='h',
                     title="⚠️ Churn Risk by Segment",
                     color=churn_segment.values, color_continuous_scale='Reds')
        st.plotly_chart(fig2, use_container_width=True)
    
    # At-Risk Customers Table
    st.markdown("### 🎯 Top 30 At-Risk High-Value Customers")
    risk_table = high_risk.head(30)[
        ['customer_id', 'brand', 'model', 'loyalty_score',
         'churn_probability', 'total_customer_lifetime_value', 'segment_name']
    ]
    st.dataframe(risk_table, use_container_width=True)
    st.download_button("📥 Download High-Risk List", 
                      high_risk.to_csv(index=False),
                      "high_risk_customers.csv")

# =============================================================================
# PAGE 4: CUSTOMER LIFETIME VALUE
# =============================================================================
elif page == "💰 Customer Lifetime Value":
    st.markdown("<h1 class='main-header'>Customer Lifetime Value Prediction</h1>", unsafe_allow_html=True)
    
    high_potential = df[df['clv_growth_potential'] > df['clv_growth_potential'].quantile(0.9)].sort_values('clv_growth_potential', ascending=False)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_current = df['total_customer_lifetime_value'].mean()
        st.metric("💵 Avg Current CLV", f"QAR {avg_current/1e3:.0f}K")
    with col2:
        avg_predicted = df['predicted_clv'].mean()
        st.metric("📈 Avg Predicted CLV", f"QAR {avg_predicted/1e3:.0f}K")
    with col3:
        total_potential = df['clv_growth_potential'].sum()
        st.metric("🚀 Total Growth Potential", f"QAR {total_potential/1e6:.1f}M")
    
    # CLV Comparison Plot
    fig = px.scatter(df, x='total_customer_lifetime_value', y='predicted_clv',
                    color='segment_name', 
                    title='💰 Current vs Predicted CLV by Segment',
                    labels={'total_customer_lifetime_value': 'Current CLV (QAR)',
                           'predicted_clv': 'Predicted CLV (QAR)'},
                    hover_data=['brand', 'model'])
    
    # Add break-even line
    max_val = max(df['total_customer_lifetime_value'].max(), df['predicted_clv'].max())
    fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val],
                            mode='lines', name='Break-even',
                            line=dict(dash='dash', color='gray')))
    st.plotly_chart(fig, use_container_width=True)
    
    # High-Potential Customers
    st.markdown("### 🌟 Top 30 High-Potential Customers")
    potential_table = high_potential.head(30)[
        ['customer_id', 'brand', 'model', 'segment_name',
         'total_customer_lifetime_value', 'predicted_clv', 'clv_growth_potential']
    ]
    st.dataframe(potential_table, use_container_width=True)
    st.download_button("📥 Download High-Potential List",
                      high_potential.to_csv(index=False),
                      "high_potential_customers.csv")

# =============================================================================
# PAGE 5: SERVICE OPTIMIZATION
# =============================================================================
elif page == "🔧 Service Optimization":
    st.markdown("<h1 class='main-header'>Service Revenue Optimization</h1>", unsafe_allow_html=True)
    
    service_opps = df[df['service_due_soon'] | df['service_overdue']].copy()
    service_opps['priority'] = service_opps.apply(
        lambda x: 'URGENT' if x['service_overdue'] else 'HIGH' if x['next_service_due_days'] <= 14 else 'MEDIUM',
        axis=1
    )
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔴 Overdue Services", f"{df['service_overdue'].sum():,}")
    with col2:
        st.metric("🟡 Due Within 30 Days", f"{df['service_due_soon'].sum():,}")
    with col3:
        potential_rev = service_opps['avg_service_cost'].sum()
        st.metric("💰 Potential Revenue", f"QAR {potential_rev/1e3:.0f}K")
    
    # Service Opportunities by Brand
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
    
    # Priority Contacts
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

# =============================================================================
# PAGE 6: SALES INSIGHTS
# =============================================================================
elif page == "📈 Sales Insights":
    st.markdown("<h1 class='main-header'>Inventory & Sales Insights</h1>", unsafe_allow_html=True)
    
    # Top Models
    popular_models = df.groupby(['brand', 'model']).agg({
        'customer_id': 'count',
        'vehicle_price': 'mean',
        'loyalty_score': 'mean'
    }).round(0)
    popular_models.columns = ['Sales Count', 'Avg Price (QAR)', 'Avg Loyalty']
    popular_models = popular_models.sort_values('Sales Count', ascending=False).head(20)
    
    st.markdown("### 🏆 Top 20 Best-Selling Models")
    st.dataframe(popular_models, use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        year_analysis = df['model_year'].value_counts().sort_index()
        fig1 = px.bar(x=year_analysis.index, y=year_analysis.values,
                     title="📅 Sales by Model Year",
                     labels={'x': 'Model Year', 'y': 'Units Sold'})
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        brand_type = df.groupby(['brand', 'customer_type']).size().unstack(fill_value=0)
        fig2 = px.imshow(brand_type.values, 
                        x=brand_type.columns, 
                        y=brand_type.index,
                        title="👥 Customer Type Preferences by Brand",
                        labels=dict(x="Customer Type", y="Brand", color="Count"),
                        color_continuous_scale="Blues")
        st.plotly_chart(fig2, use_container_width=True)

# =============================================================================
# PAGE 7: MARKETING CAMPAIGNS
# =============================================================================
elif page == "🎯 Marketing Campaigns":
    st.markdown("<h1 class='main-header'>Personalized Marketing Campaigns</h1>", unsafe_allow_html=True)
    
    # Define campaigns
    campaign1 = df[(df['churn_probability'] > 0.65) & 
                   (df['total_customer_lifetime_value'] > df['total_customer_lifetime_value'].quantile(0.75))]
    campaign2 = df[(df['days_since_last_service'] > 365) & (df['is_under_warranty'] == 0)]
    campaign3 = df[(df['clv_growth_potential'] > df['clv_growth_potential'].quantile(0.85)) & 
                   (df['loyalty_score'] > 70) & (df['vehicle_age_years'] >= 3)]
    campaign4 = df[(df['is_under_warranty'] == 1) & (df['vehicle_age_years'] >= 2)]
    
    # Campaign Metrics
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
    
    # Campaign Selection
    campaign_select = st.selectbox("Select Campaign:", [
        "🛡️ Churn Prevention",
        "🔄 Service Reactivation",
        "⬆️ Upsell/Upgrade",
        "📋 Warranty Expiration"
    ])
    
    # Show campaign details
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

# =============================================================================
# PAGE 8: AI SALES ASSISTANT
# =============================================================================
elif page == "🤖 AI Sales Assistant":
    st.markdown("<h1 class='main-header'>AI-Powered Sales Message Generator</h1>", unsafe_allow_html=True)
    
    st.info("💡 Generate personalized sales messages using AI (powered by OpenAI GPT-4)")
    
    # API Key Input
    api_key_input = st.text_input("Enter OpenAI API Key:", type="password", 
                                  value=OPENAI_API_KEY,
                                  help="Get your key from: https://platform.openai.com/api-keys")
    
    if api_key_input:
        # Customer Selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            customer_id = st.selectbox("Select Customer:", df['customer_id'].unique())
        
        with col2:
            language = st.selectbox("Language:", ["English", "Arabic", "Hindi"])
        
        customer_data = df[df['customer_id'] == customer_id].iloc[0]
        
        # Display customer info
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
        
        # Generate message
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
                    
                    # Copy button
                    st.code(message, language=None)
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("Please check your API key and try again.")
    else:
        st.warning("⚠️ Please enter your OpenAI API key to use this feature")

# =============================================================================
# PAGE 9: CUSTOMER SEARCH
# =============================================================================
elif page == "🔍 Customer Search":
    st.markdown("<h1 class='main-header'>Customer Search & Profile</h1>", unsafe_allow_html=True)
    
    search_id = st.text_input("🔍 Enter Customer ID:")
    
    if search_id:
        customer = df[df['customer_id'] == search_id]
        
        if len(customer) > 0:
            customer = customer.iloc[0]
            
            # Key Metrics
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
            
            # Detailed Profile
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
st.sidebar.metric("Total Customers", f"{len(df):,}")
st.sidebar.metric("Total Revenue", f"QAR {df['total_customer_lifetime_value'].sum()/1e6:.1f}M")
st.sidebar.metric("Avg CLV", f"QAR {df['total_customer_lifetime_value'].mean()/1e3:.0f}K")
st.sidebar.metric("High Churn Risk", f"{len(df[df['churn_probability'] > 0.7]):,}")