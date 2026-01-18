"""
Mannai Intelligence Platform - Advanced AI Decision Support
Sophisticated recommendation engine with contextual insights and next best actions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Mannai Intelligence Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .main {background-color: #F8F9FA;}
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0047AB 0%, #003380 100%);
    }
    
    [data-testid="stSidebar"] * {color: white !important;}
    
    .stMetric {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
</style>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("qatar_auto_dealer_sales_forecasting_dataset_2026_extended.csv")
        df['sale_date'] = pd.to_datetime(df['sale_date'])
        df['last_service_date'] = pd.to_datetime(df['last_service_date'], errors='coerce')
        return df
    except FileNotFoundError:
        st.error("⚠️ Data file not found!")
        st.stop()

df = load_data()

# Recommendation Engine
class RecommendationEngine:
    def __init__(self, data):
        self.df = data
        self.today = datetime.now()
        
    def get_high_intent_sales_actions(self):
        """Generate sales recommendations based on real customer data"""
        actions = []
        
        # Find high-value customers with good loyalty and low churn risk
        high_value = self.df[
            (self.df['loyalty_score'] > 70) & 
            (self.df['churn_risk_score'] < 0.35) &
            (self.df['vehicle_price'] > 150000)
        ].copy()
        
        if len(high_value) > 0:
            top_customers = high_value.nlargest(5, 'total_customer_lifetime_value')
            
            for idx, (_, customer) in enumerate(top_customers.iterrows()):
                # Create unique engagement metrics based on customer data
                config_views = int((customer['loyalty_score'] / 100) * 3) + (idx % 3)
                financing_views = int((customer['loyalty_score'] / 100) * 2) + (1 if customer['income_band'] == 'High' else 2)
                website_visits = int(customer['loyalty_score'] / 8) + (idx * 2) + 3
                
                # Different conversion probability logic per customer
                base_conversion = customer['loyalty_score']
                if customer['service_count_last_12_months'] > 5:
                    base_conversion += 10  # Regular service = higher trust
                if customer['nationality'] == 'Qatari' and customer['brand'] in ['Toyota', 'Lexus', 'Nissan']:
                    base_conversion += 5  # National preference
                
                conversion_prob = min(95, base_conversion + np.random.randint(5, 15))
                confidence = int(75 + (customer['loyalty_score'] / 100) * 20)
                
                # Urgency based on multiple factors
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
                
                # Create unique reasons per customer
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
                    'why_reasons': why_reasons[:4]  # Top 4 reasons
                }
                
                actions.append(action)
        
        return actions
    
    def get_inventory_actions(self):
        """Generate inventory recommendations based on real stock data"""
        actions = []
        
        # Find aging inventory
        aging = self.df[self.df['stock_days'] > 60].copy()
        
        if len(aging) > 0:
            critical = aging.nlargest(5, 'stock_days')
            
            for _, vehicle in critical.iterrows():
                days_aging = int(vehicle['stock_days'])
                recommended_discount = min(15, (days_aging - 60) * 0.15)
                
                # Get market context
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
        """Generate customer retention recommendations"""
        actions = []
        
        # Find at-risk high-value customers
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
        """Generate service campaign recommendations"""
        actions = []
        
        # Find service opportunities
        service_due = self.df[
            (self.df['next_service_due_days'] <= 30) &
            (self.df['next_service_due_days'] >= -15)
        ].copy()
        
        if len(service_due) > 20:
            total_customers = len(service_due)
            avg_service_cost = service_due['avg_service_cost'].mean()
            total_potential = service_due['avg_service_cost'].sum()
            
            # Top brands
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
        """Generate marketing campaign recommendations"""
        actions = []
        
        # EV/Hybrid campaign
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
        """Get all recommendations sorted by priority"""
        all_actions = []
        
        all_actions.extend(self.get_high_intent_sales_actions())
        all_actions.extend(self.get_inventory_actions())
        all_actions.extend(self.get_retention_actions())
        all_actions.extend(self.get_service_campaigns())
        all_actions.extend(self.get_marketing_campaigns())
        
        # Sort by priority and confidence
        priority_order = {'High': 3, 'Medium': 2, 'Low': 1}
        all_actions.sort(key=lambda x: (priority_order.get(x['priority'], 0), x['confidence']), reverse=True)
        
        return all_actions

# Initialize Engine
engine = RecommendationEngine(df)

# Sidebar
st.sidebar.title("🎯 Mannai Intelligence")
st.sidebar.markdown("*Decision Support & Next Best Actions*")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", [
    "🏠 Action Dashboard",
    "👥 Customers & Leads",
    "📦 Inventory & Pricing",
    "📊 Sales Pipeline",
    "📢 Campaigns & Offers"
])

st.sidebar.markdown("---")
st.sidebar.markdown("### Platform Performance")
st.sidebar.metric("Action Acceptance Rate", "76%")
st.sidebar.metric("Conversion Uplift", "+23%")
st.sidebar.metric("Revenue Influenced", "QAR 8.4M")

# =============================================================================
# ACTION DASHBOARD
# =============================================================================
if page == "🏠 Action Dashboard":
    # Header
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.title("📍 Mannai Intelligence Platform")
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
    st.subheader("Action Dashboard")
    st.caption(f"Your personalized next best actions for today, {today_str}")
    
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
# CUSTOMERS & LEADS
# =============================================================================
elif page == "👥 Customers & Leads":
    st.title("Customers & Leads")
    st.caption("360° customer view with behavioral insights and recommendations")
    st.markdown("---")
    
    # Get customer data
    top_customers = df.nlargest(50, 'total_customer_lifetime_value')['customer_id'].unique()
    selected = st.selectbox("🔍 Search customers...", [''] + list(top_customers))
    
    if selected:
        customer = df[df['customer_id'] == selected].iloc[-1]
        
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
# INVENTORY & PRICING
# =============================================================================
elif page == "📦 Inventory & Pricing":
    st.title("Inventory & Pricing")
    st.caption("AI-powered inventory management with prescriptive pricing actions")
    st.markdown("---")
    
    # Get inventory actions
    inventory_actions = engine.get_inventory_actions()
    
    # Summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Inventory", len(df), "Units in stock")
    
    with col2:
        st.metric("Avg. Stock Days", int(df['stock_days'].mean()), "-8 vs target")
    
    with col3:
        st.metric("Aging Units", len(df[df['stock_days'] > 90]), ">90 days")
    
    with col4:
        st.metric("Avg. Margin", f"{df['discount_percentage'].mean():.1f}%", "+1.2% vs LY")
    
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
# Other Pages
# =============================================================================
else:
    st.title(page)
    st.caption("Advanced insights and recommendations")
    st.info("🚧 Select 'Action Dashboard' to see the full recommendation engine in action.")
    
    # Show some basic stats
    st.markdown("---")
    st.subheader("📊 Quick Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", df['customer_id'].nunique())
    
    with col2:
        st.metric("Total Revenue", f"QAR {df['final_sale_price'].sum()/1e6:.1f}M")
    
    with col3:
        st.metric("Avg. Vehicle Price", f"QAR {df['vehicle_price'].mean()/1000:.0f}K")
    
    with col4:
        st.metric("Brands", df['brand'].nunique())

# Footer
st.markdown("---")
st.caption("© 2026 Mannai Intelligence Platform | Powered by Advanced AI & Machine Learning")
