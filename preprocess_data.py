"""
Data Preprocessing Script for Car Dealership AI
Run this FIRST to prepare your data for the Streamlit app
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("🔄 Starting data preprocessing...")
print("="*80)

# Load data
df = pd.read_csv('qatar_auto_dealer_ai_big_dataset.csv')
print(f"✅ Loaded {len(df):,} records")

# Convert date columns
date_columns = ['sale_date', 'last_service_date']
for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Feature Engineering
print("\n🔧 Creating features...")
df['days_since_purchase'] = (pd.Timestamp.now() - df['sale_date']).dt.days
df['days_since_last_service'] = (pd.Timestamp.now() - df['last_service_date']).dt.days
df['vehicle_age_years'] = (pd.Timestamp.now().year - df['model_year'])
df['is_under_warranty'] = (df['warranty_status'] == 'Under Warranty').astype(int)
df['revenue_per_service'] = df['service_revenue_to_date'] / (df['service_count_last_12_months'] + 1)
df['total_revenue'] = df['service_revenue_to_date'] + df['parts_revenue_to_date']

# Customer Segmentation (K-Means)
print("\n🎯 Running customer segmentation...")
features_for_clustering = ['loyalty_score', 'churn_risk_score', 'total_customer_lifetime_value',
                           'service_count_last_12_months', 'avg_monthly_km', 'vehicle_age_years']

X_cluster = df[features_for_clustering].fillna(df[features_for_clustering].median())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['customer_segment'] = kmeans.fit_predict(X_scaled)

segment_labels = {
    0: 'Budget Conscious',
    1: 'Premium Loyalists',
    2: 'High-Risk Churners',
    3: 'Service Champions',
    4: 'VIP Elite'
}
df['segment_name'] = df['customer_segment'].map(segment_labels)

# Churn Prediction Model
print("\n⚡ Training churn prediction model...")
df['will_churn'] = (df['churn_risk_score'] > 0.35).astype(int)

le_brand = LabelEncoder()
le_customer_type = LabelEncoder()
le_income = LabelEncoder()

df['brand_encoded'] = le_brand.fit_transform(df['brand'])
df['customer_type_encoded'] = le_customer_type.fit_transform(df['customer_type'])
df['income_band_encoded'] = le_income.fit_transform(df['income_band'])

churn_features = ['loyalty_score', 'total_customer_lifetime_value', 'service_count_last_12_months',
                  'avg_monthly_km', 'vehicle_age_years', 'days_since_last_service',
                  'is_under_warranty', 'revenue_per_service', 'brand_encoded', 
                  'customer_type_encoded', 'income_band_encoded']

X = df[churn_features].fillna(df[churn_features].median())
y = df['will_churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

churn_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
churn_model.fit(X_train, y_train)

df['churn_prediction'] = churn_model.predict(X)
df['churn_probability'] = churn_model.predict_proba(X)[:, 1]

# CLV Prediction Model
print("\n💰 Training CLV prediction model...")
clv_features = ['loyalty_score', 'service_count_last_12_months', 'avg_monthly_km',
                'vehicle_age_years', 'days_since_last_service', 'is_under_warranty',
                'brand_encoded', 'customer_type_encoded', 'income_band_encoded',
                'avg_service_cost', 'vehicle_price']

X_clv = df[clv_features].fillna(df[clv_features].median())
y_clv = df['total_customer_lifetime_value']

X_train_clv, X_test_clv, y_train_clv, y_test_clv = train_test_split(X_clv, y_clv, test_size=0.2, random_state=42)

clv_model = lgb.LGBMRegressor(n_estimators=200, max_depth=7, learning_rate=0.05, random_state=42, verbose=-1)
clv_model.fit(X_train_clv, y_train_clv)

df['predicted_clv'] = clv_model.predict(X_clv)
df['clv_growth_potential'] = df['predicted_clv'] - df['total_customer_lifetime_value']

# Service Opportunities
df['service_overdue'] = df['next_service_due_days'] < 0
df['service_due_soon'] = (df['next_service_due_days'] >= 0) & (df['next_service_due_days'] <= 30)

# Save processed data
print("\n💾 Saving processed data...")
df.to_csv('processed_dealer_data.csv', index=False)

print("\n✅ SUCCESS! Data preprocessing complete!")
print("="*80)
print(f"📁 Created: processed_dealer_data.csv")
print(f"📊 Total records: {len(df):,}")
print(f"📈 Total columns: {len(df.columns)}")
print("\n🚀 Now run: streamlit run app.py")
print("="*80)