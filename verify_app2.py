"""
Verification Script for App2.py
Tests all components before running the main application
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Verify Python version"""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} - Need 3.8 or higher")
        return False

def check_packages():
    """Verify required packages are installed"""
    print("\n🔍 Checking required packages...")
    
    required_packages = {
        'streamlit': 'Streamlit',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'plotly': 'Plotly',
        'openai': 'OpenAI',
        'sklearn': 'Scikit-learn'
    }
    
    all_installed = True
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {name} - Installed")
        except ImportError:
            print(f"❌ {name} - NOT INSTALLED")
            all_installed = False
    
    return all_installed

def check_data_files():
    """Verify required data files exist"""
    print("\n🔍 Checking data files...")
    
    files_status = {}
    
    # Required file
    required_file = "qatar_auto_dealer_sales_forecasting_dataset_2026_final_external_factors.csv"
    if Path(required_file).exists():
        file_size = Path(required_file).stat().st_size / (1024 * 1024)  # MB
        print(f"✅ {required_file} - Found ({file_size:.2f} MB)")
        files_status['required'] = True
    else:
        print(f"❌ {required_file} - NOT FOUND (REQUIRED)")
        files_status['required'] = False
    
    # Optional file
    optional_file = "processed_dealer_data.csv"
    if Path(optional_file).exists():
        file_size = Path(optional_file).stat().st_size / (1024 * 1024)  # MB
        print(f"✅ {optional_file} - Found ({file_size:.2f} MB)")
        files_status['optional'] = True
    else:
        print(f"⚠️  {optional_file} - Not found (Optional - Customer analytics disabled)")
        files_status['optional'] = False
    
    return files_status

def test_data_loading():
    """Test if data can be loaded properly"""
    print("\n🔍 Testing data loading...")
    
    try:
        import pandas as pd
        
        # Test main dataset
        df = pd.read_csv("qatar_auto_dealer_sales_forecasting_dataset_2026_final_external_factors.csv")
        print(f"✅ Main dataset loaded - {len(df):,} rows, {len(df.columns)} columns")
        
        # Check for required columns
        required_cols = ['sale_date', 'brand', 'final_sale_price', 'auto_loan_interest_rate', 
                        'petrol_price_per_liter', 'oem_supply_constraint_index']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"⚠️  Missing columns: {missing_cols}")
        else:
            print(f"✅ All required columns present")
        
        # Check date parsing
        df['sale_date'] = pd.to_datetime(df['sale_date'])
        print(f"✅ Date parsing successful - Range: {df['sale_date'].min()} to {df['sale_date'].max()}")
        
        # Check brands
        brands = df['brand'].unique()
        print(f"✅ Brands found: {', '.join(brands)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return False

def test_model_components():
    """Test if forecasting components work"""
    print("\n🔍 Testing forecasting models...")
    
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        import numpy as np
        
        # Quick model test
        X = np.random.rand(100, 10)
        y = np.random.rand(100)
        
        model = GradientBoostingRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X[:5])
        
        print(f"✅ Gradient Boosting model working")
        print(f"✅ Forecasting engine ready")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with forecasting models: {str(e)}")
        return False

def check_app_file():
    """Verify app2.py exists and is valid"""
    print("\n🔍 Checking app2.py file...")
    
    if not Path("app2.py").exists():
        print("❌ app2.py not found!")
        return False
    
    try:
        with open("app2.py", 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for key components
        checks = {
            'Streamlit import': 'import streamlit as st',
            'Data loading': 'def load_data()',
            'Forecasting function': 'def forecast_next_12_months',
            'Page navigation': 'Sales Forecasting',
        }
        
        all_present = True
        for check_name, check_string in checks.items():
            if check_string in content:
                print(f"✅ {check_name} - Found")
            else:
                print(f"❌ {check_name} - Missing")
                all_present = False
        
        file_size = len(content) / 1024  # KB
        print(f"✅ App file size: {file_size:.2f} KB")
        
        return all_present
        
    except Exception as e:
        print(f"❌ Error reading app2.py: {str(e)}")
        return False

def print_summary(results):
    """Print final summary"""
    print("\n" + "="*60)
    print("📊 VERIFICATION SUMMARY")
    print("="*60)
    
    all_passed = all(results.values())
    
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {check}")
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 ALL CHECKS PASSED! Ready to run app2.py")
        print("\n🚀 Run the app with: streamlit run app2.py")
    else:
        print("\n⚠️  SOME CHECKS FAILED! Please fix issues before running.")
        print("\n📖 See README_APP2.md for detailed instructions")
    
    print("\n")

def main():
    """Run all verification checks"""
    print("="*60)
    print("🔬 App2.py Verification Script")
    print("="*60)
    print()
    
    results = {}
    
    # Run checks
    results['Python Version'] = check_python_version()
    results['Required Packages'] = check_packages()
    
    data_files = check_data_files()
    results['Data Files'] = data_files.get('required', False)
    
    if results['Required Packages'] and results['Data Files']:
        results['Data Loading'] = test_data_loading()
        results['Forecasting Models'] = test_model_components()
    else:
        print("\n⏭️  Skipping data and model tests (requirements not met)")
        results['Data Loading'] = False
        results['Forecasting Models'] = False
    
    results['App File'] = check_app_file()
    
    # Print summary
    print_summary(results)
    
    # Additional info
    if results.get('Data Files'):
        print("💡 TIP: All features will be available")
    elif data_files.get('required'):
        print("💡 TIP: Forecasting features will work, customer analytics disabled")
    
    return all(results.values())

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

