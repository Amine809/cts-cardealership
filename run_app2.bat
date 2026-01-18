@echo off
REM ============================================
REM Qatar Auto Dealer AI Platform - App2 Launcher
REM ============================================

echo.
echo ==========================================
echo  Qatar Dealer AI Platform Pro (App2)
echo ==========================================
echo.

REM Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please run setup first or activate venv manually.
    echo.
    pause
    exit /b 1
)

echo [1/3] Activating virtual environment...
call venv\Scripts\activate.bat

echo [2/3] Checking data files...
if not exist "qatar_auto_dealer_sales_forecasting_dataset_2026_final_external_factors.csv" (
    echo [WARNING] Main dataset not found!
    echo Please ensure 'qatar_auto_dealer_sales_forecasting_dataset_2026_final_external_factors.csv' exists.
    echo.
    pause
    exit /b 1
)

echo [3/3] Starting App2.py...
echo.
echo ==========================================
echo  App will open in your browser shortly
echo  URL: http://localhost:8501
echo ==========================================
echo.
echo Press Ctrl+C to stop the application
echo.

streamlit run app2.py

pause

