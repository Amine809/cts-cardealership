# ============================================
# Qatar Auto Dealer AI Platform - App2 Launcher (PowerShell)
# ============================================

Write-Host ""
Write-Host "=========================================="
Write-Host " Qatar Dealer AI Platform Pro (App2)"
Write-Host "=========================================="
Write-Host ""

# Check if venv exists
if (-not (Test-Path "venv\Scripts\Activate.ps1")) {
    Write-Host "[ERROR] Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run setup first or activate venv manually."
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "[1/3] Activating virtual environment..." -ForegroundColor Cyan
. .\venv\Scripts\Activate.ps1

Write-Host "[2/3] Checking data files..." -ForegroundColor Cyan
if (-not (Test-Path "qatar_auto_dealer_sales_forecasting_dataset_2026_final_external_factors.csv")) {
    Write-Host "[WARNING] Main dataset not found!" -ForegroundColor Yellow
    Write-Host "Please ensure 'qatar_auto_dealer_sales_forecasting_dataset_2026_final_external_factors.csv' exists."
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "[3/3] Starting App2.py..." -ForegroundColor Cyan
Write-Host ""
Write-Host "=========================================="
Write-Host " App will open in your browser shortly" -ForegroundColor Green
Write-Host " URL: http://localhost:8501"
Write-Host "=========================================="
Write-Host ""
Write-Host "Press Ctrl+C to stop the application" -ForegroundColor Yellow
Write-Host ""

streamlit run app2.py

