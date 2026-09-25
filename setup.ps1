# Agentic Loan Decision System - PowerShell Setup Script
# Cross-platform automation for Windows development environments

param(
    [Parameter(Position=0)]
    [ValidateSet("init", "api", "ui", "test", "clean", "help", "")]
    [string]$Command = "help"
)

function Show-Help {
    Write-Host "Agentic Loan Decision System - Available Commands:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  .\setup.ps1 init    - Complete system initialization (dependencies + artifacts + RAG)" -ForegroundColor Green
    Write-Host "  .\setup.ps1 api     - Start FastAPI backend server on port 8000" -ForegroundColor Green
    Write-Host "  .\setup.ps1 ui      - Start Streamlit dashboard on port 8501" -ForegroundColor Green
    Write-Host "  .\setup.ps1 test    - Run pytest test suite" -ForegroundColor Green
    Write-Host "  .\setup.ps1 clean   - Clean generated files and caches" -ForegroundColor Green
    Write-Host ""
    Write-Host "For detailed usage, see README.md" -ForegroundColor Yellow
}

function Initialize-System {
    Write-Host "Initializing Agentic Loan Decision System..." -ForegroundColor Cyan
    
    # Install dependencies
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    
    # Build artifacts
    Write-Host "Building ML artifacts..." -ForegroundColor Yellow
    python pipeline/build_artifacts.py
    
    # Ingest documents
    Write-Host "Ingesting RAG documents..." -ForegroundColor Yellow
    python pipeline/ingest_rag.py
    
    Write-Host "Initialization complete!" -ForegroundColor Green
}

function Start-API {
    Write-Host "Starting FastAPI backend on http://localhost:8000" -ForegroundColor Cyan
    Write-Host "API documentation: http://localhost:8000/docs" -ForegroundColor Yellow
    uvicorn api.app:app --reload --port 8000
}

function Start-UI {
    Write-Host "Starting Streamlit dashboard on http://localhost:8501" -ForegroundColor Cyan
    streamlit run frontend/streamlit_app.py
}

function Run-Tests {
    Write-Host "Running pytest test suite..." -ForegroundColor Cyan
    python -m pytest tests/ -v --tb=short
}

function Clean-Files {
    Write-Host "Cleaning generated files and caches..." -ForegroundColor Cyan
    
    # Remove Python cache directories
    Get-ChildItem -Path . -Recurse -Name "__pycache__" | ForEach-Object { 
        Remove-Item -Path $_ -Recurse -Force -ErrorAction SilentlyContinue 
    }
    
    # Remove pytest cache
    Get-ChildItem -Path . -Recurse -Name ".pytest_cache" | ForEach-Object { 
        Remove-Item -Path $_ -Recurse -Force -ErrorAction SilentlyContinue 
    }
    
    # Remove .pyc files
    Get-ChildItem -Path . -Recurse -Name "*.pyc" | ForEach-Object { 
        Remove-Item -Path $_ -Force -ErrorAction SilentlyContinue 
    }
    
    Write-Host "Cleanup complete!" -ForegroundColor Green
}

# Main command dispatcher
switch ($Command) {
    "init" { Initialize-System }
    "api" { Start-API }
    "ui" { Start-UI }
    "test" { Run-Tests }
    "clean" { Clean-Files }
    "help" { Show-Help }
    "" { Show-Help }
    default { 
        Write-Host "Unknown command: $Command" -ForegroundColor Red
        Show-Help 
    }
}