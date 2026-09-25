#!/bin/bash

# Agentic Loan Decision System - Setup Script
# This script automates the complete setup process for the loan approval system

set -e  # Exit on any error

echo "🚀 Setting up Agentic Loan Decision System..."

# Detect Python command (Windows vs Unix)
PYTHON_CMD=""
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "❌ Error: Python is not installed. Please install Python 3.8+ and try again."
    exit 1
fi

echo "📦 Installing Python dependencies..."
$PYTHON_CMD -m pip install --upgrade pip
$PYTHON_CMD -m pip install -r requirements.txt

echo "🔧 Building ML artifacts (models, scalers, explainers)..."
$PYTHON_CMD pipeline/build_artifacts.py

echo "📚 Ingesting regulatory documents into RAG system..."
$PYTHON_CMD pipeline/ingest_rag.py

echo "🧪 Running tests to verify system integrity..."
$PYTHON_CMD -m pytest tests/ -v

echo "✅ Setup complete! System is ready for use."
echo ""
echo "🎯 Quick Start Commands:"
echo "  Start API server:    $PYTHON_CMD -m uvicorn api.app:app --reload --port 8000"
echo "  Start UI dashboard:  $PYTHON_CMD -m streamlit run frontend/streamlit_app.py"
echo "  Run tests:          $PYTHON_CMD -m pytest tests/ -v"
echo ""
echo "📖 See README.md for detailed usage instructions."