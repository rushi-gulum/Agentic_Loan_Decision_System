# Agentic Loan Decision System - Makefile
# Cross-platform automation for development and deployment

.PHONY: help init api ui test clean setup

# Default target
help:
	@echo "🚀 Agentic Loan Decision System - Available Commands:"
	@echo ""
	@echo "  make init    - Complete system initialization (dependencies + artifacts + RAG)"
	@echo "  make api     - Start FastAPI backend server on port 8000"
	@echo "  make ui      - Start Streamlit dashboard on port 8501"
	@echo "  make test    - Run pytest test suite"
	@echo "  make setup   - Run automated setup script"
	@echo "  make clean   - Clean generated files and caches"
	@echo ""
	@echo "📖 For detailed usage, see README.md"

# Complete system initialization
init:
	@echo "🔧 Initializing Agentic Loan Decision System..."
	@python -m pip install --upgrade pip
	@python -m pip install -r requirements.txt
	@python pipeline/build_artifacts.py
	@python pipeline/ingest_rag.py
	@echo "✅ Initialization complete!"

# Start API server
api:
	@echo "🌐 Starting FastAPI backend on http://localhost:8000"
	@echo "📋 API documentation: http://localhost:8000/docs"
	@uvicorn api.app:app --reload --port 8000

# Start UI dashboard
ui:
	@echo "📊 Starting Streamlit dashboard on http://localhost:8501"
	@streamlit run frontend/streamlit_app.py

# Run tests
test:
	@echo "🧪 Running pytest test suite..."
	@python -m pytest tests/ -v --tb=short

# Run setup script (alternative to init)
setup:
	@chmod +x setup.sh 2>/dev/null || true
	@./setup.sh

# Clean generated files
clean:
	@echo "🧹 Cleaning generated files and caches..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cleanup complete!"