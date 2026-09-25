"""
Environment Configuration Management for Cloud Deployment
========================================================

Handles environment variables, API keys, and deployment configurations
with fallbacks for different cloud providers.
"""

import os
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

class Config:
    """
    Centralized configuration management for the Agentic Loan Decision System.
    Handles environment variables with proper defaults and validation.
    """
    
    # ============================================================================
    # LLM Provider Configuration
    # ============================================================================
    
    # Primary LLM (Groq)
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    GROQ_MAX_TOKENS: int = int(os.getenv("GROQ_MAX_TOKENS", "8192"))
    GROQ_TEMPERATURE: float = float(os.getenv("GROQ_TEMPERATURE", "0.1"))
    
    # Fallback LLMs
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    
    # ============================================================================
    # Vector Database Configuration
    # ============================================================================
    
    # Local ChromaDB
    CHROMA_HOST: str = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", "8000"))
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "rbi_guidelines")
    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    
    # ChromaDB Cloud (optional)
    CHROMA_CLOUD_API_KEY: str = os.getenv("CHROMA_CLOUD_API_KEY", "")
    CHROMA_CLOUD_TENANT: str = os.getenv("CHROMA_CLOUD_TENANT", "")
    CHROMA_CLOUD_DATABASE: str = os.getenv("CHROMA_CLOUD_DATABASE", "")
    
    # ============================================================================
    # HuggingFace Configuration
    # ============================================================================
    
    HUGGINGFACE_API_TOKEN: str = os.getenv("HUGGINGFACE_API_TOKEN", "")
    HUGGINGFACE_REPO_ID: str = os.getenv("HUGGINGFACE_REPO_ID", "")
    
    # ============================================================================
    # Application Configuration
    # ============================================================================
    
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # ============================================================================
    # Security Configuration
    # ============================================================================
    
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev_secret_change_in_production")
    ALLOWED_HOSTS: List[str] = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
    CORS_ORIGINS: List[str] = os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")
    
    # ============================================================================
    # Model and Data Configuration
    # ============================================================================
    
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "./models")
    DATA_CACHE_DIR: str = os.getenv("DATA_CACHE_DIR", "./data")
    MAX_MODEL_SIZE_MB: int = int(os.getenv("MAX_MODEL_SIZE_MB", "500"))
    MODEL_TIMEOUT_SECONDS: int = int(os.getenv("MODEL_TIMEOUT_SECONDS", "30"))
    
    # ============================================================================
    # API Rate Limiting
    # ============================================================================
    
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    RATE_LIMIT_PER_HOUR: int = int(os.getenv("RATE_LIMIT_PER_HOUR", "1000"))
    
    # ============================================================================
    # File Upload Configuration
    # ============================================================================
    
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
    ALLOWED_FILE_TYPES: List[str] = os.getenv("ALLOWED_FILE_TYPES", "pdf,csv,json").split(",")
    
    # ============================================================================
    # Validation and Derived Properties
    # ============================================================================
    
    @classmethod
    def validate_required_keys(cls) -> Dict[str, bool]:
        """Validate that required API keys are present"""
        validations = {
            "groq_api_key": bool(cls.GROQ_API_KEY),
            "has_llm_provider": bool(cls.GROQ_API_KEY or cls.OPENAI_API_KEY or cls.ANTHROPIC_API_KEY),
            "chroma_configured": bool(cls.CHROMA_HOST and cls.CHROMA_PORT),
        }
        return validations
    
    @classmethod
    def get_primary_llm_config(cls) -> Dict[str, Any]:
        """Get the configuration for the primary LLM provider"""
        if cls.GROQ_API_KEY:
            return {
                "provider": "groq",
                "api_key": cls.GROQ_API_KEY,
                "model": cls.GROQ_MODEL,
                "max_tokens": cls.GROQ_MAX_TOKENS,
                "temperature": cls.GROQ_TEMPERATURE,
            }
        elif cls.OPENAI_API_KEY:
            return {
                "provider": "openai",
                "api_key": cls.OPENAI_API_KEY,
                "model": "gpt-3.5-turbo",
                "max_tokens": 4096,
                "temperature": 0.1,
            }
        elif cls.ANTHROPIC_API_KEY:
            return {
                "provider": "anthropic",
                "api_key": cls.ANTHROPIC_API_KEY,
                "model": "claude-3-haiku-20240307",
                "max_tokens": 4096,
                "temperature": 0.1,
            }
        else:
            logger.warning("No LLM API key configured, using mock LLM")
            return {
                "provider": "mock",
                "api_key": "mock",
                "model": "mock-model",
                "max_tokens": 1000,
                "temperature": 0.1,
            }
    
    @classmethod
    def get_chroma_config(cls) -> Dict[str, Any]:
        """Get ChromaDB configuration (cloud or local)"""
        if cls.CHROMA_CLOUD_API_KEY and cls.CHROMA_CLOUD_TENANT:
            return {
                "mode": "cloud",
                "api_key": cls.CHROMA_CLOUD_API_KEY,
                "tenant": cls.CHROMA_CLOUD_TENANT,
                "database": cls.CHROMA_CLOUD_DATABASE,
                "collection_name": cls.CHROMA_COLLECTION_NAME,
            }
        else:
            return {
                "mode": "local",
                "host": cls.CHROMA_HOST,
                "port": cls.CHROMA_PORT,
                "persist_directory": cls.CHROMA_PERSIST_DIRECTORY,
                "collection_name": cls.CHROMA_COLLECTION_NAME,
            }
    
    @classmethod
    def is_production(cls) -> bool:
        """Check if running in production environment"""
        return cls.ENVIRONMENT.lower() == "production"
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure required directories exist"""
        directories = [
            cls.MODEL_CACHE_DIR,
            cls.DATA_CACHE_DIR,
            cls.CHROMA_PERSIST_DIRECTORY,
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")

# Initialize configuration
config = Config()

# Ensure required directories exist
config.ensure_directories()

# Log configuration status
if __name__ == "__main__":
    print("🔧 Configuration Status:")
    validations = config.validate_required_keys()
    for key, status in validations.items():
        emoji = "✅" if status else "❌"
        print(f"  {emoji} {key}: {status}")
    
    print(f"\n🤖 Primary LLM: {config.get_primary_llm_config()['provider']}")
    print(f"🗄️ ChromaDB Mode: {config.get_chroma_config()['mode']}")
    print(f"🌍 Environment: {config.ENVIRONMENT}")