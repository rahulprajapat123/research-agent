"""
Setup script to initialize the RAG Research Intelligence System
"""
import os
import sys
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.10+"""
    if sys.version_info < (3, 10):
        print("ERROR: Python 3.10+ is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"OK: Python version: {sys.version.split()[0]}")
    return True


def check_env_file():
    """Check if .env file exists"""
    if not Path(".env").exists():
        print("WARN: .env file not found")
        print("Creating from .env.example...")
        
        if Path(".env.example").exists():
            import shutil
            shutil.copy(".env.example", ".env")
            print("OK: Created .env file - please edit with your credentials")
            return False
        else:
            print("ERROR: .env.example not found")
            return False
    
    print("OK: .env file exists")
    return True


def check_dependencies():
    """Check if dependencies are installed"""
    try:
        import fastapi
        import openai
        import psycopg2
        import redis
        print("OK: Core dependencies installed")
        return True
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e.name}")
        print("Run: pip install -r requirements.txt")
        return False


def check_azure_storage():
    """Check Azure Blob Storage configuration"""
    try:
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
        
        if not connection_string or not container_name:
            print("WARN: Azure Storage not configured in .env")
            print("   Add AZURE_STORAGE_CONNECTION_STRING and AZURE_STORAGE_CONTAINER_NAME")
            return False
        
        print("OK: Azure Storage credentials found")
        return True
    except Exception as e:
        print(f"WARN: Azure Storage check failed: {e}")
        return False
        print("OK: Database schema initialized")
        return True
        
    except Exception as e:
        print(f"ERROR: Schema initialization failed: {e}")
        return False


def create_directories():
    """Create required directories"""
    directories = [
        "logs",
        "documents",
        "raw_documents"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("OK: Created required directories")
    return True


def main():
    """Main setup routine"""
    print("=" * 60)
    print("RAG Research Intelligence System - Setup")
    print("=" * 60)
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("Environment File", check_env_file),
        ("Dependencies", check_dependencies),
        ("Directories", create_directories),
    ]
    
    # Run basic checks
    all_passed = True
    for name, check_func in checks:
        print(f"\nChecking: {name}")
        if not check_func():
            all_passed = False
    
    if not all_passed:
        print("\n" + "=" * 60)
        print("WARN: Setup incomplete - please fix the issues above")
        print("=" * 60)
        return False
    
    # Storage checks (optional if credentials not set yet)
    print("\n" + "-" * 60)
    print("Azure Blob Storage Setup (optional)")
    print("-" * 60)
    check_azure_storage()
    
    print("\n" + "=" * 60)
    print("OK: Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Edit .env with your OpenAI API key")
    print("2. (Optional) Add Azure Storage credentials for paper caching")
    print("3. Run: uvicorn main:app --reload")
    print("4. Access Copilot at: http://localhost:8000")
    print("5. API docs at: http://localhost:8000/docs")
    print("\nMode: Research Copilot (Database-free, using live arXiv API)")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
