#!/usr/bin/env python3
"""
Setup script for Document Classification System
Automates initial setup and validation
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def print_step(step_num, total_steps, description):
    """Print formatted step information"""
    print(f"\n[{step_num}/{total_steps}] {description}")
    print("-" * 50)


def check_python_version():
    """Check Python version requirements"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True


def create_directories():
    """Create necessary directories"""
    directories = ['models', 'data', 'logs', 'saved_models']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}/")
    
    return True


def setup_environment_file():
    """Setup .env file from template"""
    env_template = Path('env_template.txt')
    env_file = Path('.env')
    
    if not env_template.exists():
        print("❌ env_template.txt not found")
        return False
    
    if not env_file.exists():
        shutil.copy(env_template, env_file)
        print("✅ Created .env file from template")
        print("⚠️  IMPORTANT: Edit .env file with your GROQ_API_KEY")
        return True
    else:
        print("✅ .env file already exists")
        return True


def install_dependencies():
    """Install Python dependencies"""
    try:
        print("Installing dependencies...")
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print("❌ Failed to install dependencies:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False


def download_nltk_data():
    """Download required NLTK data"""
    try:
        import nltk
        print("Downloading NLTK data...")
        
        nltk_downloads = ['punkt', 'stopwords', 'wordnet']
        for item in nltk_downloads:
            try:
                nltk.data.find(f'tokenizers/{item}')
                print(f"✅ NLTK {item} already available")
            except LookupError:
                nltk.download(item, quiet=True)
                print(f"✅ Downloaded NLTK {item}")
        
        return True
    except ImportError:
        print("❌ NLTK not installed")
        return False
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")
        return False


def check_dataset():
    """Check if dataset exists"""
    dataset_file = Path('business_documents_dataset_cleaned.csv')
    
    if dataset_file.exists():
        file_size = dataset_file.stat().st_size / (1024 * 1024)  # MB
        print(f"✅ Dataset found: {file_size:.1f}MB")
        return True
    else:
        print("⚠️  Dataset not found: business_documents_dataset_cleaned.csv")
        print("   You'll need this file to train ML models")
        return False


def validate_environment():
    """Validate environment configuration"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        # Check for API key
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key or api_key == 'your-groq-api-key-here':
            print("⚠️  GROQ_API_KEY not set properly in .env file")
            print("   Get your free API key at: https://console.groq.com")
            return False
        else:
            # Mask API key for display
            masked_key = api_key[:8] + '...' + api_key[-4:] if len(api_key) > 12 else 'set'
            print(f"✅ GROQ_API_KEY configured: {masked_key}")
            return True
    
    except ImportError:
        print("❌ python-dotenv not installed")
        return False
    except Exception as e:
        print(f"❌ Error validating environment: {e}")
        return False


def test_imports():
    """Test critical imports"""
    critical_imports = [
        'numpy',
        'pandas', 
        'sklearn',
        'langchain',
        'requests'
    ]
    
    success = True
    for module in critical_imports:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} - not installed")
            success = False
    
    return success


def run_basic_tests():
    """Run basic system tests"""
    try:
        print("Testing basic functionality...")
        
        # Test configuration
        from config_manager import ConfigManager
        config_manager = ConfigManager()
        if config_manager.validate_config():
            print("✅ Configuration validation passed")
        else:
            print("⚠️  Configuration validation failed")
            return False
        
        # Test feature extraction
        from learned_routing import FastFeatureExtractor
        from base_classifier import DocumentMetadata
        
        extractor = FastFeatureExtractor()
        features = extractor.extract(
            "Test document for feature extraction", 
            DocumentMetadata(file_extension="pdf")
        )
        print("✅ Feature extraction working")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic tests failed: {e}")
        return False


def print_next_steps():
    """Print next steps for user"""
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETE!")
    print("=" * 60)
    
    print("\n📋 NEXT STEPS:")
    print("\n1. Configure API Key (if not done):")
    print("   - Edit .env file")
    print("   - Set GROQ_API_KEY=your-actual-key")
    print("   - Get free key at: https://console.groq.com")
    
    print("\n2. Train ML Models:")
    print("   python document_classifier.py")
    
    print("\n3. Test LLM Integration:")
    print("   python test_llm.py")
    
    print("\n4. Run Complete System:")
    print("   python agent.py")
    
    print("\n5. Run All Tests:")
    print("   python test_phase1.py")
    print("   python test_phase2.py")
    
    print("\n📚 Documentation:")
    print("   - README_COMPLETE.md - Full documentation")
    print("   - README_LLM.md - LLM-specific guide")
    
    print("\n🔧 Troubleshooting:")
    print("   - Check logs/ directory for error details")
    print("   - Run with DEBUG logging: export LOG_LEVEL=DEBUG")
    
    print("\n💡 Pro Tips:")
    print("   - Start with small documents to test")
    print("   - Monitor routing decisions with agent.get_system_statistics()")
    print("   - Add user corrections to improve accuracy")


def main():
    """Main setup function"""
    print("🚀 Document Classification System Setup")
    print("=" * 60)
    
    total_steps = 8
    current_step = 0
    
    # Step 1: Check Python version
    current_step += 1
    print_step(current_step, total_steps, "Checking Python version")
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Create directories
    current_step += 1
    print_step(current_step, total_steps, "Creating directories")
    create_directories()
    
    # Step 3: Setup environment file
    current_step += 1
    print_step(current_step, total_steps, "Setting up environment file")
    setup_environment_file()
    
    # Step 4: Install dependencies
    current_step += 1
    print_step(current_step, total_steps, "Installing dependencies")
    if not install_dependencies():
        print("⚠️  You may need to install dependencies manually:")
        print("   pip install -r requirements.txt")
    
    # Step 5: Download NLTK data
    current_step += 1
    print_step(current_step, total_steps, "Downloading NLTK data")
    download_nltk_data()
    
    # Step 6: Check dataset
    current_step += 1
    print_step(current_step, total_steps, "Checking dataset")
    check_dataset()
    
    # Step 7: Validate environment
    current_step += 1
    print_step(current_step, total_steps, "Validating environment")
    validate_environment()
    
    # Step 8: Test imports and basic functionality
    current_step += 1
    print_step(current_step, total_steps, "Testing system components")
    if not test_imports():
        print("⚠️  Some imports failed. Check dependency installation.")
    
    if not run_basic_tests():
        print("⚠️  Basic tests failed. Check configuration.")
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main()