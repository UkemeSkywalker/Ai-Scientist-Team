#!/usr/bin/env python3
"""
Sample installation script for Amazon Bedrock Strands SDK
Note: This SDK is not yet publicly available - this is a preparation script
"""

import subprocess
import sys
import os
from pathlib import Path

def install_strands_sdk():
    """
    Install Amazon Bedrock Strands SDK when it becomes available
    """
    print("🔧 Installing Amazon Bedrock Strands SDK...")
    
    # Ensure we're in the python directory
    os.chdir(Path(__file__).parent)
    
    # Activate virtual environment if it exists
    venv_path = Path("venv")
    if venv_path.exists():
        print("📦 Using existing virtual environment")
        if sys.platform == "win32":
            activate_script = venv_path / "Scripts" / "activate"
        else:
            activate_script = venv_path / "bin" / "activate"
    else:
        print("📦 Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
    
    # Install AWS CLI and boto3 first (prerequisites)
    print("🔧 Installing AWS prerequisites...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--upgrade",
        "boto3>=1.34.0",
        "botocore>=1.34.0", 
        "awscli>=2.0.0"
    ], check=True)
    
    # When Strands SDK becomes available, it will likely be installed like this:
    print("🔧 Installing Bedrock Strands SDK (when available)...")
    try:
        # This is the expected installation command (not yet available)
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "amazon-bedrock-strands-sdk"
        ], check=True)
        print("✅ Strands SDK installed successfully!")
    except subprocess.CalledProcessError:
        print("⚠️  Strands SDK not yet available publicly")
        print("   Installing mock dependencies for development...")
        
        # Install development dependencies instead
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "pydantic>=2.0.0",
            "asyncio-mqtt",
            "structlog",
            "python-dotenv"
        ], check=True)
    
    # Verify installation
    print("🔍 Verifying installation...")
    try:
        import boto3
        print(f"✅ boto3 version: {boto3.__version__}")
        
        # When Strands is available, this will work:
        # import bedrock_strands
        # print(f"✅ Strands SDK version: {bedrock_strands.__version__}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    print("🎉 Installation complete!")
    return True

def setup_aws_credentials():
    """
    Guide user through AWS credentials setup
    """
    print("\n🔐 AWS Credentials Setup")
    print("You'll need AWS credentials to use Bedrock services:")
    print("1. AWS Access Key ID")
    print("2. AWS Secret Access Key") 
    print("3. AWS Region (e.g., us-east-1)")
    
    # Check if credentials already exist
    aws_dir = Path.home() / ".aws"
    credentials_file = aws_dir / "credentials"
    
    if credentials_file.exists():
        print("✅ AWS credentials file found")
    else:
        print("⚠️  No AWS credentials found")
        print("Run: aws configure")
        print("Or set environment variables:")
        print("export AWS_ACCESS_KEY_ID=your_key")
        print("export AWS_SECRET_ACCESS_KEY=your_secret")
        print("export AWS_DEFAULT_REGION=us-east-1")

def update_requirements():
    """
    Update requirements.txt with Strands SDK dependencies
    """
    print("\n📝 Updating requirements.txt...")
    
    requirements_content = """# Core dependencies
pydantic>=2.0.0
asyncio>=3.4.3
structlog>=23.0.0
python-dotenv>=1.0.0

# AWS dependencies
boto3>=1.34.0
botocore>=1.34.0

# Bedrock Strands SDK (when available)
# amazon-bedrock-strands-sdk>=1.0.0

# Development dependencies
pytest>=7.0.0
pytest-asyncio>=0.21.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.0.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements_content)
    
    print("✅ requirements.txt updated")

if __name__ == "__main__":
    print("🚀 Amazon Bedrock Strands SDK Installation")
    print("=" * 50)
    
    try:
        install_strands_sdk()
        setup_aws_credentials()
        update_requirements()
        
        print("\n🎯 Next Steps:")
        print("1. Configure AWS credentials: aws configure")
        print("2. Enable Bedrock access in AWS Console")
        print("3. Update your .env file with AWS settings")
        print("4. Test connection with: python test_strands_connection.py")
        
    except Exception as e:
        print(f"❌ Installation failed: {e}")
        sys.exit(1)