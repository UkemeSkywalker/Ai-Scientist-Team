#!/usr/bin/env python3
"""
Setup script for Kaggle API credentials
Helps users configure Kaggle API for the Data Agent
"""

import os
import json
from pathlib import Path

def setup_kaggle_credentials():
    """Interactive setup for Kaggle API credentials"""
    print("🔧 Kaggle API Setup")
    print("=" * 50)
    
    # Check if kaggle.json already exists
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if kaggle_json.exists():
        print("✅ Kaggle credentials already configured!")
        print(f"   Found: {kaggle_json}")
        
        # Test the credentials
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            print("✅ Kaggle API authentication successful!")
            return True
        except Exception as e:
            print(f"❌ Kaggle API authentication failed: {str(e)}")
            print("   Your credentials may be invalid or expired.")
    
    print("\n📋 To set up Kaggle API credentials:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Scroll to 'API' section")
    print("3. Click 'Create New API Token'")
    print("4. Download the kaggle.json file")
    print("5. Choose setup method below:")
    
    print("\n🔧 Setup Options:")
    print("1. Automatic setup (provide username and key)")
    print("2. Manual setup (place kaggle.json file)")
    print("3. Environment variables")
    
    choice = input("\nChoose option (1-3): ").strip()
    
    if choice == "1":
        return setup_automatic()
    elif choice == "2":
        return setup_manual()
    elif choice == "3":
        return setup_environment()
    else:
        print("❌ Invalid choice")
        return False

def setup_automatic():
    """Automatic setup by asking for username and key"""
    print("\n🔧 Automatic Setup")
    print("-" * 30)
    
    username = input("Enter your Kaggle username: ").strip()
    if not username:
        print("❌ Username is required")
        return False
    
    key = input("Enter your Kaggle API key: ").strip()
    if not key:
        print("❌ API key is required")
        return False
    
    # Create .kaggle directory
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    
    # Create kaggle.json
    kaggle_json = kaggle_dir / "kaggle.json"
    credentials = {
        "username": username,
        "key": key
    }
    
    try:
        with open(kaggle_json, 'w') as f:
            json.dump(credentials, f, indent=2)
        
        # Set proper permissions (600)
        os.chmod(kaggle_json, 0o600)
        
        print(f"✅ Kaggle credentials saved to: {kaggle_json}")
        
        # Test the credentials
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            print("✅ Kaggle API authentication successful!")
            return True
        except Exception as e:
            print(f"❌ Kaggle API authentication failed: {str(e)}")
            print("   Please check your username and key.")
            return False
            
    except Exception as e:
        print(f"❌ Failed to save credentials: {str(e)}")
        return False

def setup_manual():
    """Manual setup instructions"""
    print("\n🔧 Manual Setup")
    print("-" * 30)
    
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    print(f"1. Create directory: {kaggle_dir}")
    print(f"2. Place your downloaded kaggle.json file in: {kaggle_json}")
    print("3. Set proper permissions:")
    print(f"   chmod 600 {kaggle_json}")
    
    input("\nPress Enter when you've completed the setup...")
    
    # Check if file exists now
    if kaggle_json.exists():
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            print("✅ Kaggle API authentication successful!")
            return True
        except Exception as e:
            print(f"❌ Kaggle API authentication failed: {str(e)}")
            return False
    else:
        print(f"❌ kaggle.json not found at: {kaggle_json}")
        return False

def setup_environment():
    """Environment variables setup"""
    print("\n🔧 Environment Variables Setup")
    print("-" * 30)
    
    print("Add these environment variables to your shell profile:")
    print("export KAGGLE_USERNAME='your_username'")
    print("export KAGGLE_KEY='your_api_key'")
    print("\nOr add them to your .env file:")
    print("KAGGLE_USERNAME=your_username")
    print("KAGGLE_KEY=your_api_key")
    
    username = os.getenv('KAGGLE_USERNAME')
    key = os.getenv('KAGGLE_KEY')
    
    if username and key:
        print(f"✅ Found environment variables:")
        print(f"   KAGGLE_USERNAME: {username}")
        print(f"   KAGGLE_KEY: {'*' * len(key)}")
        
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            print("✅ Kaggle API authentication successful!")
            return True
        except Exception as e:
            print(f"❌ Kaggle API authentication failed: {str(e)}")
            return False
    else:
        print("❌ Environment variables not found")
        print("   Please set KAGGLE_USERNAME and KAGGLE_KEY")
        return False

def main():
    """Main setup function"""
    try:
        # Check if kaggle package is installed
        try:
            import kaggle
            print("✅ Kaggle package is installed")
        except ImportError:
            print("❌ Kaggle package not installed")
            print("   Install with: pip install kaggle")
            return
        
        # Run setup
        success = setup_kaggle_credentials()
        
        if success:
            print("\n🎉 Kaggle API setup completed successfully!")
            print("   You can now use the Data Agent with real Kaggle datasets.")
        else:
            print("\n❌ Kaggle API setup failed")
            print("   Please check the instructions and try again.")
            
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
    except Exception as e:
        print(f"\n❌ Setup failed with error: {str(e)}")

if __name__ == "__main__":
    main()