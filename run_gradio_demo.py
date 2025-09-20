#!/usr/bin/env python3
"""
CRQUBO Gradio Demo Launcher

Simple launcher script for the CRQUBO Gradio demo.
"""

import os
import sys
from pathlib import Path

def check_requirements():
    """Check if required packages are installed."""
    try:
        import gradio
        import matplotlib
        import seaborn
        import pandas
        import numpy
        print("✅ All required packages are installed.")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def check_api_key():
    """Check if OpenAI API key is set."""
    api_key = os.getenv('OPENAI_API_KEY')
    # This check is advisory — CRQUBO supports modular backends.
    # If you are using a non-OpenAI backend (local model, HF endpoint, Anthropic, etc.),
    # ensure your backend is configured in `config.json` or via environment variables.
    if not api_key or api_key == 'your-openai-api-key-here':
        print("⚠️  OPENAI_API_KEY not set or using placeholder value. If you intend to use OpenAI, set the key.")
        print("   Example (Linux/macOS): export OPENAI_API_KEY='your-actual-key-here'")
        print("   Example (Windows PowerShell): setx OPENAI_API_KEY 'your-actual-key-here'")
        # We do not force exit here because other backends may be configured.
        return False
    else:
        print("✅ OpenAI API key is set.")
        return True

def main():
    """Main launcher function."""
    print("🚀 CRQUBO Gradio Demo Launcher")
    print("=" * 40)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check API key
    api_key_ok = check_api_key()
    if not api_key_ok:
        response = input("\nContinue anyway? (y/N): ").lower().strip()
        if response != 'y':
            print("Exiting. Please set your API key and try again.")
            sys.exit(1)
    
    print("\n🌐 Starting Gradio demo...")
    print("   The demo will open in your browser automatically.")
    print("   Press Ctrl+C to stop the demo.")
    print("-" * 40)
    
    try:
        # Import and run the demo
        from gradio_demo import main as run_demo
        run_demo()
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user.")
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
