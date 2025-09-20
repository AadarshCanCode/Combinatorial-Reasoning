#!/usr/bin/env python3
"""
CRLLM Gradio Demo Launcher

Simple launcher script for the CRLLM Gradio demo.
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
    if not api_key or api_key == 'your-openai-api-key-here':
        print("⚠️  Warning: OPENAI_API_KEY not set or using placeholder value.")
        print("   Set your API key: export OPENAI_API_KEY='your-actual-key-here'")
        print("   Or edit the gradio_demo.py file to set it directly.")
        return False
    else:
        print("✅ OpenAI API key is set.")
        return True

def main():
    """Main launcher function."""
    print("🚀 CRLLM Gradio Demo Launcher")
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
