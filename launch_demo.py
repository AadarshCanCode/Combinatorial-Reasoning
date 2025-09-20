#!/usr/bin/env python3
"""
CRLLM Demo Launcher

Comprehensive launcher script that provides multiple ways to run the CRLLM demos.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def print_banner():
    """Print the CRLLM banner."""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║    🚀 CRLLM: Combinatorial Reasoning with Large Language     ║
    ║                    Models - Demo Launcher                    ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'gradio',
        'matplotlib',
        'seaborn',
        'pandas',
        'numpy',
        'openai',
        'sentence-transformers',
        'scikit-learn',
        'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("   Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_api_key():
    """Check if OpenAI API key is set."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key == 'your-openai-api-key-here':
        print("⚠️  OpenAI API key not set or using placeholder value")
        print("   Set your API key:")
        print("   export OPENAI_API_KEY='your-actual-key-here'")
        return False
    print("✅ OpenAI API key is set")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True, capture_output=True, text=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def run_gradio_demo(port=7860, share=False, debug=False):
    """Run the Gradio web demo."""
    print(f"🌐 Starting Gradio demo on port {port}...")
    try:
        from gradio_demo import main as run_demo
        # Modify the demo to use custom port
        import gradio_demo
        original_launch = gradio_demo.CRLLMGradioDemo.create_interface
        
        def custom_launch(self):
            interface = original_launch(self)
            interface.launch(
                server_name="0.0.0.0",
                server_port=port,
                share=share,
                debug=debug,
                show_error=True
            )
            return interface
        
        gradio_demo.CRLLMGradioDemo.create_interface = custom_launch
        run_demo()
    except Exception as e:
        print(f"❌ Failed to start Gradio demo: {e}")
        return False

def run_command_line_demo():
    """Run the command line demo."""
    print("💻 Starting command line demo...")
    try:
        subprocess.run([sys.executable, '-m', 'crllm.main'], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start command line demo: {e}")
        return False

def run_jupyter_demo():
    """Run the Jupyter notebook demo."""
    print("📓 Starting Jupyter notebook demo...")
    try:
        subprocess.run([sys.executable, '-m', 'jupyter', 'notebook', 'examples/crllm_demo.ipynb'], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Jupyter demo: {e}")
        return False

def run_simple_demo():
    """Run the simple Python demo."""
    print("🐍 Starting simple Python demo...")
    try:
        subprocess.run([sys.executable, 'demo.py'], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start simple demo: {e}")
        return False

def run_tests():
    """Run the test suite."""
    print("🧪 Running test suite...")
    try:
        subprocess.run([sys.executable, 'test_gradio_demo.py'], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Tests failed: {e}")
        return False

def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(description="CRLLM Demo Launcher")
    parser.add_argument('--demo', choices=['gradio', 'cli', 'jupyter', 'simple', 'test'], 
                       default='gradio', help='Demo to run')
    parser.add_argument('--port', type=int, default=7860, help='Port for Gradio demo')
    parser.add_argument('--share', action='store_true', help='Create public link for Gradio demo')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--install', action='store_true', help='Install dependencies first')
    parser.add_argument('--skip-checks', action='store_true', help='Skip dependency and API key checks')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies if requested
    if args.install:
        if not install_dependencies():
            sys.exit(1)
    
    # Check dependencies
    if not args.skip_checks:
        if not check_dependencies():
            print("\n💡 Run with --install to install dependencies automatically")
            sys.exit(1)
        
        # Check API key for demos that need it
        if args.demo in ['gradio', 'cli', 'simple']:
            if not check_api_key():
                response = input("\nContinue anyway? (y/N): ").lower().strip()
                if response != 'y':
                    print("Exiting. Please set your API key and try again.")
                    sys.exit(1)
    
    print(f"\n🚀 Starting {args.demo} demo...")
    print("-" * 50)
    
    # Run the selected demo
    success = False
    
    if args.demo == 'gradio':
        success = run_gradio_demo(args.port, args.share, args.debug)
    elif args.demo == 'cli':
        success = run_command_line_demo()
    elif args.demo == 'jupyter':
        success = run_jupyter_demo()
    elif args.demo == 'simple':
        success = run_simple_demo()
    elif args.demo == 'test':
        success = run_tests()
    
    if success:
        print(f"\n✅ {args.demo} demo completed successfully!")
    else:
        print(f"\n❌ {args.demo} demo failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
