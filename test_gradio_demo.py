#!/usr/bin/env python3
"""
Test script for CRLLM Gradio Demo

This script tests the Gradio demo functionality without launching the web interface.
"""

import os
import sys
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test if all required imports work."""
    print("Testing imports...")
    
    try:
        import gradio as gr
        print("✅ Gradio imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Gradio: {e}")
        return False
    
    try:
        from crllm import CRLLMPipeline
        print("✅ CRLLM imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import CRLLM: {e}")
        return False
    
    try:
        from gradio_demo import CRLLMGradioDemo
        print("✅ Gradio demo imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Gradio demo: {e}")
        return False
    
    return True

def test_demo_initialization():
    """Test demo initialization."""
    print("\nTesting demo initialization...")
    
    try:
        from gradio_demo import CRLLMGradioDemo
        demo = CRLLMGradioDemo()
        print("✅ Demo initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize demo: {e}")
        return False

def test_pipeline_creation():
    """Test pipeline creation."""
    print("\nTesting pipeline creation...")
    
    try:
        from gradio_demo import CRLLMGradioDemo
        demo = CRLLMGradioDemo()
        
        # Test with empty config
        result = demo.update_pipeline({})
        if "✅" in result:
            print("✅ Pipeline created successfully")
            return True
        else:
            print(f"❌ Pipeline creation failed: {result}")
            return False
    except Exception as e:
        print(f"❌ Failed to create pipeline: {e}")
        return False

def test_example_queries():
    """Test example queries generation."""
    print("\nTesting example queries...")
    
    try:
        from gradio_demo import CRLLMGradioDemo
        demo = CRLLMGradioDemo()
        
        examples = demo.get_example_queries()
        if examples and len(examples) > 0:
            print(f"✅ Generated {len(examples)} example queries")
            return True
        else:
            print("❌ No example queries generated")
            return False
    except Exception as e:
        print(f"❌ Failed to generate example queries: {e}")
        return False

def test_history_functionality():
    """Test history functionality."""
    print("\nTesting history functionality...")
    
    try:
        from gradio_demo import CRLLMGradioDemo
        demo = CRLLMGradioDemo()
        
        # Test history dataframe
        df = demo.get_history_dataframe()
        if df is not None:
            print("✅ History dataframe created successfully")
        else:
            print("❌ Failed to create history dataframe")
            return False
        
        # Test clear history
        result = demo.clear_history()
        if "✅" in result[5]:  # Status is the 6th element
            print("✅ History cleared successfully")
            return True
        else:
            print("❌ Failed to clear history")
            return False
    except Exception as e:
        print(f"❌ Failed to test history functionality: {e}")
        return False

def test_interface_creation():
    """Test interface creation (without launching)."""
    print("\nTesting interface creation...")
    
    try:
        from gradio_demo import CRLLMGradioDemo
        demo = CRLLMGradioDemo()
        
        # This will create the interface but not launch it
        interface = demo.create_interface()
        if interface is not None:
            print("✅ Interface created successfully")
            return True
        else:
            print("❌ Failed to create interface")
            return False
    except Exception as e:
        print(f"❌ Failed to create interface: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 CRLLM Gradio Demo Test Suite")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_demo_initialization,
        test_pipeline_creation,
        test_example_queries,
        test_history_functionality,
        test_interface_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The Gradio demo should work correctly.")
        print("\nTo run the demo:")
        print("  python run_gradio_demo.py")
        print("  or")
        print("  python gradio_demo.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("  1. Install requirements: pip install -r requirements.txt")
        print("  2. Set API key: export OPENAI_API_KEY='your-key-here'")
        print("  3. Check Python version (3.8+ required)")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
