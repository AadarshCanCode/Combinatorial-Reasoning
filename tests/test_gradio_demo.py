#!/usr/bin/env python3
"""
Test script for CRQUBO Gradio Demo

This script tests the Gradio demo functionality without launching the web interface.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch, Mock

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test if all required imports work."""
    # gradio should be importable for demo tests
    import gradio as gr

    # Use the canonical lowercase package name
    from crqubo import CRLLMPipeline  # type: ignore

    from gradio_demo import CRQUBOGradioDemo  # type: ignore


def test_demo_initialization():
    """Test demo initialization."""
    from gradio_demo import CRQUBOGradioDemo  # type: ignore
    demo = CRQUBOGradioDemo()
    assert demo is not None

def test_pipeline_creation():
    """Test pipeline creation."""
    from gradio_demo import CRQUBOGradioDemo  # type: ignore
    demo = CRQUBOGradioDemo()
    result = demo.update_pipeline({})
    assert "✅" in result

def test_example_queries():
    """Test example queries generation."""
    from gradio_demo import CRQUBOGradioDemo  # type: ignore
    demo = CRQUBOGradioDemo()
    examples = demo.get_example_queries()
    assert examples and len(examples) > 0

def test_history_functionality():
    """Test history functionality."""
    from gradio_demo import CRQUBOGradioDemo  # type: ignore
    demo = CRQUBOGradioDemo()
    df = demo.get_history_dataframe()
    assert df is not None
    # Test clear history returns a tuple/status (if implemented)
    result = demo.clear_history()
    # If clear_history returns a tuple, check status inside; else, ensure history is empty
    if isinstance(result, tuple) and len(result) > 5:
        assert "✅" in result[5]
    else:
        assert demo.get_history_dataframe().empty

def test_interface_creation():
    """Test interface creation (without launching)."""
    from gradio_demo import CRQUBOGradioDemo  # type: ignore
    demo = CRQUBOGradioDemo()
    interface = demo.create_interface()
    assert interface is not None

def test_demo_processing():
    """Test demo query processing functionality."""
    from gradio_demo import CRQUBOGradioDemo  # type: ignore
    demo = CRQUBOGradioDemo()
    with patch.object(demo, 'pipeline') as mock_pipeline:
        mock_result = Mock()
        mock_result.final_answer = "Test answer"
        mock_result.confidence = 0.8
        mock_result.reasoning_chain = ["Step 1", "Step 2"]
        mock_result.metadata = {
            'domain': 'test',
            'used_retrieval': False,
            'used_verification': False
        }
        mock_pipeline.process_query.return_value = mock_result

        result = demo.process_query(
            query="Test query",
            domain="general",
            use_retrieval=False,
            use_verification=False
        )

        assert len(result) == 5
        assert result[0] == "Test answer"
        assert "Step 1" in result[1]
        assert "0.80" in result[2]

def test_demo_error_handling():
    """Test demo error handling."""
    from gradio_demo import CRQUBOGradioDemo  # type: ignore
    demo = CRQUBOGradioDemo()
    with patch.object(demo, 'pipeline') as mock_pipeline:
        mock_pipeline.process_query.side_effect = Exception("Test error")

        result = demo.process_query(
            query="Test query",
            domain="general",
            use_retrieval=False,
            use_verification=False
        )

        assert len(result) == 5
        assert "Error processing query" in result[0]
        assert result[4].startswith("❌")

def test_demo_edge_cases():
    """Test demo with edge cases."""
    from gradio_demo import CRQUBOGradioDemo  # type: ignore
    demo = CRQUBOGradioDemo()

    result = demo.process_query("", "general", False, False)
    assert "Please enter a query" in result[0]

    result = demo.process_query("   ", "general", False, False)
    assert "Please enter a query" in result[0]

if __name__ == "__main__":
    # Allow running the demo tests as a script for local checks, but prefer pytest
    import pytest
    raise SystemExit(pytest.main(["-q", "test_gradio_demo.py"]))
