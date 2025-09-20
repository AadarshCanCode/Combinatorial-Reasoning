#!/usr/bin/env python3
"""
Simple CRLLM Example

This script demonstrates basic usage of the CRLLM framework
with a simple causal reasoning example.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import crllm
sys.path.insert(0, str(Path(__file__).parent.parent))

from crllm import CRLLMPipeline


def main():
    """Run a simple CRLLM example."""
    print("CRLLM Simple Example")
    print("=" * 50)
    
    # Set up API key (replace with your actual key)
    os.environ['OPENAI_API_KEY'] = 'your-openai-api-key-here'
    
    # Create a basic pipeline
    pipeline = CRLLMPipeline()
    
    # Example query
    query = "Why does smoking cause lung cancer?"
    
    print(f"Query: {query}")
    print("Processing...")
    
    try:
        # Process the query
        result = pipeline.process_query(
            query=query,
            domain="causal",
            use_retrieval=False,
            use_verification=True
        )
        
        # Display results
        print(f"\nAnswer: {result.final_answer}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Reasoning steps: {len(result.reasoning_chain)}")
        
        print("\nReasoning Chain:")
        for i, step in enumerate(result.reasoning_chain, 1):
            print(f"{i}. {step}")
        
        print(f"\nMetadata: {result.metadata}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to set your OpenAI API key!")


if __name__ == "__main__":
    main()
