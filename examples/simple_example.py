#!/usr/bin/env python3
"""
Simple CRQUBO Example

This script demonstrates basic usage of the CRQUBO framework
with a simple causal reasoning example.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import crqubo
sys.path.insert(0, str(Path(__file__).parent.parent))
from crqubo import CRLLMPipeline


def main():
    """Run a simple CRQUBO example."""
    print("CRQUBO Simple Example")
    print("=" * 50)
    
    # By default this example uses whatever backend is configured in `config.json`.
    # The project is backend-agnostic — OpenAI is optional. To use OpenAI set the
    # environment variable `OPENAI_API_KEY`, or point `config.json` at a different
    # provider (e.g., local/huggingface) and provide the corresponding adapter.
    
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
        print("Tip: configure a backend in config.json or implement a small adapter class for your preferred provider.")


if __name__ == "__main__":
    main()
