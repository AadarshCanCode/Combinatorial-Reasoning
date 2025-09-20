#!/usr/bin/env python3
"""
Simple CRQUBO Example (public subtree)

This is a minimal copy of the example for the public subtree. It demonstrates
how to run a basic query using the `crqubo` package. The example assumes the
public package is installed or the example is run from the package root.
"""

import os
import sys
from pathlib import Path

# If running the public subtree directly, ensure crqubo is importable (adjust PYTHONPATH as needed)
# sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from crqubo import CRLLMPipeline


def main():
    print("CRQUBO Simple Example (public)")
    print("=" * 40)

    # The project supports modular backends. If you want to use OpenAI, set
    # OPENAI_API_KEY as an environment variable. Otherwise configure a local or
    # different provider via config.json or adapter registration.

    pipeline = CRLLMPipeline()

    query = "Why does smoking cause lung cancer?"
    print(f"Query: {query}")
    print("Processing...")

    try:
        result = pipeline.process_query(query=query, domain="causal", use_retrieval=False, use_verification=True)
        print(f"Answer: {result.final_answer}")
        print(f"Confidence: {result.confidence:.2f}")
    except Exception as e:
        print(f"Error running example: {e}")
        print("See USAGE.md in the public subtree for guidance on configuring providers.")


if __name__ == "__main__":
    main()
