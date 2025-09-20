#!/usr/bin/env python3
"""
CRQUBO Framework Demo

This script demonstrates the complete CRQUBO framework with various
reasoning tasks across different domains.
"""

import os
import sys
import time
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from crqubo import CRLLMPipeline


def print_section(title, char="=", width=60):
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}")


def print_result(result, show_reasoning=True):
    """Print a formatted result."""
    print(f"Answer: {result.final_answer}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Domain: {result.metadata.get('domain', 'unknown')}")
    print(f"Processing time: {result.metadata.get('processing_time', 'N/A')}")
    
    if show_reasoning and result.reasoning_chain:
        print(f"\nReasoning Chain ({len(result.reasoning_chain)} steps):")
        for i, step in enumerate(result.reasoning_chain, 1):
            print(f"  {i}. {step}")


def main():
    """Run the CRQUBO demo."""
    print("🚀 CRQUBO: Combinatorial Reasoning with Large Language Models")
    print("=" * 60)
    
    # NOTE: Do not hardcode API keys in scripts. If you want to use OpenAI, set
    # OPENAI_API_KEY in your environment or configure a different backend in config.json.
    
    # Example queries for different domains
    examples = [
        {
            "query": "Why does smoking cause lung cancer?",
            "domain": "causal",
            "description": "Causal Reasoning Example"
        },
        {
            "query": "If all birds can fly and penguins are birds, can penguins fly?",
            "domain": "logical", 
            "description": "Logical Reasoning Example"
        },
        {
            "query": "What is 15% of 200?",
            "domain": "arithmetic",
            "description": "Arithmetic Reasoning Example"
        },
        {
            "query": "How can we improve team productivity?",
            "domain": "general",
            "description": "General Reasoning Example"
        }
    ]
    
    # Create different pipeline configurations
    configs = {
        "Basic": {},
        "With Retrieval": {"use_retrieval": True},
        "With Verification": {"use_verification": True},
        "Full Pipeline": {"use_retrieval": True, "use_verification": True}
    }
    
    for config_name, config in configs.items():
        print_section(f"Configuration: {config_name}")
        
        # Create pipeline with configuration
        pipeline = CRLLMPipeline(config=config)
        
        for i, example in enumerate(examples, 1):
            print(f"\n{i}. {example['description']}")
            print(f"Query: {example['query']}")
            print("-" * 40)
            
            try:
                start_time = time.time()
                result = pipeline.process_query(
                    query=example['query'],
                    domain=example['domain'],
                    use_retrieval=config.get('use_retrieval', False),
                    use_verification=config.get('use_verification', False)
                )
                processing_time = time.time() - start_time
                
                # Add processing time to metadata
                result.metadata['processing_time'] = f"{processing_time:.2f}s"
                
                print_result(result, show_reasoning=True)
                
                # Show additional metadata
                if result.metadata.get('used_retrieval'):
                    print("📚 Used knowledge retrieval")
                if result.metadata.get('used_verification'):
                    print("✅ Used reasoning verification")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                if "API key" in str(e):
                    print("💡 Make sure to set your OpenAI API key!")
    
    print_section("Framework Information")
    
    # Show pipeline information
    pipeline = CRLLMPipeline()
    info = pipeline.get_pipeline_info()
    
    print("Available Modules:")
    for module_name, module_type in info['modules'].items():
        print(f"  - {module_name}: {module_type}")
    
    print(f"\nConfiguration: {info['config']}")

    print_section("Usage Instructions")
    print("""
To use CRQUBO in your own projects:

1. Install the package:
    pip install -e .

2. Configure a backend (OpenAI optional):
    - For OpenAI: set OPENAI_API_KEY in your environment.
    - For other providers or local models: update `config.json` or implement a small adapter class.

3. Basic usage:
    from crqubo import CRLLMPipeline

    pipeline = CRLLMPipeline()
    result = pipeline.process_query("Your question here")
    print(result.final_answer)

4. Advanced usage with configuration:
    config = {
         "use_retrieval": True,
         "use_verification": True,
         "reason_sampler": {"num_samples": 8}
    }
    pipeline = CRLLMPipeline(config=config)

5. Command line usage:
    python -m crqubo.main "Your question" --domain causal --use-retrieval

For more examples, see the examples/ directory and the Jupyter notebook.
    """)
    
    print_section("Demo Complete! 🎉")
    print("Thank you for trying CRQUBO!")
    print("Visit the project repository for more information.")


if __name__ == "__main__":
    main()
