"""
CRLLM Main Entry Point

This module provides the main entry point for the CRLLM framework,
including command-line interface and example usage.
"""

import argparse
import json
import sys
from typing import Dict, Any, Optional
from pathlib import Path

from .core import CRLLMPipeline
from .modules import (
    TaskAgnosticInterface,
    RetrievalModule,
    ReasonSampler,
    SemanticFilter,
    CombinatorialOptimizer,
    ReasonOrderer,
    FinalInference,
    ReasonVerifier,
)


def create_default_pipeline(config: Optional[Dict[str, Any]] = None) -> CRLLMPipeline:
    """Create a default CRLLM pipeline with all modules."""
    config = config or {}
    
    # Create modules with default configurations
    task_interface = TaskAgnosticInterface(config.get('task_interface', {}))
    
    retrieval_module = None
    if config.get('use_retrieval', False):
        retrieval_module = RetrievalModule(config.get('retrieval', {}))
    
    reason_sampler = ReasonSampler(config.get('reason_sampler', {}))
    semantic_filter = SemanticFilter(config.get('semantic_filter', {}))
    combinatorial_optimizer = CombinatorialOptimizer(config.get('combinatorial_optimizer', {}))
    reason_orderer = ReasonOrderer(config.get('reason_orderer', {}))
    final_inference = FinalInference(config.get('final_inference', {}))
    
    reason_verifier = None
    if config.get('use_verification', False):
        reason_verifier = ReasonVerifier(config.get('reason_verifier', {}))
    
    # Create pipeline
    pipeline = CRLLMPipeline(
        task_interface=task_interface,
        retrieval_module=retrieval_module,
        reason_sampler=reason_sampler,
        semantic_filter=semantic_filter,
        combinatorial_optimizer=combinatorial_optimizer,
        reason_orderer=reason_orderer,
        final_inference=final_inference,
        reason_verifier=reason_verifier,
        config=config
    )
    
    return pipeline


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Config file {config_path} not found, using defaults")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing config file {config_path}: {e}")
        return {}


def save_result(result: Dict[str, Any], output_path: str) -> None:
    """Save result to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)


def main():
    """Main entry point for command-line interface."""
    parser = argparse.ArgumentParser(
        description="CRLLM: Combinatorial Reasoning with Large Language Models"
    )
    
    parser.add_argument(
        "query",
        help="Query to process through the CRLLM pipeline"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration JSON file"
    )
    
    parser.add_argument(
        "--domain",
        type=str,
        choices=["causal", "logical", "arithmetic", "spatial", "temporal", "comparative", "general"],
        default="general",
        help="Reasoning domain for the query"
    )
    
    parser.add_argument(
        "--use-retrieval",
        action="store_true",
        help="Enable knowledge retrieval (RAG)"
    )
    
    parser.add_argument(
        "--use-verification",
        action="store_true",
        help="Enable reasoning verification"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save the result JSON file"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # Override config with command-line arguments
    config['use_retrieval'] = args.use_retrieval
    config['use_verification'] = args.use_verification
    
    try:
        # Create pipeline
        pipeline = create_default_pipeline(config)
        
        if args.verbose:
            print("CRLLM Pipeline created successfully")
            print(f"Pipeline info: {pipeline.get_pipeline_info()}")
        
        # Process query
        result = pipeline.process_query(
            query=args.query,
            domain=args.domain,
            use_retrieval=args.use_retrieval,
            use_verification=args.use_verification
        )
        
        # Display result
        print("\n" + "="*50)
        print("CRLLM RESULT")
        print("="*50)
        print(f"Query: {result.query}")
        print(f"Domain: {result.metadata.get('domain', 'unknown')}")
        print(f"Confidence: {result.confidence:.2f}")
        print("\nReasoning Chain:")
        for i, step in enumerate(result.reasoning_chain, 1):
            print(f"{i}. {step}")
        print(f"\nFinal Answer: {result.final_answer}")
        print("="*50)
        
        # Save result if requested
        if args.output:
            result_dict = {
                'query': result.query,
                'reasoning_chain': result.reasoning_chain,
                'final_answer': result.final_answer,
                'confidence': result.confidence,
                'metadata': result.metadata
            }
            save_result(result_dict, args.output)
            print(f"\nResult saved to {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run_example():
    """Run example usage of CRLLM."""
    print("CRLLM Example Usage")
    print("="*50)
    
    # Example queries for different domains
    examples = [
        {
            "query": "Why does smoking cause lung cancer?",
            "domain": "causal",
            "use_retrieval": True,
            "use_verification": True
        },
        {
            "query": "If all birds can fly and penguins are birds, can penguins fly?",
            "domain": "logical",
            "use_retrieval": False,
            "use_verification": True
        },
        {
            "query": "What is 15% of 200?",
            "domain": "arithmetic",
            "use_retrieval": False,
            "use_verification": False
        },
        {
            "query": "How can we improve team productivity?",
            "domain": "general",
            "use_retrieval": True,
            "use_verification": False
        }
    ]
    
    # Create pipeline
    pipeline = create_default_pipeline()
    
    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}: {example['query']}")
        print("-" * 30)
        
        try:
            result = pipeline.process_query(**example)
            print(f"Answer: {result.final_answer}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Reasoning steps: {len(result.reasoning_chain)}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Run example if no arguments provided
        run_example()
    else:
        main()

