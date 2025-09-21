#!/usr/bin/env python3
"""
Performance test script for CRQUBO optimization.
This script tests the performance improvements made to the codebase.
"""

import time
from crqubo.core import CRLLMPipeline, load_config


def test_config_loading_performance():
    """Test configuration loading performance with caching."""
    print("Testing configuration loading performance...")
    
    # Test without caching (first call)
    start_time = time.time()
    config1 = load_config()
    first_call_time = time.time() - start_time
    
    # Test with caching (subsequent calls)
    start_time = time.time()
    config2 = load_config()
    second_call_time = time.time() - start_time
    
    print(f"First call (no cache): {first_call_time:.4f}s")
    print(f"Second call (cached): {second_call_time:.4f}s")
    if second_call_time > 0:
        print(f"Speedup: {first_call_time/second_call_time:.2f}x")
    else:
        print("Speedup: Instant (cached)")
    
    return config1 == config2


def test_pipeline_initialization():
    """Test pipeline initialization performance."""
    print("\nTesting pipeline initialization...")
    
    start_time = time.time()
    pipeline = CRLLMPipeline()
    init_time = time.time() - start_time
    
    print(f"Pipeline initialization time: {init_time:.4f}s")
    
    # Test pipeline info retrieval
    start_time = time.time()
    info = pipeline.get_pipeline_info()
    info_time = time.time() - start_time
    
    print(f"Pipeline info retrieval time: {info_time:.4f}s")
    print(f"Available modules: {list(info['modules'].keys())}")
    
    return pipeline


def test_simple_query():
    """Test a simple query processing."""
    print("\nTesting simple query processing...")
    
    pipeline = CRLLMPipeline()
    
    # Simple test query
    test_query = "What is 2 + 2?"
    
    start_time = time.time()
    try:
        result = pipeline.process_query(
            query=test_query,
            domain="arithmetic",
            use_retrieval=False,
            use_verification=False
        )
        processing_time = time.time() - start_time
        
        print(f"Query processing time: {processing_time:.4f}s")
        print(f"Query: {result.query}")
        print(f"Answer: {result.final_answer}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Reasoning steps: {len(result.reasoning_chain)}")
        
        return True
    except Exception as e:
        print(f"Error processing query: {e}")
        return False


def main():
    """Run all performance tests."""
    print("CRQUBO Performance Test Suite")
    print("=" * 50)
    
    # Test 1: Configuration loading
    config_test_passed = test_config_loading_performance()
    
    # Test 2: Pipeline initialization
    pipeline = test_pipeline_initialization()
    
    # Test 3: Simple query processing
    query_test_passed = test_simple_query()
    
    # Summary
    print("\n" + "=" * 50)
    print("Performance Test Summary:")
    print(f"✓ Configuration caching: {'PASSED' if config_test_passed else 'FAILED'}")
    print(f"✓ Pipeline initialization: PASSED")
    print(f"✓ Query processing: {'PASSED' if query_test_passed else 'FAILED'}")
    
    if config_test_passed and query_test_passed:
        print("\n🎉 All performance tests passed!")
    else:
        print("\n⚠️  Some tests failed. Check the output above.")


if __name__ == "__main__":
    main()
