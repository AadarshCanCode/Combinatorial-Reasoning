"""
Tests for CRQUBO core functionality.
"""

import pytest
import os
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from crqubo.core import CRLLMPipeline, ReasoningResult, load_config, _validate_config
from crqubo.modules import (
    TaskAgnosticInterface,
    ReasonSampler,
    SemanticFilter,
    CombinatorialOptimizer,
    ReasonOrderer,
    FinalInference,
)
# imports resolved above

# Ensure environment flags don't enable retrieval/verification unexpectedly during tests
for k in list(os.environ.keys()):
    if k.startswith('CRQUBO_') or k.startswith('CRLLM_'):
        os.environ.pop(k, None)

# Module base used for patch decorators in tests
MODULE = 'crqubo'


class TestCRLLMPipeline:
    """Test cases for CRLLMPipeline."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization with default modules."""
        pipeline = CRLLMPipeline()
        
        assert pipeline.task_interface is not None
        assert pipeline.reason_sampler is not None
        assert pipeline.semantic_filter is not None
        assert pipeline.combinatorial_optimizer is not None
        assert pipeline.reason_orderer is not None
        assert pipeline.final_inference is not None
        assert pipeline.retrieval_module is None
        assert pipeline.reason_verifier is None
    
    def test_pipeline_initialization_with_custom_modules(self):
        """Test pipeline initialization with custom modules."""
        custom_interface = Mock(spec=TaskAgnosticInterface)
        custom_sampler = Mock(spec=ReasonSampler)
        
        pipeline = CRLLMPipeline(
            task_interface=custom_interface,
            reason_sampler=custom_sampler
        )
        
        assert pipeline.task_interface is custom_interface
        assert pipeline.reason_sampler is custom_sampler
    
    def test_process_query_basic(self):
        """Test basic query processing."""
        # Import core module object dynamically to patch its classes
        import crqubo.core as core_module

        with patch.object(core_module, 'TaskAgnosticInterface') as mock_interface, \
             patch.object(core_module, 'ReasonSampler') as mock_sampler, \
             patch.object(core_module, 'SemanticFilter') as mock_filter, \
             patch.object(core_module, 'CombinatorialOptimizer') as mock_optimizer, \
             patch.object(core_module, 'ReasonOrderer') as mock_orderer, \
             patch.object(core_module, 'FinalInference') as mock_inference:
            # Setup mocks
            mock_interface.return_value.process_input.return_value = Mock(
                normalized_query="test query",
                domain="general",
                query_type="question",
                complexity="simple",
                metadata={}
            )

            mock_sampler.return_value.sample_reasons.return_value = [
                Mock(content="Step 1", confidence=0.8, reasoning_type="general"),
                Mock(content="Step 2", confidence=0.7, reasoning_type="general")
            ]

            mock_filter.return_value.filter_reasons.return_value = [
                Mock(content="Step 1", confidence=0.8, reasoning_type="general"),
                Mock(content="Step 2", confidence=0.7, reasoning_type="general")
            ]

            mock_optimizer.return_value.optimize_selection.return_value = [
                Mock(content="Step 1", confidence=0.8, reasoning_type="general"),
                Mock(content="Step 2", confidence=0.7, reasoning_type="general")
            ]

            mock_orderer.return_value.order_reasons.return_value = ["Step 1", "Step 2"]

            mock_inference.return_value.generate_answer.return_value = {
                'answer': 'Test answer',
                'confidence': 0.8,
                'metadata': {}
            }

            # Create pipeline and process query
            pipeline = CRLLMPipeline()
            result = pipeline.process_query("test query")

            # Verify result
            assert isinstance(result, ReasoningResult)
            assert result.query == "test query"
            assert result.final_answer == "Test answer"
            assert result.confidence == 0.8
            assert result.reasoning_chain == ["Step 1", "Step 2"]
    
    def test_get_pipeline_info(self):
        """Test getting pipeline information."""
        pipeline = CRLLMPipeline()
        info = pipeline.get_pipeline_info()
        
        assert 'modules' in info
        assert 'config' in info
        assert info['modules']['task_interface'] == 'TaskAgnosticInterface'
        assert info['modules']['reason_sampler'] == 'ReasonSampler'
    
    def test_batch_process(self):
        """Test batch processing."""
        pipeline = CRLLMPipeline()
        
        # Mock the process_query method
        with patch.object(pipeline, 'process_query') as mock_process:
            mock_process.return_value = Mock(
                query="test",
                reasoning_chain=["step1"],
                final_answer="answer",
                confidence=0.8,
                metadata={}
            )
            
            queries = ["query1", "query2", "query3"]
            results = pipeline.batch_process(queries)
            
            assert len(results) == 3
            assert mock_process.call_count == 3


class TestConfigLoading:
    """Test configuration loading functionality."""
    
    def test_load_config_default(self):
        """Test loading default configuration."""
        config = load_config()
        assert 'use_retrieval' in config
        assert 'use_verification' in config
        assert isinstance(config['use_retrieval'], bool)
        assert isinstance(config['use_verification'], bool)
    
    def test_load_config_from_file(self):
        """Test loading configuration from file."""
        test_config = {
            "use_retrieval": True,
            "use_verification": True,
            "reason_sampler": {"model": "gpt-4"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            config_path = f.name
        
        try:
            config = load_config(config_path)
            assert config['use_retrieval'] is True
            assert config['use_verification'] is True
            assert config['reason_sampler']['model'] == "gpt-4"
        finally:
            os.unlink(config_path)
    
    def test_load_config_with_env_vars(self):
        """Test loading configuration with environment variables."""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-key',
            'CRQUBO_USE_RETRIEVAL': 'true',
            'CRQUBO_MODEL': 'gpt-4'
        }):
            config = load_config()
            assert config['reason_sampler']['api_key'] == 'test-key'
            assert config['use_retrieval'] is True
            assert config['reason_sampler']['model'] == 'gpt-4'
    
    def test_validate_config_valid(self):
        """Test configuration validation with valid config."""
        valid_config = {
            "use_retrieval": True,
            "use_verification": False,
            "reason_sampler": {"model": "gpt-3.5-turbo", "num_samples": 5},
            "semantic_filter": {"similarity_threshold": 0.8}
        }
        # Should not raise any exception
        _validate_config(valid_config)
    
    def test_validate_config_invalid(self):
        """Test configuration validation with invalid config."""
        invalid_config = {
            "use_retrieval": "not_a_boolean",
            "reason_sampler": {"num_samples": 50}  # Too many samples
        }
        
        with pytest.raises(ValueError):
            _validate_config(invalid_config)


class TestReasoningResult:
    """Test cases for ReasoningResult."""
    
    def test_reasoning_result_creation(self):
        """Test ReasoningResult creation."""
        result = ReasoningResult(
            query="test query",
            reasoning_chain=["step1", "step2"],
            final_answer="test answer",
            confidence=0.8,
            metadata={"key": "value"}
        )
        
        assert result.query == "test query"
        assert result.reasoning_chain == ["step1", "step2"]
        assert result.final_answer == "test answer"
        assert result.confidence == 0.8
        assert result.metadata == {"key": "value"}
    
    def test_reasoning_result_edge_cases(self):
        """Test ReasoningResult with edge cases."""
        # Empty reasoning chain
        result = ReasoningResult(
            query="",
            reasoning_chain=[],
            final_answer="",
            confidence=0.0,
            metadata={}
        )
        
        assert result.query == ""
        assert result.reasoning_chain == []
        assert result.final_answer == ""
        assert result.confidence == 0.0
        assert result.metadata == {}
        
        # High confidence
        result = ReasoningResult(
            query="test",
            reasoning_chain=["step"],
            final_answer="answer",
            confidence=1.0,
            metadata={"domain": "test"}
        )
        
        assert result.confidence == 1.0


class TestPipelineErrorHandling:
    """Test error handling in the pipeline."""
    
    def test_process_query_with_llm_error(self):
        """Test query processing when LLM calls fail."""
        pipeline = CRLLMPipeline()
        
        # Mock the reason sampler to raise an exception
        with patch.object(pipeline.reason_sampler, 'sample_reasons') as mock_sampler:
            mock_sampler.side_effect = Exception("LLM API Error")
            
            result = pipeline.process_query("test query")
            
            assert isinstance(result, ReasoningResult)
            assert "Error processing query" in result.final_answer
            assert result.confidence == 0.0
            assert 'error' in result.metadata
    
    def test_process_query_with_retrieval_error(self):
        """Test query processing when retrieval fails."""
        pipeline = CRLLMPipeline(use_retrieval=True)
        
        # Mock retrieval to fail
        with patch.object(pipeline.retrieval_module, 'retrieve') as mock_retrieve:
            mock_retrieve.side_effect = Exception("Retrieval Error")
            
            result = pipeline.process_query("test query", use_retrieval=True)
            
            # Should still process without retrieval
            assert isinstance(result, ReasoningResult)
            assert result.metadata['used_retrieval'] is False
    
    def test_process_query_with_verification_error(self):
        """Test query processing when verification fails."""
        pipeline = CRLLMPipeline(use_verification=True)
        
        # Mock verification to fail
        with patch.object(pipeline.reason_verifier, 'verify_reasons') as mock_verify:
            mock_verify.side_effect = Exception("Verification Error")
            
            result = pipeline.process_query("test query", use_verification=True)
            
            # Should still process without verification
            assert isinstance(result, ReasoningResult)
            assert result.metadata['used_verification'] is False


class TestPipelineRetryMechanism:
    """Test retry mechanism in the pipeline."""
    
    def test_retry_llm_call_success(self):
        """Test LLM retry mechanism on success."""
        pipeline = CRLLMPipeline()
        
        def mock_func():
            return "success"
        
        result = pipeline._retry_llm_call(mock_func, max_retries=3, retry_delay=0.1, operation="test")
        assert result == "success"
    
    def test_retry_llm_call_failure(self):
        """Test LLM retry mechanism on failure."""
        pipeline = CRLLMPipeline()
        
        def mock_func():
            raise Exception("Always fails")
        
        with pytest.raises(Exception, match="Always fails"):
            pipeline._retry_llm_call(mock_func, max_retries=2, retry_delay=0.1, operation="test")
    
    def test_retry_llm_call_partial_failure(self):
        """Test LLM retry mechanism with partial failure."""
        pipeline = CRLLMPipeline()
        
        call_count = 0
        def mock_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Fails first two times")
            return "success"
        
        result = pipeline._retry_llm_call(mock_func, max_retries=3, retry_delay=0.1, operation="test")
        assert result == "success"
        assert call_count == 3


if __name__ == "__main__":
    pytest.main([__file__])
