"""
Tests for CRLLM core functionality.
"""

import pytest
from unittest.mock import Mock, patch
from crllm.core import CRLLMPipeline, ReasoningResult
from crllm.modules import (
    TaskAgnosticInterface,
    ReasonSampler,
    SemanticFilter,
    CombinatorialOptimizer,
    ReasonOrderer,
    FinalInference
)


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
    
    @patch('crllm.core.TaskAgnosticInterface')
    @patch('crllm.core.ReasonSampler')
    @patch('crllm.core.SemanticFilter')
    @patch('crllm.core.CombinatorialOptimizer')
    @patch('crllm.core.ReasonOrderer')
    @patch('crllm.core.FinalInference')
    def test_process_query_basic(self, mock_inference, mock_orderer, 
                                mock_optimizer, mock_filter, mock_sampler, mock_interface):
        """Test basic query processing."""
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


if __name__ == "__main__":
    pytest.main([__file__])
