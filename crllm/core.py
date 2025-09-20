"""
Core CRLLM Pipeline orchestrator that coordinates all modules.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

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


@dataclass
class ReasoningResult:
    """Container for the final reasoning result."""
    query: str
    reasoning_chain: List[str]
    final_answer: str
    confidence: float
    metadata: Dict[str, Any]


class CRLLMPipeline:
    """
    Main pipeline orchestrator for the Combinatorial Reasoning framework.
    
    This class coordinates all modules to process queries through the complete
    reasoning pipeline: input → retrieval → sampling → filtering → optimization 
    → ordering → verification → final inference.
    """
    
    def __init__(
        self,
        task_interface: Optional[TaskAgnosticInterface] = None,
        retrieval_module: Optional[RetrievalModule] = None,
        reason_sampler: Optional[ReasonSampler] = None,
        semantic_filter: Optional[SemanticFilter] = None,
        combinatorial_optimizer: Optional[CombinatorialOptimizer] = None,
        reason_orderer: Optional[ReasonOrderer] = None,
        final_inference: Optional[FinalInference] = None,
        reason_verifier: Optional[ReasonVerifier] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the CRLLM pipeline with optional modules.
        
        Args:
            task_interface: Module for handling task-agnostic input
            retrieval_module: Optional RAG module for knowledge retrieval
            reason_sampler: Module for generating candidate reasoning steps
            semantic_filter: Module for deduplicating reasoning chains
            combinatorial_optimizer: QUBO-based optimizer for reason selection
            reason_orderer: Module for arranging reasons into logical chains
            final_inference: Module for generating final answers
            reason_verifier: Optional module for verifying reasoning consistency
            config: Configuration dictionary for the pipeline
        """
        self.task_interface = task_interface or TaskAgnosticInterface()
        self.retrieval_module = retrieval_module
        self.reason_sampler = reason_sampler or ReasonSampler()
        self.semantic_filter = semantic_filter or SemanticFilter()
        self.combinatorial_optimizer = combinatorial_optimizer or CombinatorialOptimizer()
        self.reason_orderer = reason_orderer or ReasonOrderer()
        self.final_inference = final_inference or FinalInference()
        self.reason_verifier = reason_verifier
        self.config = config or {}
        
    def process_query(
        self,
        query: Union[str, Dict[str, Any]],
        domain: Optional[str] = None,
        use_retrieval: bool = False,
        use_verification: bool = False,
        **kwargs
    ) -> ReasoningResult:
        """
        Process a query through the complete reasoning pipeline.
        
        Args:
            query: Input query (string or structured dict)
            domain: Optional domain specification (e.g., 'causal', 'logical', 'arithmetic')
            use_retrieval: Whether to use knowledge retrieval
            use_verification: Whether to use reason verification
            **kwargs: Additional parameters for specific modules
            
        Returns:
            ReasoningResult containing the final answer and reasoning chain
        """
        # Step 1: Process input through task-agnostic interface
        processed_query = self.task_interface.process_input(query, domain=domain)
        
        # Step 2: Optional knowledge retrieval
        retrieved_knowledge = None
        if use_retrieval and self.retrieval_module:
            retrieved_knowledge = self.retrieval_module.retrieve(
                processed_query, **kwargs.get('retrieval_kwargs', {})
            )
        
        # Step 3: Generate candidate reasoning steps
        candidate_reasons = self.reason_sampler.sample_reasons(
            processed_query, 
            domain=domain,
            retrieved_knowledge=retrieved_knowledge,
            **kwargs.get('sampling_kwargs', {})
        )
        
        # Step 4: Semantic filtering to remove duplicates
        filtered_reasons = self.semantic_filter.filter_reasons(
            candidate_reasons, 
            **kwargs.get('filtering_kwargs', {})
        )
        
        # Step 5: Combinatorial optimization to select diverse, high-utility reasons
        selected_reasons = self.combinatorial_optimizer.optimize_selection(
            filtered_reasons,
            processed_query,
            **kwargs.get('optimization_kwargs', {})
        )
        
        # Step 6: Order reasons into logical chain
        ordered_reasons = self.reason_orderer.order_reasons(
            selected_reasons,
            processed_query,
            **kwargs.get('ordering_kwargs', {})
        )
        
        # Step 7: Optional reason verification
        verified_reasons = ordered_reasons
        if use_verification and self.reason_verifier:
            verified_reasons = self.reason_verifier.verify_reasons(
                ordered_reasons,
                processed_query,
                **kwargs.get('verification_kwargs', {})
            )
        
        # Step 8: Generate final answer
        final_result = self.final_inference.generate_answer(
            processed_query,
            verified_reasons,
            retrieved_knowledge=retrieved_knowledge,
            **kwargs.get('inference_kwargs', {})
        )
        
        return ReasoningResult(
            query=str(processed_query),
            reasoning_chain=verified_reasons,
            final_answer=final_result['answer'],
            confidence=final_result.get('confidence', 0.0),
            metadata={
                'domain': domain,
                'used_retrieval': use_retrieval and retrieved_knowledge is not None,
                'used_verification': use_verification and self.reason_verifier is not None,
                'num_candidates': len(candidate_reasons),
                'num_filtered': len(filtered_reasons),
                'num_selected': len(selected_reasons),
                'num_verified': len(verified_reasons),
                **final_result.get('metadata', {})
            }
        )
    
    def batch_process(
        self,
        queries: List[Union[str, Dict[str, Any]]],
        **kwargs
    ) -> List[ReasoningResult]:
        """
        Process multiple queries in batch.
        
        Args:
            queries: List of queries to process
            **kwargs: Parameters passed to process_query
            
        Returns:
            List of ReasoningResult objects
        """
        return [self.process_query(query, **kwargs) for query in queries]
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the current pipeline configuration."""
        return {
            'modules': {
                'task_interface': type(self.task_interface).__name__,
                'retrieval_module': type(self.retrieval_module).__name__ if self.retrieval_module else None,
                'reason_sampler': type(self.reason_sampler).__name__,
                'semantic_filter': type(self.semantic_filter).__name__,
                'combinatorial_optimizer': type(self.combinatorial_optimizer).__name__,
                'reason_orderer': type(self.reason_orderer).__name__,
                'final_inference': type(self.final_inference).__name__,
                'reason_verifier': type(self.reason_verifier).__name__ if self.reason_verifier else None,
            },
            'config': self.config
        }
