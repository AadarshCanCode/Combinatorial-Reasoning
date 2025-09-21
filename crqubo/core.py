"""
Core CRQUBO Pipeline orchestrator that coordinates all modules.
"""

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .modules import (
    CombinatorialOptimizer,
    FinalInference,
    ReasonOrderer,
    ReasonSampler,
    ReasonVerifier,
    RetrievalModule,
    SemanticFilter,
    TaskAgnosticInterface,
)

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class ReasoningResult:
    """Container for the final reasoning result."""

    query: str
    reasoning_chain: List[str]
    final_answer: str
    confidence: float
    metadata: Dict[str, Any]


@lru_cache(maxsize=1)
def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file and environment variables.

    Args:
        config_path: Path to configuration file. If None, looks for config.json

    Returns:
        Merged configuration dictionary
    """
    # Default configuration
    default_config = {
        "use_retrieval": False,
        "use_verification": False,
        "task_interface": {},
        "retrieval": {},
        "reason_sampler": {},
        "semantic_filter": {},
        "combinatorial_optimizer": {},
        "reason_orderer": {},
        "final_inference": {},
        "reason_verifier": {},
    }

    # Load from file if provided
    file_config = {}
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, "r") as f:
                file_config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
    elif (
        os.getenv("CRQUBO_USE_PROJECT_CONFIG", "").lower() in ("true", "1", "yes")
        and Path("config.json").exists()
    ):
        try:
            with open("config.json", "r") as f:
                file_config = json.load(f)
            logger.info("Loaded configuration from config.json")
        except Exception as e:
            logger.warning(f"Failed to load config.json: {e}")

    # Load from environment variables
    env_config = {}

    # API Keys
    if os.getenv("OPENAI_API_KEY"):
        env_config.setdefault("reason_sampler", {})["api_key"] = os.getenv(
            "OPENAI_API_KEY"
        )
        env_config.setdefault("final_inference", {})["api_key"] = os.getenv(
            "OPENAI_API_KEY"
        )

    if os.getenv("DWAVE_API_TOKEN"):
        env_config.setdefault("combinatorial_optimizer", {})["api_token"] = os.getenv(
            "DWAVE_API_TOKEN"
        )

    if os.getenv("HUGGINGFACE_API_TOKEN"):
        env_config.setdefault("reason_sampler", {})["hf_token"] = os.getenv(
            "HUGGINGFACE_API_TOKEN"
        )

    # Feature flags
    if os.getenv("CRQUBO_USE_RETRIEVAL", "").lower() in ("true", "1", "yes"):
        env_config["use_retrieval"] = True

    if os.getenv("CRQUBO_USE_VERIFICATION", "").lower() in ("true", "1", "yes"):
        env_config["use_verification"] = True

    # Model configurations
    if os.getenv("CRQUBO_MODEL"):
        env_config.setdefault("reason_sampler", {})["model"] = os.getenv("CRQUBO_MODEL")
        env_config.setdefault("final_inference", {})["model"] = os.getenv(
            "CRQUBO_MODEL"
        )

    if os.getenv("CRQUBO_EMBEDDING_MODEL"):
        env_config.setdefault("semantic_filter", {})["model_name"] = os.getenv(
            "CRQUBO_EMBEDDING_MODEL"
        )
        env_config.setdefault("retrieval", {})["embedding_model"] = os.getenv(
            "CRQUBO_EMBEDDING_MODEL"
        )

    # Merge configurations (file config overrides defaults, env overrides file)
    config = default_config.copy()
    config.update(file_config)
    config.update(env_config)

    # Validate configuration
    _validate_config(config)

    return config


def _validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration parameters.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate boolean flags
    for flag in ["use_retrieval", "use_verification"]:
        if flag in config and not isinstance(config[flag], bool):
            raise ValueError(f"{flag} must be a boolean")

    # Validate API keys if modules are enabled
    if config.get("use_retrieval", False):
        if not config.get("retrieval", {}).get("api_key") and not os.getenv(
            "OPENAI_API_KEY"
        ):
            logger.warning("Retrieval enabled but no API key found")

    # Validate model names
    for module in ["reason_sampler", "final_inference"]:
        if module in config and "model" in config[module]:
            model = config[module]["model"]
            if not isinstance(model, str) or not model.strip():
                raise ValueError(f"{module}.model must be a non-empty string")

    # Validate numeric parameters
    numeric_params = [
        ("reason_sampler", "num_samples", 1, 20),
        ("reason_sampler", "temperature", 0.0, 2.0),
        ("semantic_filter", "similarity_threshold", 0.0, 1.0),
        ("semantic_filter", "quality_threshold", 0.0, 1.0),
        ("combinatorial_optimizer", "max_selections", 1, 50),
    ]

    for module, param, min_val, max_val in numeric_params:
        if module in config and param in config[module]:
            value = config[module][param]
            if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
                raise ValueError(
                    f"{module}.{param} must be between {min_val} and {max_val}"
                )


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
        config_path: Optional[str] = None,
        use_retrieval: Optional[bool] = None,
        use_verification: Optional[bool] = None,
    ):
        """
        Initialize the CRQUBO pipeline with optional modules.

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
            config_path: Path to configuration file
        """
        # Load configuration
        if config is None:
            config = load_config(config_path)
        else:
            # Merge with loaded config if config_path is provided
            if config_path:
                loaded_config = load_config(config_path)
                loaded_config.update(config)
                config = loaded_config

        # Allow explicit constructor flags to override configuration
        if use_retrieval is not None:
            config["use_retrieval"] = bool(use_retrieval)
        if use_verification is not None:
            config["use_verification"] = bool(use_verification)

        self.config = config

        # Initialize modules with configuration
        self.task_interface = task_interface or TaskAgnosticInterface(
            config.get("task_interface", {})
        )
        self.retrieval_module = retrieval_module
        if config.get("use_retrieval", False):
            self.retrieval_module = retrieval_module or RetrievalModule(
                config.get("retrieval", {})
            )

        self.reason_sampler = reason_sampler or ReasonSampler(
            config.get("reason_sampler", {})
        )
        self.semantic_filter = semantic_filter or SemanticFilter(
            config.get("semantic_filter", {})
        )
        self.combinatorial_optimizer = (
            combinatorial_optimizer
            or CombinatorialOptimizer(config.get("combinatorial_optimizer", {}))
        )
        self.reason_orderer = reason_orderer or ReasonOrderer(
            config.get("reason_orderer", {})
        )
        self.final_inference = final_inference or FinalInference(
            config.get("final_inference", {})
        )

        self.reason_verifier = reason_verifier
        if config.get("use_verification", False):
            self.reason_verifier = reason_verifier or ReasonVerifier(
                config.get("reason_verifier", {})
            )

    logger.info("CRQUBO Pipeline initialized successfully")

    def process_query(
        self,
        query: Union[str, Dict[str, Any]],
        domain: Optional[str] = None,
        use_retrieval: bool = False,
        use_verification: bool = False,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs,
    ) -> ReasoningResult:
        """
        Process a query through the complete reasoning pipeline.

        Args:
            query: Input query (string or structured dict)
            domain: Optional domain specification
                (e.g., 'causal', 'logical', 'arithmetic')
            use_retrieval: Whether to use knowledge retrieval
            use_verification: Whether to use reason verification
            max_retries: Maximum number of retries for LLM calls
            retry_delay: Delay between retries in seconds
            **kwargs: Additional parameters for specific modules

        Returns:
            ReasoningResult containing the final answer and reasoning chain
        """
        import time

        start_time = time.time()
        logger.info(f"Processing query: {str(query)[:100]}...")

        try:
            # Step 1: Process input through task-agnostic interface
            processed_query = self.task_interface.process_input(query, domain=domain)
            logger.debug(f"Processed query: {processed_query.normalized_query}")

            # Step 2: Optional knowledge retrieval
            retrieved_knowledge = None
            if use_retrieval and self.retrieval_module:
                try:
                    retrieved_knowledge = self.retrieval_module.retrieve(
                        processed_query, **kwargs.get("retrieval_kwargs", {})
                    )
                    doc_count = (
                        len(retrieved_knowledge.documents) if retrieved_knowledge else 0
                    )
                    logger.debug(f"Retrieved {doc_count} documents")
                except Exception as e:
                    logger.warning(f"Retrieval failed: {e}")
                    retrieved_knowledge = None

            # Step 3: Generate candidate reasoning steps with retries
            candidate_reasons = self._retry_llm_call(
                lambda: self.reason_sampler.sample_reasons(
                    processed_query,
                    domain=domain,
                    retrieved_knowledge=retrieved_knowledge,
                    **kwargs.get("sampling_kwargs", {}),
                ),
                max_retries=max_retries,
                retry_delay=retry_delay,
                operation="reason sampling",
            )
            logger.debug(f"Generated {len(candidate_reasons)} candidate reasons")

            # Step 4: Semantic filtering to remove duplicates
            filtered_reasons = self.semantic_filter.filter_reasons(
                candidate_reasons, **kwargs.get("filtering_kwargs", {})
            )
            logger.debug(f"Filtered to {len(filtered_reasons)} reasons")

            # Step 5: Combinatorial optimization to select diverse, high-utility reasons
            selected_reasons = self.combinatorial_optimizer.optimize_selection(
                filtered_reasons,
                processed_query,
                **kwargs.get("optimization_kwargs", {}),
            )
            logger.debug(f"Selected {len(selected_reasons)} reasons")

            # Step 6: Order reasons into logical chain
            ordered_reasons = self.reason_orderer.order_reasons(
                selected_reasons, processed_query, **kwargs.get("ordering_kwargs", {})
            )
            logger.debug(f"Ordered {len(ordered_reasons)} reasons")

            # Step 7: Optional reason verification
            verified_reasons = ordered_reasons
            if use_verification and self.reason_verifier:
                try:
                    verified_reasons = self.reason_verifier.verify_reasons(
                        ordered_reasons,
                        processed_query,
                        **kwargs.get("verification_kwargs", {}),
                    )
                    logger.debug(f"Verified {len(verified_reasons)} reasons")
                except Exception as e:
                    logger.warning(f"Verification failed: {e}")
                    verified_reasons = ordered_reasons

            # Step 8: Generate final answer with retries
            final_result = self._retry_llm_call(
                lambda: self.final_inference.generate_answer(
                    processed_query,
                    verified_reasons,
                    retrieved_knowledge=retrieved_knowledge,
                    **kwargs.get("inference_kwargs", {}),
                ),
                max_retries=max_retries,
                retry_delay=retry_delay,
                operation="final inference",
            )

            processing_time = time.time() - start_time
            logger.info(f"Query processed successfully in {processing_time:.2f}s")

            query_value = getattr(processed_query, "normalized_query", None)
            if query_value is None:
                query_value = str(processed_query)

            return ReasoningResult(
                query=query_value,
                reasoning_chain=verified_reasons,
                final_answer=final_result["answer"],
                confidence=final_result.get("confidence", 0.0),
                metadata={
                    "domain": domain,
                    "used_retrieval": use_retrieval and retrieved_knowledge is not None,
                    "used_verification": use_verification
                    and self.reason_verifier is not None,
                    "num_candidates": len(candidate_reasons),
                    "num_filtered": len(filtered_reasons),
                    "num_selected": len(selected_reasons),
                    "num_verified": len(verified_reasons),
                    "processing_time": processing_time,
                    **final_result.get("metadata", {}),
                },
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Query processing failed after {processing_time:.2f}s: {e}")

            # Return error result
            return ReasoningResult(
                query=str(query),
                reasoning_chain=[],
                final_answer=f"Error processing query: {str(e)}",
                confidence=0.0,
                metadata={
                    "domain": domain,
                    "error": str(e),
                    "processing_time": processing_time,
                    "used_retrieval": False,
                    "used_verification": False,
                },
            )

    def _retry_llm_call(
        self, func, max_retries: int, retry_delay: float, operation: str
    ):
        """Retry an LLM call with exponential backoff.

        Args:
            func: Function to retry
            max_retries: Maximum number of retries
            retry_delay: Base delay between retries
            operation: Name of the operation for logging

        Returns:
            Result of the function call

        Raises:
            Exception: If all retries fail
        """
        import time

        for attempt in range(max_retries + 1):
            try:
                return func()
            except Exception as e:
                if attempt == max_retries:
                    logger.error(
                        f"{operation} failed after {max_retries + 1} attempts: {e}"
                    )
                    raise
                else:
                    delay = retry_delay * (2**attempt)
                    logger.warning(
                        f"{operation} failed (attempt {attempt + 1}/"
                        f"{max_retries + 1}): {e}. Retrying in {delay}s..."
                    )
                    time.sleep(delay)

    def batch_process(
        self, queries: List[Union[str, Dict[str, Any]]], **kwargs
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
            "modules": {
                "task_interface": type(self.task_interface).__name__,
                "retrieval_module": (
                    type(self.retrieval_module).__name__
                    if self.retrieval_module
                    else None
                ),
                "reason_sampler": type(self.reason_sampler).__name__,
                "semantic_filter": type(self.semantic_filter).__name__,
                "combinatorial_optimizer": type(self.combinatorial_optimizer).__name__,
                "reason_orderer": type(self.reason_orderer).__name__,
                "final_inference": type(self.final_inference).__name__,
                "reason_verifier": (
                    type(self.reason_verifier).__name__
                    if self.reason_verifier
                    else None
                ),
            },
            "config": self.config,
        }
