"""
CRLLM Modules

This package contains all the modular components of the CRLLM framework.
Each module is designed to be independently configurable and replaceable.
"""

from .task_interface import TaskAgnosticInterface
from .retrieval import RetrievalModule
from .reason_sampler import ReasonSampler
from .semantic_filter import SemanticFilter
from .combinatorial_optimizer import CombinatorialOptimizer
from .reason_orderer import ReasonOrderer
from .final_inference import FinalInference
from .reason_verifier import ReasonVerifier

__all__ = [
    "TaskAgnosticInterface",
    "RetrievalModule",
    "ReasonSampler", 
    "SemanticFilter",
    "CombinatorialOptimizer",
    "ReasonOrderer",
    "FinalInference",
    "ReasonVerifier",
]
