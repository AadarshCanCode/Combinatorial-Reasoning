"""crqubo package: public exports for the CRQUBO framework."""

from .core import CRLLMPipeline
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

__version__ = "0.1.0"
__author__ = "CRQUBO Team"

__all__ = [
    "CRLLMPipeline",
    "TaskAgnosticInterface",
    "RetrievalModule",
    "ReasonSampler",
    "SemanticFilter",
    "CombinatorialOptimizer",
    "ReasonOrderer",
    "FinalInference",
    "ReasonVerifier",
]
