"""crqubo package: public exports for the CRQUBO framework."""

from .core import CRLLMPipeline
from .exceptions import CRQUBOError
from .logging_utils import configure_logging, get_logger
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

# Ensure logging has a default configuration when package is imported
configure_logging()

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
    "CRQUBOError",
    "configure_logging",
    "get_logger",
]
