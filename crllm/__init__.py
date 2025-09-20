"""
CRLLM: Combinatorial Reasoning with Large Language Models

A modular reasoning framework that generalizes the Combinatorial Reasoning (CR) pipeline
across diverse reasoning tasks using Large Language Models.
"""

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

__version__ = "0.1.0"
__author__ = "CRLLM Team"

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
