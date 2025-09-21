"""
Semantic Filtering Module

This module uses embeddings (e.g., Sentence-BERT) to remove near-duplicate
reasoning chains and filter out low-quality reasoning steps.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .reason_sampler import ReasoningStep

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class FilteredReasoningStep:
    """Container for filtered reasoning step with metadata."""

    step: "ReasoningStep"
    cluster_id: Optional[int]
    quality_score: float
    is_duplicate: bool
    similar_steps: List[str]


@dataclass
class FilteringResult:
    """Container for filtering results."""

    filtered_steps: List[FilteredReasoningStep]
    original_count: int
    filtered_count: int
    duplicates_removed: int
    quality_threshold: float
    similarity_threshold: float


class BaseSemanticFilter(ABC):
    """Abstract base class for semantic filtering implementations."""

    @abstractmethod
    def filter_reasons(
        self,
        reasoning_steps: List["ReasoningStep"],
        similarity_threshold: float = 0.8,
        quality_threshold: float = 0.3,
        **kwargs,
    ) -> List[FilteredReasoningStep]:
        """Filter reasoning steps based on semantic similarity and quality."""
        pass


class SentenceBERTFilter(BaseSemanticFilter):
    """Sentence-BERT based semantic filter implementation."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Sentence-BERT filter.

        Args:
            model_name: Sentence transformer model name
            config: Additional configuration
        """
        self.model_name = model_name
        self.config = config or {}

        # Load configuration parameters
        self.similarity_threshold = self.config.get("similarity_threshold", 0.8)
        self.quality_threshold = self.config.get("quality_threshold", 0.3)
        self.min_length = self.config.get("min_length", 5)
        self.max_length = self.config.get("max_length", 1000)
        self.use_clustering = self.config.get("use_clustering", True)
        self.clustering_eps = self.config.get("clustering_eps", 0.3)
        self.clustering_min_samples = self.config.get("clustering_min_samples", 2)

        self.embedding_model = SentenceTransformer(model_name)

        # Quality assessment patterns
        self.quality_patterns = self._load_quality_patterns()

    def _load_quality_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for quality assessment."""
        return {
            "high_quality": [
                r"because|since|due to|therefore|thus|hence|consequently",
                r"clearly|obviously|evidently|apparently",
                r"step \d+|first|second|third|next|finally",
                r"if.*then|when.*then|given.*then",
            ],
            "low_quality": [
                r"^.{1,10}$",  # Very short steps
                r"^[^a-zA-Z]*$",  # Non-alphabetic
                r"^.*\?$",  # Questions instead of statements
                r"^.*\.{3,}$",  # Ellipsis
                r"^.*\s{3,}.*$",  # Multiple spaces
            ],
            "reasoning_indicators": [
                r"analyze|examine|consider|evaluate|assess",
                r"compare|contrast|differentiate|distinguish",
                r"identify|determine|establish|prove|demonstrate",
                r"explain|describe|illustrate|clarify",
            ],
        }

    def filter_reasons(
        self,
        reasoning_steps: List["ReasoningStep"],
        similarity_threshold: float = 0.8,
        quality_threshold: float = 0.3,
        **kwargs,
    ) -> List[FilteredReasoningStep]:
        """Filter reasoning steps using semantic similarity and quality assessment."""
        if not reasoning_steps:
            return []

        # Use config values if not provided
        similarity_threshold = similarity_threshold or self.similarity_threshold
        quality_threshold = quality_threshold or self.quality_threshold

        # Step 1: Length filtering
        length_filtered = self._filter_by_length(reasoning_steps)

        # Extract step contents
        step_contents = [step.content for step in length_filtered]

        # Generate embeddings
        embeddings = self.embedding_model.encode(step_contents)

        # Assess quality of each step
        quality_scores = self._assess_quality(step_contents)

        # Find duplicates using clustering
        duplicate_groups = self._find_duplicates(embeddings, similarity_threshold)

        # Create filtered steps
        filtered_steps = []
        processed_indices = set()

        for i, step in enumerate(reasoning_steps):
            if i in processed_indices:
                continue

            # Check quality threshold
            if quality_scores[i] < quality_threshold:
                continue

            # Find similar steps
            similar_steps = []
            cluster_id = None

            if i in duplicate_groups:
                cluster_id = duplicate_groups[i]
                similar_indices = [
                    j
                    for j, cluster in duplicate_groups.items()
                    if cluster == cluster_id and j != i
                ]
                similar_steps = [reasoning_steps[j].content for j in similar_indices]
                processed_indices.update(similar_indices)

            # Create filtered step
            filtered_step = FilteredReasoningStep(
                step=step,
                cluster_id=cluster_id,
                quality_score=quality_scores[i],
                is_duplicate=len(similar_steps) > 0,
                similar_steps=similar_steps,
            )

            filtered_steps.append(filtered_step)
            processed_indices.add(i)

        return filtered_steps

    def _filter_by_length(
        self, reasoning_steps: List["ReasoningStep"]
    ) -> List["ReasoningStep"]:
        """Filter reasoning steps by length."""
        filtered = []
        for step in reasoning_steps:
            word_count = len(step.content.split())
            if self.min_length <= word_count <= self.max_length:
                filtered.append(step)
        return filtered

    def _assess_quality(self, step_contents: List[str]) -> List[float]:
        """Assess quality of reasoning steps."""
        quality_scores = []

        for content in step_contents:
            score = 0.0

            # Length-based scoring
            word_count = len(content.split())
            if word_count >= 5:
                score += 0.2
            if word_count >= 10:
                score += 0.1

            # Pattern-based scoring
            content_lower = content.lower()

            # High quality patterns
            for pattern in self.quality_patterns["high_quality"]:
                if re.search(pattern, content_lower):
                    score += 0.1

            # Reasoning indicators
            for pattern in self.quality_patterns["reasoning_indicators"]:
                if re.search(pattern, content_lower):
                    score += 0.15

            # Low quality penalties
            for pattern in self.quality_patterns["low_quality"]:
                if re.search(pattern, content):
                    score -= 0.2

            # Structure scoring
            if content[0].isupper() and content.endswith("."):
                score += 0.1

            # Normalize score
            score = max(0.0, min(1.0, score))
            quality_scores.append(score)

        return quality_scores

    def _find_duplicates(
        self, embeddings: np.ndarray, similarity_threshold: float
    ) -> Dict[int, int]:
        """Find duplicate reasoning steps using clustering."""
        if len(embeddings) < 2:
            return {}

        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings)

        # Use DBSCAN clustering to find duplicates
        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix

        # Set eps based on similarity threshold
        eps = 1 - similarity_threshold

        clustering = DBSCAN(eps=eps, min_samples=1, metric="precomputed")
        cluster_labels = clustering.fit_predict(distance_matrix)

        # Create mapping of indices to cluster IDs
        duplicate_groups = {}
        for i, cluster_id in enumerate(cluster_labels):
            if cluster_id != -1:  # Not noise
                duplicate_groups[i] = cluster_id

        return duplicate_groups


class SemanticFilter:
    """
    Main semantic filtering module that coordinates reasoning step filtering.

    This module provides a unified interface for removing duplicates and
    filtering low-quality reasoning steps using semantic similarity.
    """

    def __init__(
        self,
        filter_impl: Optional[BaseSemanticFilter] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the semantic filter.

        Args:
            filter_impl: Filter implementation (defaults to SentenceBERTFilter)
            config: Configuration dictionary
        """
        self.filter_impl = filter_impl or SentenceBERTFilter()
        self.config = config or {}

        # Load domain-specific configurations
        self.domain_configs = self._load_domain_configs()

    def _load_domain_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load domain-specific filtering configurations."""
        return {
            "causal": {
                "similarity_threshold": 0.75,
                "quality_threshold": 0.4,
                "min_length": 8,
            },
            "logical": {
                "similarity_threshold": 0.8,
                "quality_threshold": 0.5,
                "min_length": 6,
            },
            "arithmetic": {
                "similarity_threshold": 0.85,
                "quality_threshold": 0.3,
                "min_length": 5,
            },
            "general": {
                "similarity_threshold": 0.8,
                "quality_threshold": 0.3,
                "min_length": 5,
            },
        }

    def filter_reasons(
        self,
        reasoning_steps: List["ReasoningStep"],
        domain: Optional[str] = None,
        **kwargs,
    ) -> List["ReasoningStep"]:
        """
        Filter reasoning steps based on semantic similarity and quality.

        Args:
            reasoning_steps: List of reasoning steps to filter
            domain: Reasoning domain for domain-specific filtering
            **kwargs: Additional filtering parameters

        Returns:
            List of filtered reasoning steps
        """
        if not reasoning_steps:
            return []

        domain = domain or "general"

        # Get domain-specific configuration
        domain_config = self.domain_configs.get(domain, self.domain_configs["general"])

        # Merge with provided kwargs
        filter_params = {**domain_config, **kwargs}

        # Apply filtering
        filtered_steps = self.filter_impl.filter_reasons(
            reasoning_steps=reasoning_steps, **filter_params
        )

        # Extract original steps from filtered results
        return [filtered_step.step for filtered_step in filtered_steps]

    def get_filtering_stats(
        self,
        original_steps: List["ReasoningStep"],
        filtered_steps: List["ReasoningStep"],
    ) -> Dict[str, Any]:
        """Get statistics about the filtering process."""
        return {
            "original_count": len(original_steps),
            "filtered_count": len(filtered_steps),
            "duplicates_removed": len(original_steps) - len(filtered_steps),
            "retention_rate": (
                len(filtered_steps) / len(original_steps) if original_steps else 0
            ),
            "filter_type": type(self.filter_impl).__name__,
        }

    def analyze_quality_distribution(
        self, reasoning_steps: List["ReasoningStep"]
    ) -> Dict[str, Any]:
        """Analyze the quality distribution of reasoning steps."""
        if not reasoning_steps:
            return {}

        # Get quality scores
        step_contents = [step.content for step in reasoning_steps]
        quality_scores = self.filter_impl._assess_quality(step_contents)

        return {
            "mean_quality": np.mean(quality_scores),
            "median_quality": np.median(quality_scores),
            "std_quality": np.std(quality_scores),
            "min_quality": np.min(quality_scores),
            "max_quality": np.max(quality_scores),
            "high_quality_count": sum(1 for score in quality_scores if score >= 0.7),
            "medium_quality_count": sum(
                1 for score in quality_scores if 0.4 <= score < 0.7
            ),
            "low_quality_count": sum(1 for score in quality_scores if score < 0.4),
        }

    def find_similar_steps(
        self, reasoning_steps: List["ReasoningStep"], target_step: str, top_k: int = 5
    ) -> List[Tuple["ReasoningStep", float]]:
        """Find steps similar to a target step."""
        if not reasoning_steps:
            return []

        # Generate embeddings
        all_contents = [step.content for step in reasoning_steps] + [target_step]
        embeddings = self.filter_impl.embedding_model.encode(all_contents)

        # Compute similarities
        target_embedding = embeddings[-1].reshape(1, -1)
        step_embeddings = embeddings[:-1]

        similarities = cosine_similarity(target_embedding, step_embeddings)[0]

        # Get top-k similar steps
        similar_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            (reasoning_steps[i], similarities[i])
            for i in similar_indices
            if similarities[i] > 0.5  # Minimum similarity threshold
        ]
