"""
Reason Ordering Module

This module arranges selected reasoning steps into logical chains, supporting
both Chain-of-Thought and Tree-of-Thought patterns.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .reason_sampler import ReasoningStep

import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class OrderingStrategy(Enum):
    """Enumeration of ordering strategies."""

    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHT = "tree_of_thought"
    DEPENDENCY_BASED = "dependency_based"
    CONFIDENCE_BASED = "confidence_based"
    SEMANTIC_FLOW = "semantic_flow"


@dataclass
class OrderedReasoningStep:
    """Container for ordered reasoning step with position information."""

    step: "ReasoningStep"
    position: int
    parent_positions: List[int]
    child_positions: List[int]
    ordering_score: float
    metadata: Dict[str, Any]


@dataclass
class OrderingResult:
    """Container for ordering results."""

    ordered_steps: List[OrderedReasoningStep]
    ordering_strategy: str
    logical_flow_score: float
    ordering_time: float
    graph_structure: Optional[nx.DiGraph] = None


class BaseOrderer(ABC):
    """Abstract base class for ordering implementations."""

    @abstractmethod
    def order_reasons(
        self,
        reasoning_steps: List["ReasoningStep"],
        query: str,
        strategy: OrderingStrategy = OrderingStrategy.CHAIN_OF_THOUGHT,
        **kwargs,
    ) -> OrderingResult:
        """Order reasoning steps into logical chains."""
        pass


class SemanticFlowOrderer(BaseOrderer):
    """Semantic flow-based ordering implementation."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize semantic flow orderer.

        Args:
            embedding_model: Sentence transformer model for embeddings
            config: Additional configuration
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.config = config or {}

    def order_reasons(
        self,
        reasoning_steps: List["ReasoningStep"],
        query: str,
        strategy: OrderingStrategy = OrderingStrategy.CHAIN_OF_THOUGHT,
        **kwargs,
    ) -> OrderingResult:
        """Order reasoning steps using semantic flow analysis."""
        import time

        start_time = time.time()

        if not reasoning_steps:
            return OrderingResult(
                ordered_steps=[],
                ordering_strategy=strategy.value,
                logical_flow_score=0.0,
                ordering_time=0.0,
            )

        # Generate embeddings for all steps and query
        all_texts = [query] + [step.content for step in reasoning_steps]
        embeddings = self.embedding_model.encode(all_texts)
        query_embedding = embeddings[0]
        step_embeddings = embeddings[1:]

        # Order based on strategy
        if strategy == OrderingStrategy.CHAIN_OF_THOUGHT:
            ordered_steps = self._order_chain_of_thought(
                reasoning_steps, query_embedding, step_embeddings
            )
        elif strategy == OrderingStrategy.TREE_OF_THOUGHT:
            ordered_steps, graph = self._order_tree_of_thought(
                reasoning_steps, query_embedding, step_embeddings
            )
        elif strategy == OrderingStrategy.DEPENDENCY_BASED:
            ordered_steps = self._order_dependency_based(
                reasoning_steps, query_embedding, step_embeddings
            )
        elif strategy == OrderingStrategy.CONFIDENCE_BASED:
            ordered_steps = self._order_confidence_based(reasoning_steps)
        elif strategy == OrderingStrategy.SEMANTIC_FLOW:
            ordered_steps = self._order_semantic_flow(
                reasoning_steps, query_embedding, step_embeddings
            )
        else:
            ordered_steps = self._order_chain_of_thought(
                reasoning_steps, query_embedding, step_embeddings
            )

        # Compute logical flow score
        logical_flow_score = self._compute_logical_flow_score(ordered_steps)

        ordering_time = time.time() - start_time

        return OrderingResult(
            ordered_steps=ordered_steps,
            ordering_strategy=strategy.value,
            logical_flow_score=logical_flow_score,
            ordering_time=ordering_time,
        )

    def _order_chain_of_thought(
        self,
        reasoning_steps: List["ReasoningStep"],
        query_embedding: np.ndarray,
        step_embeddings: np.ndarray,
    ) -> List[OrderedReasoningStep]:
        """Order steps in a linear chain-of-thought pattern."""
        n = len(reasoning_steps)

        # Compute similarity to query
        query_similarities = cosine_similarity(
            query_embedding.reshape(1, -1), step_embeddings
        )[0]

        # Start with step most similar to query
        start_idx = np.argmax(query_similarities)

        # Build chain by finding next most similar step
        ordered_indices = [start_idx]
        remaining_indices = list(range(n))
        remaining_indices.remove(start_idx)

        while remaining_indices:
            current_embedding = step_embeddings[ordered_indices[-1]]

            # Find next step with highest similarity to current
            similarities = cosine_similarity(
                current_embedding.reshape(1, -1), step_embeddings[remaining_indices]
            )[0]

            next_idx = remaining_indices[np.argmax(similarities)]
            ordered_indices.append(next_idx)
            remaining_indices.remove(next_idx)

        # Create ordered steps
        ordered_steps = []
        for i, idx in enumerate(ordered_indices):
            step = reasoning_steps[idx]
            parent_positions = [i - 1] if i > 0 else []
            child_positions = [i + 1] if i < len(ordered_indices) - 1 else []

            ordered_steps.append(
                OrderedReasoningStep(
                    step=step,
                    position=i,
                    parent_positions=parent_positions,
                    child_positions=child_positions,
                    ordering_score=query_similarities[idx],
                    metadata={"original_index": idx},
                )
            )

        return ordered_steps

    def _order_tree_of_thought(
        self,
        reasoning_steps: List["ReasoningStep"],
        query_embedding: np.ndarray,
        step_embeddings: np.ndarray,
    ) -> Tuple[List[OrderedReasoningStep], nx.DiGraph]:
        """Order steps in a tree-of-thought pattern."""
        n = len(reasoning_steps)

        # Create graph
        G = nx.DiGraph()

        # Add nodes
        for i, step in enumerate(reasoning_steps):
            G.add_node(i, step=step, embedding=step_embeddings[i])

        # Compute similarities between all pairs
        similarity_matrix = cosine_similarity(step_embeddings)

        # Add edges based on similarity and reasoning type compatibility
        for i in range(n):
            for j in range(n):
                if i != j:
                    similarity = similarity_matrix[i, j]
                    type_compatibility = self._compute_type_compatibility(
                        reasoning_steps[i], reasoning_steps[j]
                    )

                    # Edge weight combines similarity and type compatibility
                    edge_weight = 0.7 * similarity + 0.3 * type_compatibility

                    if edge_weight > 0.5:  # Threshold for adding edge
                        G.add_edge(i, j, weight=edge_weight)

        # Find root nodes (steps most similar to query)
        query_similarities = cosine_similarity(
            query_embedding.reshape(1, -1), step_embeddings
        )[0]
        root_candidates = np.argsort(query_similarities)[-2:]  # Top 2 as roots

        # Build tree structure
        ordered_steps = []
        position_map = {}

        # Process root nodes first
        for root_idx in root_candidates:
            if root_idx not in position_map:
                pos = len(ordered_steps)
                position_map[root_idx] = pos

                step = reasoning_steps[root_idx]
                ordered_steps.append(
                    OrderedReasoningStep(
                        step=step,
                        position=pos,
                        parent_positions=[],
                        child_positions=[],
                        ordering_score=query_similarities[root_idx],
                        metadata={"original_index": root_idx, "is_root": True},
                    )
                )

        # Process remaining nodes in topological order
        remaining_nodes = set(range(n)) - set(position_map.keys())

        while remaining_nodes:
            # Find nodes that can be added (have connections to already added nodes)
            candidates = []
            for node in remaining_nodes:
                for parent in position_map.keys():
                    if G.has_edge(parent, node):
                        candidates.append((node, parent))

            if not candidates:
                # No more connections, add remaining nodes arbitrarily
                for node in remaining_nodes:
                    pos = len(ordered_steps)
                    position_map[node] = pos

                    step = reasoning_steps[node]
                    ordered_steps.append(
                        OrderedReasoningStep(
                            step=step,
                            position=pos,
                            parent_positions=[],
                            child_positions=[],
                            ordering_score=0.5,
                            metadata={"original_index": node},
                        )
                    )
                break

            # Add best candidate
            node, parent = candidates[0]
            pos = len(ordered_steps)
            position_map[node] = pos

            step = reasoning_steps[node]
            parent_pos = position_map[parent]

            ordered_steps.append(
                OrderedReasoningStep(
                    step=step,
                    position=pos,
                    parent_positions=[parent_pos],
                    child_positions=[],
                    ordering_score=similarity_matrix[parent, node],
                    metadata={"original_index": node},
                )
            )

            # Update parent's child list
            ordered_steps[parent_pos].child_positions.append(pos)

            remaining_nodes.remove(node)

        return ordered_steps, G

    def _order_dependency_based(
        self,
        reasoning_steps: List["ReasoningStep"],
        query_embedding: np.ndarray,
        step_embeddings: np.ndarray,
    ) -> List[OrderedReasoningStep]:
        """Order steps based on logical dependencies."""
        n = len(reasoning_steps)

        # Analyze dependencies between steps
        dependencies = self._analyze_dependencies(reasoning_steps)

        # Topological sort
        ordered_indices = self._topological_sort(dependencies, n)

        # Create ordered steps
        ordered_steps = []
        for i, idx in enumerate(ordered_indices):
            step = reasoning_steps[idx]

            # Find parent positions
            parent_positions = []
            for dep_idx in dependencies.get(idx, []):
                if dep_idx in ordered_indices:
                    parent_positions.append(ordered_indices.index(dep_idx))

            # Find child positions
            child_positions = []
            for j, other_idx in enumerate(ordered_indices):
                if other_idx in dependencies and idx in dependencies[other_idx]:
                    child_positions.append(j)

            ordered_steps.append(
                OrderedReasoningStep(
                    step=step,
                    position=i,
                    parent_positions=parent_positions,
                    child_positions=child_positions,
                    ordering_score=1.0 - (i / n),  # Higher score for earlier steps
                    metadata={"original_index": idx},
                )
            )

        return ordered_steps

    def _order_confidence_based(
        self, reasoning_steps: List["ReasoningStep"]
    ) -> List[OrderedReasoningStep]:
        """Order steps based on confidence scores."""
        # Sort by confidence (descending)
        sorted_steps = sorted(reasoning_steps, key=lambda x: x.confidence, reverse=True)

        # Create ordered steps
        ordered_steps = []
        for i, step in enumerate(sorted_steps):
            ordered_steps.append(
                OrderedReasoningStep(
                    step=step,
                    position=i,
                    parent_positions=[i - 1] if i > 0 else [],
                    child_positions=[i + 1] if i < len(sorted_steps) - 1 else [],
                    ordering_score=step.confidence,
                    metadata={"original_index": reasoning_steps.index(step)},
                )
            )

        return ordered_steps

    def _order_semantic_flow(
        self,
        reasoning_steps: List["ReasoningStep"],
        query_embedding: np.ndarray,
        step_embeddings: np.ndarray,
    ) -> List[OrderedReasoningStep]:
        """Order steps based on semantic flow analysis."""
        n = len(reasoning_steps)

        # Compute flow scores between all pairs
        flow_scores = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    # Semantic similarity
                    similarity = cosine_similarity(
                        step_embeddings[i].reshape(1, -1),
                        step_embeddings[j].reshape(1, -1),
                    )[0, 0]

                    # Reasoning type compatibility
                    type_compat = self._compute_type_compatibility(
                        reasoning_steps[i], reasoning_steps[j]
                    )

                    # Temporal flow (earlier steps should flow to later ones)
                    temporal_bonus = 0.1 if i < j else 0.0

                    flow_scores[i, j] = similarity + type_compat + temporal_bonus

        # Find optimal ordering using TSP-like approach
        ordered_indices = self._solve_flow_optimization(flow_scores)

        # Create ordered steps
        ordered_steps = []
        for i, idx in enumerate(ordered_indices):
            step = reasoning_steps[idx]

            # Find best parent and child based on flow scores
            parent_positions = []
            child_positions = []

            if i > 0:
                parent_idx = ordered_indices[i - 1]
                if flow_scores[parent_idx, idx] > 0.5:
                    parent_positions = [i - 1]

            if i < len(ordered_indices) - 1:
                child_idx = ordered_indices[i + 1]
                if flow_scores[idx, child_idx] > 0.5:
                    child_positions = [i + 1]

            ordered_steps.append(
                OrderedReasoningStep(
                    step=step,
                    position=i,
                    parent_positions=parent_positions,
                    child_positions=child_positions,
                    ordering_score=np.mean(flow_scores[idx, :]),
                    metadata={"original_index": idx},
                )
            )

        return ordered_steps

    def _compute_type_compatibility(
        self, step1: "ReasoningStep", step2: "ReasoningStep"
    ) -> float:
        """Compute compatibility between reasoning types."""
        type_pairs = {
            ("assumption", "analysis"): 0.9,
            ("analysis", "logical"): 0.8,
            ("logical", "causal"): 0.7,
            ("causal", "computational"): 0.6,
            ("computational", "general"): 0.5,
        }

        # Check both directions
        forward = type_pairs.get((step1.reasoning_type, step2.reasoning_type), 0.3)
        backward = type_pairs.get((step2.reasoning_type, step1.reasoning_type), 0.3)

        return max(forward, backward)

    def _analyze_dependencies(
        self, reasoning_steps: List["ReasoningStep"]
    ) -> Dict[int, List[int]]:
        """Analyze logical dependencies between reasoning steps."""
        dependencies = {}

        for i, step1 in enumerate(reasoning_steps):
            deps = []
            for j, step2 in enumerate(reasoning_steps):
                if i != j and self._is_dependent(step1, step2):
                    deps.append(j)
            if deps:
                dependencies[i] = deps

        return dependencies

    def _is_dependent(self, step1: "ReasoningStep", step2: "ReasoningStep") -> bool:
        """Check if step1 depends on step2."""
        # Simple keyword-based dependency detection
        step1_words = set(step1.content.lower().split())
        step2_words = set(step2.content.lower().split())

        # Check for reference words
        reference_words = ["this", "that", "it", "the above", "previously", "earlier"]

        for ref_word in reference_words:
            if ref_word in step1_words:
                # Check if step2 contains concepts that step1 references
                overlap = len(step1_words.intersection(step2_words))
                if overlap > 2:  # Threshold for dependency
                    return True

        return False

    def _topological_sort(
        self, dependencies: Dict[int, List[int]], n: int
    ) -> List[int]:
        """Perform topological sort of dependencies."""
        # Kahn's algorithm
        in_degree = [0] * n
        for deps in dependencies.values():
            for dep in deps:
                in_degree[dep] += 1

        queue = [i for i in range(n) if in_degree[i] == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            # Remove this node and update in-degrees
            for i in range(n):
                if i in dependencies.get(node, []):
                    in_degree[i] -= 1
                    if in_degree[i] == 0:
                        queue.append(i)

        return result

    def _solve_flow_optimization(self, flow_scores: np.ndarray) -> List[int]:
        """Solve flow optimization problem (simplified TSP)."""
        n = flow_scores.shape[0]

        # Greedy approach: start with highest scoring pair
        best_score = -1
        best_start = 0

        for i in range(n):
            for j in range(n):
                if i != j and flow_scores[i, j] > best_score:
                    best_score = flow_scores[i, j]
                    best_start = i

        # Build path greedily
        path = [best_start]
        remaining = set(range(n)) - {best_start}

        while remaining:
            current = path[-1]
            best_next = max(remaining, key=lambda x: flow_scores[current, x])
            path.append(best_next)
            remaining.remove(best_next)

        return path

    def _compute_logical_flow_score(
        self, ordered_steps: List[OrderedReasoningStep]
    ) -> float:
        """Compute score for logical flow of ordered steps."""
        if len(ordered_steps) < 2:
            return 1.0

        # Compute average transition quality
        transition_scores = []

        for i in range(len(ordered_steps) - 1):
            current = ordered_steps[i]
            next_step = ordered_steps[i + 1]

            # Compute transition score
            score = 0.5  # Base score

            # Bonus for logical connections
            if current.child_positions and i + 1 in current.child_positions:
                score += 0.3

            # Bonus for type compatibility
            type_compat = self._compute_type_compatibility(current.step, next_step.step)
            score += type_compat * 0.2

            transition_scores.append(score)

        return np.mean(transition_scores) if transition_scores else 0.0


class ReasonOrderer:
    """
    Main reason ordering module that coordinates ordering strategies.

    This module provides a unified interface for arranging reasoning steps
    into logical chains using various ordering strategies.
    """

    def __init__(
        self,
        orderer: Optional[BaseOrderer] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the reason orderer.

        Args:
            orderer: Orderer implementation (defaults to SemanticFlowOrderer)
            config: Configuration dictionary
        """
        self.orderer = orderer or SemanticFlowOrderer()
        self.config = config or {}

        # Load domain-specific configurations
        self.domain_configs = self._load_domain_configs()

    def _load_domain_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load domain-specific ordering configurations."""
        return {
            "causal": {
                "strategy": OrderingStrategy.CHAIN_OF_THOUGHT,
                "max_branching": 2,
            },
            "logical": {
                "strategy": OrderingStrategy.DEPENDENCY_BASED,
                "max_branching": 1,
            },
            "arithmetic": {
                "strategy": OrderingStrategy.CHAIN_OF_THOUGHT,
                "max_branching": 1,
            },
            "general": {"strategy": OrderingStrategy.SEMANTIC_FLOW, "max_branching": 2},
        }

    def order_reasons(
        self,
        reasoning_steps: List["ReasoningStep"],
        query: str,
        domain: Optional[str] = None,
        **kwargs,
    ) -> List[str]:
        """
        Order reasoning steps into logical chains.

        Args:
            reasoning_steps: List of reasoning steps to order
            query: Original query for context
            domain: Reasoning domain for domain-specific ordering
            **kwargs: Additional ordering parameters

        Returns:
            List of ordered reasoning step contents
        """
        if not reasoning_steps:
            return []

        domain = domain or "general"

        # Get domain-specific configuration
        domain_config = self.domain_configs.get(domain, self.domain_configs["general"])

        # Merge with provided kwargs
        order_params = {**domain_config, **kwargs}

        # Run ordering
        result = self.orderer.order_reasons(
            reasoning_steps=reasoning_steps, query=query, **order_params
        )

        # Extract ordered step contents
        return [ordered_step.step.content for ordered_step in result.ordered_steps]

    def get_ordering_stats(
        self, original_steps: List["ReasoningStep"], ordered_steps: List[str]
    ) -> Dict[str, Any]:
        """Get statistics about the ordering process."""
        return {
            "original_count": len(original_steps),
            "ordered_count": len(ordered_steps),
            "orderer_type": type(self.orderer).__name__,
            "available_strategies": [strategy.value for strategy in OrderingStrategy],
        }
