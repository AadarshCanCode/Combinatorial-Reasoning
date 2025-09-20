"""
Combinatorial Optimizer Module

This module solves a QUBO (Quadratic Unconstrained Binary Optimization) problem
to select a diverse, high-utility subset of reasoning steps that maximizes
both individual quality and collective diversity.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import minimize
import dimod
from dwave.system import DWaveSampler, EmbeddingComposite
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import RecursiveMinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.algorithms import QAOA
from qiskit import Aer


@dataclass
class OptimizationResult:
    """Container for optimization results."""
    selected_indices: List[int]
    selected_steps: List['ReasoningStep']
    objective_value: float
    diversity_score: float
    utility_score: float
    optimization_time: float
    solver_used: str


class BaseOptimizer(ABC):
    """Abstract base class for optimization implementations."""
    
    @abstractmethod
    def optimize(
        self,
        reasoning_steps: List['ReasoningStep'],
        query: str,
        max_selections: int = 5,
        **kwargs
    ) -> OptimizationResult:
        """Optimize selection of reasoning steps."""
        pass


class QUBOOptimizer(BaseOptimizer):
    """QUBO-based optimizer using classical and quantum solvers."""
    
    def __init__(
        self,
        solver_type: str = "classical",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize QUBO optimizer.
        
        Args:
            solver_type: Type of solver ('classical', 'quantum', 'qaoa')
            config: Additional configuration
        """
        self.solver_type = solver_type
        self.config = config or {}
        
        # Initialize solver based on type
        self.solver = self._initialize_solver()
    
    def _initialize_solver(self):
        """Initialize the appropriate solver."""
        if self.solver_type == "quantum":
            try:
                return EmbeddingComposite(DWaveSampler())
            except Exception as e:
                print(f"Warning: Quantum solver not available, falling back to classical: {e}")
                return "classical"
        elif self.solver_type == "qaoa":
            return "qaoa"
        else:
            return "classical"
    
    def optimize(
        self,
        reasoning_steps: List['ReasoningStep'],
        query: str,
        max_selections: int = 5,
        **kwargs
    ) -> OptimizationResult:
        """Optimize selection using QUBO formulation."""
        import time
        start_time = time.time()
        
        if not reasoning_steps:
            return OptimizationResult(
                selected_indices=[],
                selected_steps=[],
                objective_value=0.0,
                diversity_score=0.0,
                utility_score=0.0,
                optimization_time=0.0,
                solver_used=self.solver_type
            )
        
        # Limit max_selections to available steps
        max_selections = min(max_selections, len(reasoning_steps))
        
        # Compute utility scores for each step
        utility_scores = self._compute_utility_scores(reasoning_steps, query)
        
        # Compute diversity matrix
        diversity_matrix = self._compute_diversity_matrix(reasoning_steps)
        
        # Formulate QUBO problem
        qubo_matrix = self._formulate_qubo(utility_scores, diversity_matrix, max_selections)
        
        # Solve QUBO problem
        if self.solver_type == "classical":
            selected_indices = self._solve_classical(qubo_matrix, max_selections)
        elif self.solver_type == "quantum":
            selected_indices = self._solve_quantum(qubo_matrix, max_selections)
        elif self.solver_type == "qaoa":
            selected_indices = self._solve_qaoa(qubo_matrix, max_selections)
        else:
            selected_indices = self._solve_classical(qubo_matrix, max_selections)
        
        # Compute result metrics
        selected_steps = [reasoning_steps[i] for i in selected_indices]
        objective_value = self._compute_objective_value(selected_indices, utility_scores, diversity_matrix)
        diversity_score = self._compute_diversity_score(selected_indices, diversity_matrix)
        utility_score = np.mean([utility_scores[i] for i in selected_indices]) if selected_indices else 0.0
        
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            selected_indices=selected_indices,
            selected_steps=selected_steps,
            objective_value=objective_value,
            diversity_score=diversity_score,
            utility_score=utility_score,
            optimization_time=optimization_time,
            solver_used=self.solver_type
        )
    
    def _compute_utility_scores(
        self,
        reasoning_steps: List['ReasoningStep'],
        query: str
    ) -> np.ndarray:
        """Compute utility scores for reasoning steps."""
        scores = []
        
        for step in reasoning_steps:
            score = 0.0
            
            # Base confidence score
            score += step.confidence * 0.3
            
            # Length-based score (prefer moderate length)
            word_count = len(step.content.split())
            if 5 <= word_count <= 20:
                score += 0.2
            elif word_count > 20:
                score += 0.1
            
            # Reasoning type bonus
            type_bonuses = {
                'causal': 0.15,
                'logical': 0.15,
                'computational': 0.1,
                'analysis': 0.1,
                'assumption': 0.05,
                'general': 0.05
            }
            score += type_bonuses.get(step.reasoning_type, 0.05)
            
            # Query relevance (simple keyword matching)
            query_words = set(query.lower().split())
            step_words = set(step.content.lower().split())
            overlap = len(query_words.intersection(step_words))
            if overlap > 0:
                score += min(0.2, overlap * 0.05)
            
            # Normalize score
            score = max(0.0, min(1.0, score))
            scores.append(score)
        
        return np.array(scores)
    
    def _compute_diversity_matrix(
        self,
        reasoning_steps: List['ReasoningStep']
    ) -> np.ndarray:
        """Compute diversity matrix between reasoning steps."""
        n = len(reasoning_steps)
        diversity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                # Compute semantic diversity
                semantic_diversity = self._compute_semantic_diversity(
                    reasoning_steps[i].content,
                    reasoning_steps[j].content
                )
                
                # Compute structural diversity
                structural_diversity = self._compute_structural_diversity(
                    reasoning_steps[i],
                    reasoning_steps[j]
                )
                
                # Combine diversities
                total_diversity = 0.7 * semantic_diversity + 0.3 * structural_diversity
                diversity_matrix[i, j] = total_diversity
                diversity_matrix[j, i] = total_diversity
        
        return diversity_matrix
    
    def _compute_semantic_diversity(self, text1: str, text2: str) -> float:
        """Compute semantic diversity between two texts."""
        # Simple word-based diversity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard diversity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return 1.0 - (intersection / union) if union > 0 else 0.0
    
    def _compute_structural_diversity(
        self,
        step1: 'ReasoningStep',
        step2: 'ReasoningStep'
    ) -> float:
        """Compute structural diversity between two reasoning steps."""
        # Different reasoning types
        if step1.reasoning_type != step2.reasoning_type:
            return 1.0
        
        # Different confidence levels
        confidence_diff = abs(step1.confidence - step2.confidence)
        
        # Different lengths
        len1 = len(step1.content.split())
        len2 = len(step2.content.split())
        length_diff = abs(len1 - len2) / max(len1, len2, 1)
        
        return 0.5 * confidence_diff + 0.5 * length_diff
    
    def _formulate_qubo(
        self,
        utility_scores: np.ndarray,
        diversity_matrix: np.ndarray,
        max_selections: int
    ) -> np.ndarray:
        """Formulate QUBO problem for optimization."""
        n = len(utility_scores)
        
        # QUBO matrix: Q[i,j] = coefficient for x[i] * x[j]
        Q = np.zeros((n, n))
        
        # Linear terms (utility scores)
        for i in range(n):
            Q[i, i] = -utility_scores[i]
        
        # Quadratic terms (diversity rewards)
        alpha = self.config.get('diversity_weight', 0.5)
        for i in range(n):
            for j in range(i + 1, n):
                diversity_reward = alpha * diversity_matrix[i, j]
                Q[i, j] = diversity_reward
                Q[j, i] = diversity_reward
        
        # Constraint: exactly max_selections variables should be 1
        # This is handled in the solver
        
        return Q
    
    def _solve_classical(
        self,
        qubo_matrix: np.ndarray,
        max_selections: int
    ) -> List[int]:
        """Solve QUBO using classical optimization."""
        n = qubo_matrix.shape[0]
        
        # Use scipy optimization
        def objective(x):
            return x.T @ qubo_matrix @ x
        
        # Constraints: exactly max_selections variables should be 1
        from scipy.optimize import LinearConstraint
        constraint = LinearConstraint(
            np.ones(n), max_selections, max_selections
        )
        
        # Bounds: binary variables
        bounds = [(0, 1) for _ in range(n)]
        
        # Initial guess
        x0 = np.zeros(n)
        x0[:max_selections] = 1
        
        # Solve
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint
        )
        
        # Extract selected indices
        selected_indices = [i for i, val in enumerate(result.x) if val > 0.5]
        
        return selected_indices[:max_selections]
    
    def _solve_quantum(
        self,
        qubo_matrix: np.ndarray,
        max_selections: int
    ) -> List[int]:
        """Solve QUBO using quantum annealing."""
        n = qubo_matrix.shape[0]
        
        # Create binary quadratic model
        bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(qubo_matrix)
        
        # Add constraint for exact number of selections
        # This is a simplified approach - in practice, you'd use penalty methods
        constraint_strength = 10.0
        for i in range(n):
            for j in range(i + 1, n):
                bqm.add_interaction(i, j, constraint_strength)
        
        # Sample
        if isinstance(self.solver, str):
            # Fallback to classical if quantum solver not available
            return self._solve_classical(qubo_matrix, max_selections)
        
        try:
            sampleset = self.solver.sample(bqm, num_reads=100)
            best_sample = sampleset.first.sample
            
            # Extract selected indices
            selected_indices = [i for i, val in best_sample.items() if val == 1]
            
            return selected_indices[:max_selections]
        except Exception as e:
            print(f"Quantum solving failed, falling back to classical: {e}")
            return self._solve_classical(qubo_matrix, max_selections)
    
    def _solve_qaoa(
        self,
        qubo_matrix: np.ndarray,
        max_selections: int
    ) -> List[int]:
        """Solve QUBO using QAOA."""
        n = qubo_matrix.shape[0]
        
        # Create quadratic program
        qp = QuadraticProgram()
        
        # Add binary variables
        for i in range(n):
            qp.binary_var(name=f'x_{i}')
        
        # Add objective
        for i in range(n):
            for j in range(n):
                if i == j:
                    qp.minimize(linear={f'x_{i}': qubo_matrix[i, i]})
                else:
                    qp.minimize(quadratic={(f'x_{i}', f'x_{j}'): qubo_matrix[i, j]})
        
        # Add constraint for exact number of selections
        qp.linear_constraint(
            linear={f'x_{i}': 1 for i in range(n)},
            sense='==',
            rhs=max_selections
        )
        
        # Convert to QUBO
        converter = QuadraticProgramToQubo()
        qubo = converter.convert(qp)
        
        # Solve with QAOA
        qaoa = QAOA(quantum_instance=Aer.get_backend('qasm_simulator'))
        optimizer = RecursiveMinimumEigenOptimizer(qaoa)
        result = optimizer.solve(qubo)
        
        # Extract selected indices
        selected_indices = []
        for i in range(n):
            if result.x[i] > 0.5:
                selected_indices.append(i)
        
        return selected_indices[:max_selections]
    
    def _compute_objective_value(
        self,
        selected_indices: List[int],
        utility_scores: np.ndarray,
        diversity_matrix: np.ndarray
    ) -> float:
        """Compute the objective value for selected indices."""
        if not selected_indices:
            return 0.0
        
        # Utility component
        utility_value = sum(utility_scores[i] for i in selected_indices)
        
        # Diversity component
        diversity_value = 0.0
        for i in selected_indices:
            for j in selected_indices:
                if i < j:
                    diversity_value += diversity_matrix[i, j]
        
        return utility_value + diversity_value
    
    def _compute_diversity_score(
        self,
        selected_indices: List[int],
        diversity_matrix: np.ndarray
    ) -> float:
        """Compute average diversity score for selected indices."""
        if len(selected_indices) < 2:
            return 0.0
        
        total_diversity = 0.0
        count = 0
        
        for i in selected_indices:
            for j in selected_indices:
                if i < j:
                    total_diversity += diversity_matrix[i, j]
                    count += 1
        
        return total_diversity / count if count > 0 else 0.0


class CombinatorialOptimizer:
    """
    Main combinatorial optimizer module that coordinates optimization strategies.
    
    This module provides a unified interface for selecting optimal subsets of
    reasoning steps using QUBO-based optimization.
    """
    
    def __init__(
        self,
        optimizer: Optional[BaseOptimizer] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the combinatorial optimizer.
        
        Args:
            optimizer: Optimizer implementation (defaults to QUBOOptimizer)
            config: Configuration dictionary
        """
        self.optimizer = optimizer or QUBOOptimizer()
        self.config = config or {}
        
        # Load domain-specific configurations
        self.domain_configs = self._load_domain_configs()
    
    def _load_domain_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load domain-specific optimization configurations."""
        return {
            'causal': {
                'max_selections': 6,
                'diversity_weight': 0.6,
                'utility_weight': 0.4
            },
            'logical': {
                'max_selections': 5,
                'diversity_weight': 0.5,
                'utility_weight': 0.5
            },
            'arithmetic': {
                'max_selections': 4,
                'diversity_weight': 0.4,
                'utility_weight': 0.6
            },
            'general': {
                'max_selections': 5,
                'diversity_weight': 0.5,
                'utility_weight': 0.5
            }
        }
    
    def optimize_selection(
        self,
        reasoning_steps: List['ReasoningStep'],
        query: str,
        domain: Optional[str] = None,
        **kwargs
    ) -> List['ReasoningStep']:
        """
        Optimize selection of reasoning steps.
        
        Args:
            reasoning_steps: List of reasoning steps to optimize
            query: Original query for context
            domain: Reasoning domain for domain-specific optimization
            **kwargs: Additional optimization parameters
            
        Returns:
            List of optimally selected reasoning steps
        """
        if not reasoning_steps:
            return []
        
        domain = domain or 'general'
        
        # Get domain-specific configuration
        domain_config = self.domain_configs.get(domain, self.domain_configs['general'])
        
        # Merge with provided kwargs
        opt_params = {**domain_config, **kwargs}
        
        # Run optimization
        result = self.optimizer.optimize(
            reasoning_steps=reasoning_steps,
            query=query,
            **opt_params
        )
        
        return result.selected_steps
    
    def get_optimization_stats(
        self,
        original_steps: List['ReasoningStep'],
        selected_steps: List['ReasoningStep']
    ) -> Dict[str, Any]:
        """Get statistics about the optimization process."""
        return {
            'original_count': len(original_steps),
            'selected_count': len(selected_steps),
            'reduction_ratio': len(selected_steps) / len(original_steps) if original_steps else 0,
            'optimizer_type': type(self.optimizer).__name__,
            'solver_type': getattr(self.optimizer, 'solver_type', 'unknown')
        }
