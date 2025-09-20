"""
Task-Agnostic Input Interface

This module provides a unified interface for processing queries from any reasoning domain.
It handles input normalization, domain detection, and query preprocessing.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import re
import json


@dataclass
class ProcessedQuery:
    """Container for processed query information."""
    original_query: str
    normalized_query: str
    domain: Optional[str]
    query_type: str
    complexity: str
    metadata: Dict[str, Any]


class TaskAgnosticInterface:
    """
    Task-agnostic interface for processing queries from diverse reasoning domains.
    
    This module normalizes inputs, detects domains, and prepares queries for
    downstream processing by other CRLLM modules.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the task-agnostic interface.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.domain_patterns = self._initialize_domain_patterns()
        self.complexity_indicators = self._initialize_complexity_indicators()
        
    def _initialize_domain_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for domain detection."""
        return {
            'causal': [
                r'cause|effect|because|due to|leads to|results in|influences?|affects?',
                r'correlation|causation|relationship|impact|consequence',
                r'why|how does|what causes|what leads to'
            ],
            'logical': [
                r'if.*then|implies|logical|deduction|inference|conclusion',
                r'premise|assumption|therefore|thus|hence|follows',
                r'contradiction|paradox|valid|invalid|sound|unsound'
            ],
            'spatial': [
                r'left|right|above|below|behind|front|inside|outside',
                r'position|location|direction|distance|coordinates',
                r'geometry|shape|area|volume|perimeter|circumference'
            ],
            'arithmetic': [
                r'calculate|compute|solve|equation|formula|math',
                r'add|subtract|multiply|divide|sum|difference|product',
                r'number|digit|integer|decimal|fraction|percentage'
            ],
            'temporal': [
                r'before|after|during|while|when|time|sequence|order',
                r'past|present|future|chronological|timeline|schedule'
            ],
            'comparative': [
                r'compare|contrast|better|worse|superior|inferior',
                r'advantage|disadvantage|pros?|cons?|versus|vs'
            ]
        }
    
    def _initialize_complexity_indicators(self) -> Dict[str, List[str]]:
        """Initialize indicators for query complexity assessment."""
        return {
            'simple': [
                r'what is|who is|when is|where is',
                r'define|explain|describe',
                r'yes|no|true|false'
            ],
            'moderate': [
                r'how|why|analyze|evaluate|assess',
                r'compare|contrast|differentiate',
                r'step|process|method|approach'
            ],
            'complex': [
                r'design|create|develop|construct|build',
                r'optimize|maximize|minimize|improve',
                r'integrate|synthesize|combine|merge',
                r'multi-step|multi-hop|chain|sequence'
            ]
        }
    
    def process_input(
        self, 
        query: Union[str, Dict[str, Any]], 
        domain: Optional[str] = None
    ) -> ProcessedQuery:
        """
        Process input query and return structured information.
        
        Args:
            query: Input query (string or structured dict)
            domain: Optional domain specification
            
        Returns:
            ProcessedQuery object with normalized information
        """
        # Extract query text and metadata
        if isinstance(query, dict):
            query_text = query.get('text', str(query))
            metadata = query.get('metadata', {})
        else:
            query_text = str(query)
            metadata = {}
        
        # Normalize query
        normalized_query = self._normalize_query(query_text)
        
        # Detect domain if not provided
        if domain is None:
            domain = self._detect_domain(normalized_query)
        
        # Determine query type
        query_type = self._classify_query_type(normalized_query)
        
        # Assess complexity
        complexity = self._assess_complexity(normalized_query)
        
        return ProcessedQuery(
            original_query=query_text,
            normalized_query=normalized_query,
            domain=domain,
            query_type=query_type,
            complexity=complexity,
            metadata=metadata
        )
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query text for consistent processing."""
        # Convert to lowercase
        normalized = query.lower().strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove special characters but keep punctuation
        normalized = re.sub(r'[^\w\s\?\!\.\,\;\:\-]', '', normalized)
        
        return normalized
    
    def _detect_domain(self, query: str) -> Optional[str]:
        """Detect the reasoning domain from query text."""
        query_lower = query.lower()
        
        domain_scores = {}
        for domain, patterns in self.domain_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower))
                score += matches
            domain_scores[domain] = score
        
        # Return domain with highest score, or None if no clear match
        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            return best_domain if domain_scores[best_domain] > 0 else None
        
        return None
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what', 'who', 'when', 'where']):
            return 'factual'
        elif any(word in query_lower for word in ['how', 'why']):
            return 'explanatory'
        elif any(word in query_lower for word in ['compare', 'contrast', 'versus']):
            return 'comparative'
        elif any(word in query_lower for word in ['solve', 'calculate', 'compute']):
            return 'computational'
        elif any(word in query_lower for word in ['design', 'create', 'develop']):
            return 'creative'
        elif query_lower.endswith('?'):
            return 'question'
        else:
            return 'statement'
    
    def _assess_complexity(self, query: str) -> str:
        """Assess the complexity level of the query."""
        query_lower = query.lower()
        
        # Check for complex indicators first
        for complexity, indicators in self.complexity_indicators.items():
            for indicator in indicators:
                if re.search(indicator, query_lower):
                    if complexity == 'complex':
                        return 'complex'
                    elif complexity == 'moderate':
                        return 'moderate'
        
        # Check query length and structure
        word_count = len(query.split())
        if word_count > 20 or 'multi' in query_lower:
            return 'complex'
        elif word_count > 10:
            return 'moderate'
        else:
            return 'simple'
    
    def extract_entities(self, query: str) -> List[str]:
        """Extract potential entities from the query."""
        # Simple entity extraction based on capitalization and common patterns
        entities = []
        
        # Extract capitalized words (potential proper nouns)
        capitalized = re.findall(r'\b[A-Z][a-z]+\b', query)
        entities.extend(capitalized)
        
        # Extract quoted strings
        quoted = re.findall(r'"([^"]*)"', query)
        entities.extend(quoted)
        
        # Extract numbers
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', query)
        entities.extend(numbers)
        
        return list(set(entities))  # Remove duplicates
    
    def get_query_requirements(self, processed_query: ProcessedQuery) -> Dict[str, Any]:
        """Analyze query requirements for downstream processing."""
        requirements = {
            'needs_external_knowledge': self._needs_external_knowledge(processed_query),
            'needs_computation': self._needs_computation(processed_query),
            'needs_reasoning_chain': self._needs_reasoning_chain(processed_query),
            'estimated_reasoning_steps': self._estimate_reasoning_steps(processed_query),
            'suggested_modules': self._suggest_modules(processed_query)
        }
        return requirements
    
    def _needs_external_knowledge(self, processed_query: ProcessedQuery) -> bool:
        """Determine if query needs external knowledge retrieval."""
        query = processed_query.normalized_query
        
        # Check for indicators that suggest need for external knowledge
        external_indicators = [
            'current', 'recent', 'latest', 'news', 'research', 'study',
            'data', 'statistics', 'facts', 'information', 'knowledge'
        ]
        
        return any(indicator in query for indicator in external_indicators)
    
    def _needs_computation(self, processed_query: ProcessedQuery) -> bool:
        """Determine if query needs computational processing."""
        query = processed_query.normalized_query
        
        computational_indicators = [
            'calculate', 'compute', 'solve', 'equation', 'formula',
            'math', 'number', 'sum', 'total', 'average', 'percentage'
        ]
        
        return any(indicator in query for indicator in computational_indicators)
    
    def _needs_reasoning_chain(self, processed_query: ProcessedQuery) -> bool:
        """Determine if query needs multi-step reasoning."""
        return processed_query.complexity in ['moderate', 'complex']
    
    def _estimate_reasoning_steps(self, processed_query: ProcessedQuery) -> int:
        """Estimate the number of reasoning steps needed."""
        if processed_query.complexity == 'simple':
            return 1
        elif processed_query.complexity == 'moderate':
            return 3
        else:
            return 5
    
    def _suggest_modules(self, processed_query: ProcessedQuery) -> List[str]:
        """Suggest which modules should be used for this query."""
        suggestions = ['reason_sampler', 'semantic_filter', 'combinatorial_optimizer', 'reason_orderer', 'final_inference']
        
        if self._needs_external_knowledge(processed_query):
            suggestions.append('retrieval_module')
        
        if processed_query.complexity == 'complex':
            suggestions.append('reason_verifier')
        
        return suggestions
