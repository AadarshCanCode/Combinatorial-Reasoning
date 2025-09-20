"""
Reason Verifier Module

This module uses theorem provers (like Z3) to filter inconsistent reasoning chains
and verify logical consistency of reasoning steps.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import re
import z3
from enum import Enum


class VerificationLevel(Enum):
    """Enumeration of verification levels."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    STRICT = "strict"


@dataclass
class VerificationResult:
    """Container for verification results."""
    is_consistent: bool
    consistency_score: float
    inconsistencies: List[str]
    verified_steps: List[str]
    verification_time: float
    verification_level: str
    metadata: Dict[str, Any]


class BaseVerifier(ABC):
    """Abstract base class for verification implementations."""
    
    @abstractmethod
    def verify_reasons(
        self,
        reasoning_steps: List[str],
        query: str,
        verification_level: VerificationLevel = VerificationLevel.BASIC,
        **kwargs
    ) -> VerificationResult:
        """Verify consistency of reasoning steps."""
        pass


class Z3Verifier(BaseVerifier):
    """Z3-based reason verifier implementation."""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Z3 verifier.
        
        Args:
            config: Additional configuration
        """
        self.config = config or {}
        
        # Initialize Z3 solver
        self.solver = z3.Solver()
        
        # Load verification patterns
        self.verification_patterns = self._load_verification_patterns()
    
    def _load_verification_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for different types of logical verification."""
        return {
            'logical_operators': [
                r'\b(and|or|not|if|then|iff|implies?)\b',
                r'\b(all|some|exists?|for all|there exists)\b',
                r'\b(necessary|sufficient|required|needed)\b'
            ],
            'quantitative_relations': [
                r'\b(equals?|=\s*|is equal to)\b',
                r'\b(greater than|>\s*|less than|<\s*|>=|<=)\b',
                r'\b(increases?|decreases?|more|less|higher|lower)\b'
            ],
            'causal_relations': [
                r'\b(causes?|leads to|results in|produces?|creates?)\b',
                r'\b(because|due to|since|as a result)\b',
                r'\b(prevents?|stops?|blocks?|inhibits?)\b'
            ],
            'temporal_relations': [
                r'\b(before|after|during|while|when|then|next|finally)\b',
                r'\b(earlier|later|previously|subsequently)\b',
                r'\b(first|second|third|last|initially|ultimately)\b'
            ]
        }
    
    def verify_reasons(
        self,
        reasoning_steps: List[str],
        query: str,
        verification_level: VerificationLevel = VerificationLevel.BASIC,
        **kwargs
    ) -> VerificationResult:
        """Verify reasoning steps using Z3 theorem prover."""
        import time
        start_time = time.time()
        
        if not reasoning_steps:
            return VerificationResult(
                is_consistent=True,
                consistency_score=1.0,
                inconsistencies=[],
                verified_steps=[],
                verification_time=0.0,
                verification_level=verification_level.value,
                metadata={'error': 'empty_reasoning_steps'}
            )
        
        # Parse reasoning steps into logical formulas
        formulas = self._parse_reasoning_steps(reasoning_steps, verification_level)
        
        # Check for logical consistency
        consistency_result = self._check_logical_consistency(formulas, verification_level)
        
        # Check for contradictions
        contradiction_result = self._check_contradictions(formulas, verification_level)
        
        # Check for circular reasoning
        circularity_result = self._check_circular_reasoning(reasoning_steps)
        
        # Combine results
        is_consistent = (
            consistency_result['is_consistent'] and
            contradiction_result['is_consistent'] and
            circularity_result['is_consistent']
        )
        
        consistency_score = (
            consistency_result['score'] * 0.4 +
            contradiction_result['score'] * 0.3 +
            circularity_result['score'] * 0.3
        )
        
        inconsistencies = (
            consistency_result['issues'] +
            contradiction_result['issues'] +
            circularity_result['issues']
        )
        
        verified_steps = reasoning_steps if is_consistent else []
        
        verification_time = time.time() - start_time
        
        return VerificationResult(
            is_consistent=is_consistent,
            consistency_score=consistency_score,
            inconsistencies=inconsistencies,
            verified_steps=verified_steps,
            verification_time=verification_time,
            verification_level=verification_level.value,
            metadata={
                'formulas_parsed': len(formulas),
                'consistency_details': consistency_result,
                'contradiction_details': contradiction_result,
                'circularity_details': circularity_result
            }
        )
    
    def _parse_reasoning_steps(
        self,
        reasoning_steps: List[str],
        verification_level: VerificationLevel
    ) -> List[Dict[str, Any]]:
        """Parse reasoning steps into logical formulas."""
        formulas = []
        
        for i, step in enumerate(reasoning_steps):
            formula = {
                'step_id': i,
                'content': step,
                'variables': [],
                'constraints': [],
                'logical_form': None,
                'type': 'unknown'
            }
            
            # Extract variables and constraints based on verification level
            if verification_level in [VerificationLevel.INTERMEDIATE, VerificationLevel.ADVANCED, VerificationLevel.STRICT]:
                formula['variables'] = self._extract_variables(step)
                formula['constraints'] = self._extract_constraints(step)
                formula['logical_form'] = self._convert_to_logical_form(step)
                formula['type'] = self._classify_reasoning_type(step)
            
            formulas.append(formula)
        
        return formulas
    
    def _extract_variables(self, step: str) -> List[str]:
        """Extract variables from reasoning step."""
        variables = []
        
        # Look for common variable patterns
        var_patterns = [
            r'\b([A-Z][a-z]+)\b',  # Capitalized words
            r'\b([a-z]+_[a-z]+)\b',  # Snake_case
            r'\b([a-z]+[A-Z][a-z]+)\b',  # camelCase
            r'\b(x|y|z|a|b|c)\b'  # Single letter variables
        ]
        
        for pattern in var_patterns:
            matches = re.findall(pattern, step)
            variables.extend(matches)
        
        return list(set(variables))  # Remove duplicates
    
    def _extract_constraints(self, step: str) -> List[Dict[str, Any]]:
        """Extract constraints from reasoning step."""
        constraints = []
        
        # Look for equality constraints
        eq_matches = re.findall(r'(\w+)\s*=\s*(\w+)', step)
        for var1, var2 in eq_matches:
            constraints.append({
                'type': 'equality',
                'left': var1,
                'right': var2,
                'expression': f"{var1} == {var2}"
            })
        
        # Look for inequality constraints
        ineq_matches = re.findall(r'(\w+)\s*([><=!]+)\s*(\w+)', step)
        for var1, op, var2 in ineq_matches:
            constraints.append({
                'type': 'inequality',
                'left': var1,
                'operator': op,
                'right': var2,
                'expression': f"{var1} {op} {var2}"
            })
        
        # Look for causal constraints
        causal_matches = re.findall(r'(\w+)\s+(causes?|leads to|results in)\s+(\w+)', step, re.IGNORECASE)
        for cause, relation, effect in causal_matches:
            constraints.append({
                'type': 'causal',
                'cause': cause,
                'effect': effect,
                'relation': relation,
                'expression': f"causes({cause}, {effect})"
            })
        
        return constraints
    
    def _convert_to_logical_form(self, step: str) -> Optional[str]:
        """Convert reasoning step to logical form."""
        # Simple conversion for basic logical patterns
        step_lower = step.lower()
        
        # If-then statements
        if_then = re.search(r'if\s+(.+?)\s+then\s+(.+)', step_lower)
        if if_then:
            condition = if_then.group(1).strip()
            conclusion = if_then.group(2).strip()
            return f"({condition}) -> ({conclusion})"
        
        # And statements
        if ' and ' in step_lower:
            parts = step_lower.split(' and ')
            if len(parts) == 2:
                return f"({parts[0].strip()}) & ({parts[1].strip()})"
        
        # Or statements
        if ' or ' in step_lower:
            parts = step_lower.split(' or ')
            if len(parts) == 2:
                return f"({parts[0].strip()}) | ({parts[1].strip()})"
        
        # Not statements
        if step_lower.startswith('not '):
            return f"!({step_lower[4:]})"
        
        return None
    
    def _classify_reasoning_type(self, step: str) -> str:
        """Classify the type of reasoning in the step."""
        step_lower = step.lower()
        
        if any(word in step_lower for word in ['if', 'then', 'implies', 'therefore']):
            return 'logical'
        elif any(word in step_lower for word in ['causes', 'leads to', 'results in', 'because']):
            return 'causal'
        elif any(word in step_lower for word in ['equals', '=', 'greater', 'less', 'more', 'fewer']):
            return 'quantitative'
        elif any(word in step_lower for word in ['before', 'after', 'during', 'when', 'while']):
            return 'temporal'
        else:
            return 'general'
    
    def _check_logical_consistency(
        self,
        formulas: List[Dict[str, Any]],
        verification_level: VerificationLevel
    ) -> Dict[str, Any]:
        """Check logical consistency of formulas."""
        if verification_level == VerificationLevel.BASIC:
            return {'is_consistent': True, 'score': 1.0, 'issues': []}
        
        # Create Z3 solver
        solver = z3.Solver()
        
        # Add formulas to solver
        for formula in formulas:
            if formula['logical_form']:
                try:
                    # Convert to Z3 expression (simplified)
                    z3_expr = self._convert_to_z3(formula['logical_form'])
                    if z3_expr:
                        solver.add(z3_expr)
                except Exception as e:
                    continue
        
        # Check satisfiability
        result = solver.check()
        
        if result == z3.sat:
            return {'is_consistent': True, 'score': 1.0, 'issues': []}
        elif result == z3.unsat:
            return {
                'is_consistent': False,
                'score': 0.0,
                'issues': ['Logical inconsistency detected: formulas are unsatisfiable']
            }
        else:
            return {
                'is_consistent': True,
                'score': 0.5,
                'issues': ['Unable to determine consistency']
            }
    
    def _check_contradictions(
        self,
        formulas: List[Dict[str, Any]],
        verification_level: VerificationLevel
    ) -> Dict[str, Any]:
        """Check for contradictions between formulas."""
        if verification_level == VerificationLevel.BASIC:
            return {'is_consistent': True, 'score': 1.0, 'issues': []}
        
        contradictions = []
        
        # Check for direct contradictions
        for i, formula1 in enumerate(formulas):
            for j, formula2 in enumerate(formulas[i+1:], i+1):
                if self._are_contradictory(formula1, formula2):
                    contradictions.append(
                        f"Contradiction between step {i+1} and step {j+1}"
                    )
        
        is_consistent = len(contradictions) == 0
        score = 1.0 - (len(contradictions) * 0.2)
        
        return {
            'is_consistent': is_consistent,
            'score': max(0.0, score),
            'issues': contradictions
        }
    
    def _check_circular_reasoning(
        self,
        reasoning_steps: List[str]
    ) -> Dict[str, Any]:
        """Check for circular reasoning patterns."""
        circularities = []
        
        # Check for direct circular references
        for i, step in enumerate(reasoning_steps):
            # Look for references to previous steps
            for j in range(i):
                prev_step = reasoning_steps[j]
                
                # Check for circular reference patterns
                if self._is_circular_reference(step, prev_step, i, j):
                    circularities.append(
                        f"Circular reasoning detected: step {i+1} references step {j+1}"
                    )
        
        # Check for logical circularity
        if len(reasoning_steps) >= 3:
            for i in range(len(reasoning_steps) - 2):
                if self._forms_circular_chain(reasoning_steps[i:i+3]):
                    circularities.append(
                        f"Circular chain detected in steps {i+1}-{i+3}"
                    )
        
        is_consistent = len(circularities) == 0
        score = 1.0 - (len(circularities) * 0.3)
        
        return {
            'is_consistent': is_consistent,
            'score': max(0.0, score),
            'issues': circularities
        }
    
    def _are_contradictory(
        self,
        formula1: Dict[str, Any],
        formula2: Dict[str, Any]
    ) -> bool:
        """Check if two formulas are contradictory."""
        # Simple contradiction detection based on constraints
        constraints1 = formula1.get('constraints', [])
        constraints2 = formula2.get('constraints', [])
        
        for c1 in constraints1:
            for c2 in constraints2:
                if self._constraints_contradict(c1, c2):
                    return True
        
        return False
    
    def _constraints_contradict(
        self,
        constraint1: Dict[str, Any],
        constraint2: Dict[str, Any]
    ) -> bool:
        """Check if two constraints contradict each other."""
        # Check for equality vs inequality contradictions
        if (constraint1['type'] == 'equality' and constraint2['type'] == 'inequality' and
            constraint1['left'] == constraint2['left'] and constraint1['right'] == constraint2['right']):
            return True
        
        # Check for opposite inequalities
        if (constraint1['type'] == 'inequality' and constraint2['type'] == 'inequality' and
            constraint1['left'] == constraint2['left'] and constraint1['right'] == constraint2['right']):
            op1, op2 = constraint1['operator'], constraint2['operator']
            if ((op1 == '>' and op2 == '<') or (op1 == '<' and op2 == '>') or
                (op1 == '>=' and op2 == '<') or (op1 == '<=' and op2 == '>')):
                return True
        
        return False
    
    def _is_circular_reference(
        self,
        step: str,
        prev_step: str,
        step_idx: int,
        prev_idx: int
    ) -> bool:
        """Check if step makes a circular reference to prev_step."""
        # Look for reference patterns
        reference_patterns = [
            r'as mentioned (?:above|earlier|before)',
            r'as (?:we|I) (?:said|stated|mentioned)',
            r'this (?:means|implies|suggests)',
            r'as (?:discussed|explained) (?:above|earlier)'
        ]
        
        step_lower = step.lower()
        for pattern in reference_patterns:
            if re.search(pattern, step_lower):
                # Check if it references the previous step
                prev_words = set(prev_step.lower().split())
                step_words = set(step_lower.split())
                overlap = len(prev_words.intersection(step_words))
                
                if overlap > 3:  # Threshold for circular reference
                    return True
        
        return False
    
    def _forms_circular_chain(self, steps: List[str]) -> bool:
        """Check if three consecutive steps form a circular chain."""
        if len(steps) < 3:
            return False
        
        # Check if step 1 references step 3, step 2 references step 1, step 3 references step 2
        step1_words = set(steps[0].lower().split())
        step2_words = set(steps[1].lower().split())
        step3_words = set(steps[2].lower().split())
        
        # Check for circular references
        ref_1_to_3 = len(step1_words.intersection(step3_words)) > 3
        ref_2_to_1 = len(step2_words.intersection(step1_words)) > 3
        ref_3_to_2 = len(step3_words.intersection(step2_words)) > 3
        
        return ref_1_to_3 and ref_2_to_1 and ref_3_to_2
    
    def _convert_to_z3(self, logical_form: str) -> Optional[z3.ExprRef]:
        """Convert logical form to Z3 expression."""
        try:
            # This is a simplified conversion - in practice, you'd need a more robust parser
            if '->' in logical_form:
                parts = logical_form.split('->')
                if len(parts) == 2:
                    left = z3.Bool(parts[0].strip())
                    right = z3.Bool(parts[1].strip())
                    return z3.Implies(left, right)
            elif '&' in logical_form:
                parts = logical_form.split('&')
                if len(parts) == 2:
                    left = z3.Bool(parts[0].strip())
                    right = z3.Bool(parts[1].strip())
                    return z3.And(left, right)
            elif '|' in logical_form:
                parts = logical_form.split('|')
                if len(parts) == 2:
                    left = z3.Bool(parts[0].strip())
                    right = z3.Bool(parts[1].strip())
                    return z3.Or(left, right)
            elif logical_form.startswith('!'):
                expr = z3.Bool(logical_form[1:].strip())
                return z3.Not(expr)
            else:
                return z3.Bool(logical_form.strip())
        except Exception:
            return None


class ReasonVerifier:
    """
    Main reason verifier module that coordinates verification strategies.
    
    This module provides a unified interface for verifying reasoning chains
    using theorem provers and logical consistency checks.
    """
    
    def __init__(
        self,
        verifier: Optional[BaseVerifier] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the reason verifier.
        
        Args:
            verifier: Verifier implementation (defaults to Z3Verifier)
            config: Configuration dictionary
        """
        self.verifier = verifier or Z3Verifier()
        self.config = config or {}
        
        # Load domain-specific configurations
        self.domain_configs = self._load_domain_configs()
    
    def _load_domain_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load domain-specific verification configurations."""
        return {
            'causal': {
                'verification_level': VerificationLevel.INTERMEDIATE,
                'check_circularity': True,
                'check_contradictions': True
            },
            'logical': {
                'verification_level': VerificationLevel.ADVANCED,
                'check_circularity': True,
                'check_contradictions': True
            },
            'arithmetic': {
                'verification_level': VerificationLevel.STRICT,
                'check_circularity': False,
                'check_contradictions': True
            },
            'general': {
                'verification_level': VerificationLevel.BASIC,
                'check_circularity': True,
                'check_contradictions': False
            }
        }
    
    def verify_reasons(
        self,
        reasoning_steps: List[str],
        query: str,
        domain: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        """
        Verify reasoning steps and return consistent ones.
        
        Args:
            reasoning_steps: List of reasoning steps to verify
            query: Original query for context
            domain: Reasoning domain for domain-specific verification
            **kwargs: Additional verification parameters
            
        Returns:
            List of verified reasoning steps
        """
        if not reasoning_steps:
            return []
        
        domain = domain or 'general'
        
        # Get domain-specific configuration
        domain_config = self.domain_configs.get(domain, self.domain_configs['general'])
        
        # Merge with provided kwargs
        verify_params = {**domain_config, **kwargs}
        
        # Run verification
        result = self.verifier.verify_reasons(
            reasoning_steps=reasoning_steps,
            query=query,
            **verify_params
        )
        
        return result.verified_steps
    
    def get_verification_stats(
        self,
        original_steps: List[str],
        verified_steps: List[str]
    ) -> Dict[str, Any]:
        """Get statistics about the verification process."""
        return {
            'original_count': len(original_steps),
            'verified_count': len(verified_steps),
            'verification_rate': len(verified_steps) / len(original_steps) if original_steps else 0,
            'verifier_type': type(self.verifier).__name__,
            'available_levels': [level.value for level in VerificationLevel]
        }
