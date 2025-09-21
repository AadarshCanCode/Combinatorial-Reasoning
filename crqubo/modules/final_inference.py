"""
Final Inference Module

This module generates the final answer using the selected reasoning path.
It combines the ordered reasoning chain with the original query to produce
a coherent, well-structured response.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import openai


class InferenceStrategy(Enum):
    """Enumeration of inference strategies."""

    DIRECT_SYNTHESIS = "direct_synthesis"
    STEP_BY_STEP = "step_by_step"
    SUMMARY_FIRST = "summary_first"
    EVIDENCE_BASED = "evidence_based"
    CONVERSATIONAL = "conversational"


@dataclass
class InferenceResult:
    """Container for inference results."""

    answer: str
    confidence: float
    reasoning_summary: str
    evidence_used: List[str]
    inference_time: float
    strategy_used: str
    metadata: Dict[str, Any]


class BaseInferenceEngine(ABC):
    """Abstract base class for inference implementations."""

    @abstractmethod
    def generate_answer(
        self,
        query: str,
        reasoning_chain: List[str],
        retrieved_knowledge: Optional[List[Any]] = None,
        **kwargs,
    ) -> InferenceResult:
        """Generate final answer from reasoning chain."""
        pass


class OpenAIInferenceEngine(BaseInferenceEngine):
    """OpenAI-based inference engine implementation."""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize OpenAI inference engine.

        Args:
            model: OpenAI model to use
            api_key: OpenAI API key (if not set in environment)
            config: Additional configuration
        """
        self.model = model
        self.config = config or {}

        if api_key:
            openai.api_key = api_key

        # Load inference templates
        self.templates = self._load_inference_templates()

    def _load_inference_templates(self) -> Dict[str, Dict[str, str]]:
        """Load inference prompt templates for different strategies."""
        return {
            "direct_synthesis": {
                "system": """You are an expert reasoning assistant. Your task is to synthesize a clear, accurate answer based on the provided reasoning chain.

Guidelines:
- Use the reasoning steps to build a comprehensive answer
- Be direct and concise
- Maintain logical flow
- Cite specific reasoning steps when relevant
- Ensure the answer directly addresses the original question""",
                "user": """Question: {query}

Reasoning Chain:
{reasoning_chain}

{knowledge_context}

Based on this reasoning chain, provide a clear and accurate answer to the question.""",
            },
            "step_by_step": {
                "system": """You are an expert reasoning assistant. Your task is to walk through the reasoning process step by step to arrive at the final answer.

Guidelines:
- Follow the reasoning chain sequentially
- Explain each step clearly
- Show how each step builds toward the answer
- Highlight key insights and connections
- Conclude with a clear final answer""",
                "user": """Question: {query}

Reasoning Chain:
{reasoning_chain}

{knowledge_context}

Walk through this reasoning process step by step to arrive at the final answer.""",
            },
            "summary_first": {
                "system": """You are an expert reasoning assistant. Your task is to first provide a summary of the reasoning process, then give the final answer.

Guidelines:
- Start with a brief summary of the reasoning approach
- Highlight the key steps and insights
- Then provide the detailed final answer
- Use clear structure and formatting
- Ensure logical coherence throughout""",
                "user": """Question: {query}

Reasoning Chain:
{reasoning_chain}

{knowledge_context}

First, summarize the reasoning approach, then provide the final answer.""",
            },
            "evidence_based": {
                "system": """You are an expert reasoning assistant. Your task is to provide an evidence-based answer that clearly shows how the reasoning chain supports the conclusion.

Guidelines:
- Present evidence from the reasoning chain
- Show how each piece of evidence supports the conclusion
- Address potential counterarguments if relevant
- Use clear, logical structure
- Conclude with a well-supported answer""",
                "user": """Question: {query}

Reasoning Chain:
{reasoning_chain}

{knowledge_context}

Provide an evidence-based answer that clearly shows how the reasoning chain supports your conclusion.""",
            },
            "conversational": {
                "system": """You are a helpful and knowledgeable assistant. Your task is to provide a conversational, easy-to-understand answer based on the reasoning process.

Guidelines:
- Use a conversational, friendly tone
- Break down complex concepts into simple terms
- Use analogies or examples when helpful
- Ask clarifying questions if needed
- Make the answer accessible to a general audience""",
                "user": """Question: {query}

Here's how I would think through this:

{reasoning_chain}

{knowledge_context}

Based on this reasoning process, here's my answer:""",
            },
        }

    def generate_answer(
        self,
        query: str,
        reasoning_chain: List[str],
        retrieved_knowledge: Optional[List[Any]] = None,
        strategy: InferenceStrategy = InferenceStrategy.DIRECT_SYNTHESIS,
        **kwargs,
    ) -> InferenceResult:
        """Generate final answer using OpenAI."""
        import time

        start_time = time.time()

        # Prepare reasoning chain text
        reasoning_text = self._format_reasoning_chain(reasoning_chain)

        # Prepare knowledge context
        knowledge_context = self._format_knowledge_context(retrieved_knowledge)

        # Get template for strategy
        template = self.templates.get(
            strategy.value, self.templates["direct_synthesis"]
        )

        # Format prompt
        system_prompt = template["system"]
        user_prompt = template["user"].format(
            query=query,
            reasoning_chain=reasoning_text,
            knowledge_context=knowledge_context,
        )

        # Generate response
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=kwargs.get("max_tokens", 1500),
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.9),
            )

            answer = response.choices[0].message.content.strip()

            # Extract additional information
            reasoning_summary = self._extract_reasoning_summary(answer)
            evidence_used = self._extract_evidence(answer, reasoning_chain)
            confidence = self._estimate_confidence(answer, reasoning_chain)

            inference_time = time.time() - start_time

            return InferenceResult(
                answer=answer,
                confidence=confidence,
                reasoning_summary=reasoning_summary,
                evidence_used=evidence_used,
                inference_time=inference_time,
                strategy_used=strategy.value,
                metadata={
                    "model_used": self.model,
                    "tokens_used": (
                        response.usage.total_tokens
                        if hasattr(response, "usage")
                        else None
                    ),
                },
            )

        except Exception as e:
            print(f"Error generating answer: {e}")
            return InferenceResult(
                answer="I apologize, but I encountered an error while generating the answer.",
                confidence=0.0,
                reasoning_summary="Error occurred during inference",
                evidence_used=[],
                inference_time=time.time() - start_time,
                strategy_used=strategy.value,
                metadata={"error": str(e)},
            )

    def _format_reasoning_chain(self, reasoning_chain: List[str]) -> str:
        """Format reasoning chain for prompt."""
        if not reasoning_chain:
            return "No reasoning steps provided."

        formatted_steps = []
        for i, step in enumerate(reasoning_chain, 1):
            formatted_steps.append(f"{i}. {step}")

        return "\n".join(formatted_steps)

    def _format_knowledge_context(
        self, retrieved_knowledge: Optional[List[Any]]
    ) -> str:
        """Format retrieved knowledge for prompt."""
        if not retrieved_knowledge:
            return ""

        knowledge_items = []
        for i, doc in enumerate(retrieved_knowledge[:3], 1):  # Limit to top 3
            if hasattr(doc, "content"):
                knowledge_items.append(f"Knowledge {i}: {doc.content[:200]}...")
            else:
                knowledge_items.append(f"Knowledge {i}: {str(doc)[:200]}...")

        return "\n\nAdditional Knowledge:\n" + "\n".join(knowledge_items)

    def _extract_reasoning_summary(self, answer: str) -> str:
        """Extract reasoning summary from answer."""
        # Look for summary patterns
        summary_patterns = [
            r"in summary[^.]*\.?",
            r"to summarize[^.]*\.?",
            r"in conclusion[^.]*\.?",
            r"the key points are[^.]*\.?",
            r"the main reasoning is[^.]*\.?",
        ]

        for pattern in summary_patterns:
            match = re.search(pattern, answer, re.IGNORECASE)
            if match:
                return match.group(0).strip()

        # Fallback: use first sentence
        sentences = answer.split(".")
        return sentences[0] + "." if sentences else answer[:100] + "..."

    def _extract_evidence(self, answer: str, reasoning_chain: List[str]) -> List[str]:
        """Extract evidence used from answer."""
        evidence = []

        # Look for references to reasoning steps
        for i, step in enumerate(reasoning_chain, 1):
            step_words = set(step.lower().split())
            answer_words = set(answer.lower().split())

            # Check for word overlap
            overlap = len(step_words.intersection(answer_words))
            if overlap > 2:  # Threshold for evidence usage
                evidence.append(f"Step {i}: {step[:100]}...")

        return evidence

    def _estimate_confidence(self, answer: str, reasoning_chain: List[str]) -> float:
        """Estimate confidence in the answer."""
        confidence = 0.5  # Base confidence

        # Length-based confidence
        if len(answer.split()) > 20:
            confidence += 0.1

        # Reasoning chain usage
        if reasoning_chain:
            evidence_count = len(self._extract_evidence(answer, reasoning_chain))
            confidence += min(0.3, evidence_count * 0.1)

        # Certainty language
        certainty_words = [
            "definitely",
            "certainly",
            "clearly",
            "obviously",
            "undoubtedly",
        ]
        uncertainty_words = [
            "maybe",
            "perhaps",
            "might",
            "could",
            "possibly",
            "unclear",
        ]

        answer_lower = answer.lower()
        certainty_count = sum(1 for word in certainty_words if word in answer_lower)
        uncertainty_count = sum(1 for word in uncertainty_words if word in answer_lower)

        confidence += certainty_count * 0.05
        confidence -= uncertainty_count * 0.05

        # Structure confidence
        if answer.endswith(".") and answer[0].isupper():
            confidence += 0.05

        return max(0.0, min(1.0, confidence))


class FinalInference:
    """
    Main final inference module that coordinates answer generation.

    This module provides a unified interface for generating final answers
    from reasoning chains using various inference strategies.
    """

    def __init__(
        self,
        inference_engine: Optional[BaseInferenceEngine] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the final inference module.

        Args:
            inference_engine: Inference engine implementation (defaults to OpenAIInferenceEngine)
            config: Configuration dictionary
        """
        self.inference_engine = inference_engine or OpenAIInferenceEngine()
        self.config = config or {}

        # Load domain-specific configurations
        self.domain_configs = self._load_domain_configs()

    def _load_domain_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load domain-specific inference configurations."""
        return {
            "causal": {
                "strategy": InferenceStrategy.EVIDENCE_BASED,
                "max_tokens": 1200,
                "temperature": 0.6,
            },
            "logical": {
                "strategy": InferenceStrategy.STEP_BY_STEP,
                "max_tokens": 1000,
                "temperature": 0.5,
            },
            "arithmetic": {
                "strategy": InferenceStrategy.DIRECT_SYNTHESIS,
                "max_tokens": 800,
                "temperature": 0.3,
            },
            "general": {
                "strategy": InferenceStrategy.DIRECT_SYNTHESIS,
                "max_tokens": 1000,
                "temperature": 0.7,
            },
        }

    def generate_answer(
        self,
        query: str,
        reasoning_chain: List[str],
        retrieved_knowledge: Optional[List[Any]] = None,
        domain: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate final answer from reasoning chain.

        Args:
            query: Original query
            reasoning_chain: Ordered list of reasoning steps
            retrieved_knowledge: Optional retrieved knowledge
            domain: Reasoning domain for domain-specific inference
            **kwargs: Additional inference parameters

        Returns:
            Dictionary containing answer and metadata
        """
        if not reasoning_chain:
            return {
                "answer": "I don't have enough reasoning steps to provide a complete answer.",
                "confidence": 0.0,
                "reasoning_summary": "No reasoning chain provided",
                "evidence_used": [],
                "metadata": {"error": "empty_reasoning_chain"},
            }

        domain = domain or "general"

        # Get domain-specific configuration
        domain_config = self.domain_configs.get(domain, self.domain_configs["general"])

        # Merge with provided kwargs
        inference_params = {**domain_config, **kwargs}

        # Generate answer
        result = self.inference_engine.generate_answer(
            query=query,
            reasoning_chain=reasoning_chain,
            retrieved_knowledge=retrieved_knowledge,
            **inference_params,
        )

        return {
            "answer": result.answer,
            "confidence": result.confidence,
            "reasoning_summary": result.reasoning_summary,
            "evidence_used": result.evidence_used,
            "metadata": {
                "strategy_used": result.strategy_used,
                "inference_time": result.inference_time,
                "domain": domain,
                **result.metadata,
            },
        }

    def generate_with_strategy(
        self,
        query: str,
        reasoning_chain: List[str],
        strategy: InferenceStrategy,
        retrieved_knowledge: Optional[List[Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate answer using a specific inference strategy."""
        result = self.inference_engine.generate_answer(
            query=query,
            reasoning_chain=reasoning_chain,
            retrieved_knowledge=retrieved_knowledge,
            strategy=strategy,
            **kwargs,
        )

        return {
            "answer": result.answer,
            "confidence": result.confidence,
            "reasoning_summary": result.reasoning_summary,
            "evidence_used": result.evidence_used,
            "metadata": {
                "strategy_used": result.strategy_used,
                "inference_time": result.inference_time,
                **result.metadata,
            },
        }

    def get_inference_stats(self) -> Dict[str, Any]:
        """Get statistics about the inference process."""
        return {
            "inference_engine": type(self.inference_engine).__name__,
            "available_strategies": [strategy.value for strategy in InferenceStrategy],
            "domain_configs": list(self.domain_configs.keys()),
            "config": self.config,
        }
