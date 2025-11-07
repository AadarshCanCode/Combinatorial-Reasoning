"""
Reason Sampling Module

This module generates candidate reasoning steps using zero-shot or few-shot
LLM prompting. It supports multiple sampling strategies and reasoning patterns.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from ..exceptions import APIKeyError, CRQUBOError, SamplingError, handle_external_exception
from ..logging_utils import log_duration
from ..retry_utils import ResourceTracker, retry_with_exponential_backoff

# Set up logging
logger = logging.getLogger(__name__)


class SamplingStrategy(Enum):
    """Enumeration of sampling strategies."""

    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHT = "tree_of_thought"
    SELF_CONSISTENCY = "self_consistency"


@dataclass
class ReasoningStep:
    """Container for a single reasoning step."""

    step_id: str
    content: str
    reasoning_type: str
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class SamplingResult:
    """Container for sampling results."""

    reasoning_steps: List[ReasoningStep]
    query: str
    strategy_used: str
    total_steps: int
    sampling_time: float


class BaseReasonSampler(ABC):
    """Abstract base class for reason sampling implementations."""

    @abstractmethod
    def sample_reasons(
        self, query: str, num_samples: int = 5, **kwargs
    ) -> List[ReasoningStep]:
        """Generate candidate reasoning steps."""
        pass


class OpenAISampler(BaseReasonSampler):
    """OpenAI-based reason sampler implementation using OpenAI v1.0+ client."""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize OpenAI sampler.

        Args:
            model: OpenAI model to use
            api_key: OpenAI API key (if not set in environment)
            config: Additional configuration
        """
        try:
            from openai import (
                APIConnectionError,
                APIError,
                APITimeoutError,
                OpenAI,
                RateLimitError,
            )
        except ImportError as e:
            raise SamplingError(
                "OpenAI package not installed",
                recovery_hint="Install with: pip install openai>=1.0.0",
            ) from e

        self.model = model
        self.config = config or {}
        self._api_error_cls = APIError
        self._retryable_errors = (
            RateLimitError,
            APITimeoutError,
            APIConnectionError,
        )

        # Initialize OpenAI client with new v1.0+ API
        try:
            self.client = OpenAI(api_key=api_key)
        except Exception as e:
            if "api_key" in str(e).lower():
                raise APIKeyError(
                    "Failed to initialize OpenAI client",
                    recovery_hint="Set OPENAI_API_KEY environment variable or pass api_key parameter",
                ) from e
            raise handle_external_exception(e, context="OpenAISampler initialization") from e

        # Initialize resource tracker
        max_calls_per_hour = self.config.get("max_api_calls_per_hour")
        max_tokens_per_hour = self.config.get("max_tokens_per_hour")
        if max_calls_per_hour or max_tokens_per_hour:
            self.resource_tracker = ResourceTracker(
                max_api_calls=max_calls_per_hour,
                max_tokens=max_tokens_per_hour,
                tracking_window=3600.0,
            )
        else:
            self.resource_tracker = None

        # Default sampling parameters and retry configuration
        self.temperature = self.config.get("temperature", 0.7)
        self.top_p = self.config.get("top_p", 0.9)
        self.max_tokens = self.config.get("max_tokens", 1000)
        self.request_timeout = self.config.get("request_timeout", 60.0)
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_initial_delay = self.config.get("retry_initial_delay", 1.0)
        self.retry_backoff_factor = self.config.get("retry_backoff_factor", 2.0)
        self.retry_max_delay = self.config.get("retry_max_delay", 30.0)
        self.system_prompt = self.config.get(
            "system_prompt",
            "You are an expert reasoning assistant. Provide clear, logical reasoning steps.",
        )

        # Load reasoning templates
        self.templates = self._load_reasoning_templates()

        logger.info("OpenAISampler initialized", extra={"model": model})

    def _load_reasoning_templates(self) -> Dict[str, Dict[str, str]]:
        """Load reasoning prompt templates for different domains."""
        return {
            "causal": {
                "zero_shot": """Analyze the following causal relationship question step by step:

Question: {query}

Provide a step-by-step causal analysis. Each step should be a clear, logical reasoning step that builds toward understanding the causal relationship.

Steps:""",
                "few_shot": """Here are examples of causal reasoning:

Example 1:
Question: Why does smoking cause lung cancer?
Steps:
1. Smoking introduces harmful chemicals into the lungs
2. These chemicals damage lung tissue over time
3. Damaged tissue is more susceptible to cancerous mutations
4. Therefore, smoking increases the risk of lung cancer

Example 2:
Question: How does education affect income?
Steps:
1. Education provides knowledge and skills
2. Knowledge and skills increase job market value
3. Higher job market value leads to better job opportunities
4. Better job opportunities result in higher income

Now analyze this question:
Question: {query}
Steps:""",
            },
            "logical": {
                "zero_shot": """Solve this logical reasoning problem step by step:

Problem: {query}

Break down the logical reasoning into clear, sequential steps. Each step should follow logically from the previous ones.

Steps:""",
                "few_shot": """Here are examples of logical reasoning:

Example 1:
Problem: If all birds can fly and penguins are birds, can penguins fly?
Steps:
1. Premise: All birds can fly
2. Premise: Penguins are birds
3. By universal instantiation: Penguins can fly
4. However, this contradicts known facts about penguins
5. Therefore, the first premise is false

Example 2:
Problem: If A implies B and B implies C, what can we conclude about A and C?
Steps:
1. Given: A → B
2. Given: B → C
3. By hypothetical syllogism: A → C
4. Therefore, A implies C

Now solve this problem:
Problem: {query}
Steps:""",
            },
            "arithmetic": {
                "zero_shot": """Solve this mathematical problem step by step:

Problem: {query}

Show each calculation step clearly and explain your reasoning.

Steps:""",
                "few_shot": """Here are examples of mathematical reasoning:

Example 1:
Problem: What is 15% of 200?
Steps:
1. Convert percentage to decimal: 15% = 0.15
2. Multiply by the number: 0.15 × 200 = 30
3. Therefore, 15% of 200 is 30

Example 2:
Problem: Solve for x: 2x + 5 = 13
Steps:
1. Subtract 5 from both sides: 2x = 8
2. Divide both sides by 2: x = 4
3. Verify: 2(4) + 5 = 8 + 5 = 13 ✓

Now solve this problem:
Problem: {query}
Steps:""",
            },
            "general": {
                "zero_shot": """Think through this problem step by step:

Problem: {query}

Provide a clear, logical sequence of reasoning steps that lead to a solution.

Steps:""",
                "few_shot": """Here are examples of step-by-step reasoning:

Example 1:
Problem: What are the main causes of climate change?
Steps:
1. Identify greenhouse gases as primary contributors
2. Consider human activities that increase these gases
3. Analyze natural processes that affect climate
4. Synthesize the interactions between these factors

Example 2:
Problem: How can we improve team productivity?
Steps:
1. Identify current productivity bottlenecks
2. Analyze team dynamics and communication patterns
3. Consider available tools and resources
4. Develop specific improvement strategies

Now think through this problem:
Problem: {query}
Steps:""",
            },
        }

    def sample_reasons(
        self,
        query: str,
        num_samples: int = 5,
        domain: Optional[str] = None,
        strategy: SamplingStrategy = SamplingStrategy.ZERO_SHOT,
        retrieved_knowledge: Optional[List[Any]] = None,
        **kwargs,
    ) -> List[ReasoningStep]:
        """Generate candidate reasoning steps using OpenAI."""

        # Select template based on domain
        domain = domain or "general"
        template_key = strategy.value

        if domain not in self.templates or template_key not in self.templates[domain]:
            domain = "general"
            template_key = "zero_shot"

        template = self.templates[domain][template_key]

        # Prepare prompt
        prompt = template.format(query=query)

        # Add retrieved knowledge if available
        if retrieved_knowledge:
            knowledge_text = "\n".join([doc.content for doc in retrieved_knowledge[:3]])
            prompt += f"\n\nRelevant knowledge:\n{knowledge_text}\n"

        # Generate multiple samples with error handling
        reasoning_steps = []
        failed_samples = 0
        
        with log_duration(logger, f"sampling {num_samples} reasoning steps", query=query):
            for i in range(num_samples):
                try:
                    # Check resource limits before making API call
                    if self.resource_tracker:
                        self.resource_tracker.ensure_call_allowed()
                    
                    # Call OpenAI API with new client and retry logic
                    content = self._call_openai_with_retry(prompt, i, **kwargs)
                    
                    # Parse reasoning steps from response
                    steps = self._parse_reasoning_steps(content, i)
                    reasoning_steps.extend(steps)
                    
                    logger.debug(f"Successfully generated sample {i+1}/{num_samples}")

                except Exception as e:
                    failed_samples += 1
                    logger.warning(
                        f"Failed to generate sample {i+1}/{num_samples}: {e}",
                        exc_info=True
                    )
                    # Continue with other samples instead of failing completely
                    if failed_samples >= num_samples // 2:
                        # If more than half failed, raise error
                        raise SamplingError(
                            f"Failed to generate sufficient samples ({failed_samples}/{num_samples} failed)",
                            recovery_hint="Check API key, quota, and connectivity",
                            query=query,
                            failed_count=failed_samples,
                        ) from e

        if not reasoning_steps:
            raise SamplingError(
                "No reasoning steps generated",
                recovery_hint="All sampling attempts failed - check logs for details",
                query=query,
                num_samples=num_samples,
            )
        
        logger.info(
            f"Generated {len(reasoning_steps)} reasoning steps from {num_samples - failed_samples}/{num_samples} successful samples"
        )
        return reasoning_steps
    
    def _call_openai_with_retry(self, prompt: str, sample_id: int, **kwargs) -> str:
        """
        Call OpenAI API with retry logic.
        
        Args:
            prompt: The prompt to send
            sample_id: Sample identifier
            **kwargs: Additional parameters
            
        Returns:
            Response content text
            
        Raises:
            SamplingError: If API call fails after retries
        """
        @retry_with_exponential_backoff(
            max_retries=self.max_retries,
            initial_delay=self.retry_initial_delay,
            max_delay=self.retry_max_delay,
            backoff_factor=self.retry_backoff_factor,
            retryable_exceptions=self._retryable_errors,
            logger_instance=logger,
        )
        def _make_request():
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                top_p=kwargs.get("top_p", self.top_p),
                timeout=self.request_timeout,
            )
            return response
        
        try:
            response = _make_request()
            
            # Track API usage
            if self.resource_tracker:
                tokens_used = getattr(response.usage, "total_tokens", 0) if hasattr(response, "usage") else 0
                self.resource_tracker.track_api_call(tokens_used)
            
            # Extract content
            if not response.choices:
                raise SamplingError(
                    "No choices in API response",
                    recovery_hint="API returned empty response",
                    sample_id=sample_id,
                )
            
            content = response.choices[0].message.content
            if not content:
                raise SamplingError(
                    "Empty content in API response",
                    recovery_hint="API returned response with no content",
                    sample_id=sample_id,
                )
            
            return content.strip()
            
        except self._retryable_errors as e:
            # These should have been retried already
            raise SamplingError(
                f"API call failed after {self.max_retries} retries",
                recovery_hint="Check network connectivity and API status",
                sample_id=sample_id,
            ) from e
        except Exception as e:
            # Convert to CRQUBO exception
            crqubo_error = handle_external_exception(e, context=f"sample_{sample_id}")
            raise SamplingError(
                f"API call failed: {crqubo_error.message}",
                recovery_hint=crqubo_error.recovery_hint,
                sample_id=sample_id,
            ) from e

    def _parse_reasoning_steps(
        self, content: str, sample_id: int
    ) -> List[ReasoningStep]:
        """Parse reasoning steps from LLM response."""
        steps = []
        lines = content.split("\n")

        step_counter = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Look for numbered steps or bullet points
            if (
                line.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."))
                or line.startswith(("•", "-", "*"))
                or line.startswith("Step")
            ):

                # Clean up the step text
                step_text = line
                if line.startswith(
                    ("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")
                ):
                    step_text = line[2:].strip()
                elif line.startswith(("•", "-", "*")):
                    step_text = line[1:].strip()
                elif line.startswith("Step"):
                    step_text = line.split(":", 1)[1].strip() if ":" in line else line

                if step_text:
                    step_counter += 1
                    step_id = f"sample_{sample_id}_step_{step_counter}"

                    # Determine reasoning type based on content
                    reasoning_type = self._classify_reasoning_type(step_text)

                    # Estimate confidence (simple heuristic)
                    confidence = self._estimate_confidence(step_text)

                    steps.append(
                        ReasoningStep(
                            step_id=step_id,
                            content=step_text,
                            reasoning_type=reasoning_type,
                            confidence=confidence,
                            metadata={
                                "sample_id": sample_id,
                                "step_number": step_counter,
                                "original_line": line,
                            },
                        )
                    )

        return steps

    def _classify_reasoning_type(self, step_text: str) -> str:
        """Classify the type of reasoning in a step."""
        step_lower = step_text.lower()

        if any(
            word in step_lower
            for word in ["because", "since", "due to", "causes", "leads to"]
        ):
            return "causal"
        elif any(
            word in step_lower
            for word in ["therefore", "thus", "hence", "follows", "implies"]
        ):
            return "logical"
        elif any(
            word in step_lower for word in ["calculate", "compute", "solve", "equation"]
        ):
            return "computational"
        elif any(word in step_lower for word in ["assume", "suppose", "hypothesis"]):
            return "assumption"
        elif any(word in step_lower for word in ["consider", "analyze", "examine"]):
            return "analysis"
        else:
            return "general"

    def _estimate_confidence(self, step_text: str) -> float:
        """Estimate confidence in a reasoning step."""
        # Simple heuristic based on step characteristics
        confidence = 0.5  # Base confidence

        # Increase confidence for longer, more detailed steps
        if len(step_text.split()) > 10:
            confidence += 0.2

        # Increase confidence for steps with specific language
        if any(
            word in step_text.lower()
            for word in ["clearly", "obviously", "definitely", "certainly"]
        ):
            confidence += 0.1

        # Decrease confidence for uncertain language
        if any(
            word in step_text.lower()
            for word in ["maybe", "perhaps", "might", "could", "possibly"]
        ):
            confidence -= 0.1

        return max(0.0, min(1.0, confidence))


class HuggingFaceSampler(BaseReasonSampler):
    """Hugging Face Transformers-based reason sampler implementation."""

    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        api_token: Optional[str] = None,
        device: str = "auto",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Hugging Face sampler.

        Args:
            model_name: Hugging Face model name or path
            api_token: Hugging Face API token (for gated models)
            device: Device to run model on ('auto', 'cpu', 'cuda')
            config: Additional configuration
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            self.transformers = __import__("transformers")
            self.torch = torch
        except ImportError:
            raise ImportError(
                "Transformers package not installed. Install with: pip install transformers torch"
            )

        self.model_name = model_name
        self.config = config or {}
        self.device = device

        # Set up token
        if api_token:
            self.transformers.utils.HF_TOKEN = api_token

        # Initialize model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=api_token)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=api_token,
                torch_dtype=(
                    self.torch.float16 if device == "cuda" else self.torch.float32
                ),
            )

            if device == "auto":
                self.device = "cuda" if self.torch.cuda.is_available() else "cpu"

            self.model.to(self.device)

            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
            )

            logger.info(f"Loaded Hugging Face model: {model_name} on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load Hugging Face model {model_name}: {e}")
            raise

        # Load reasoning templates
        self.templates = self._load_reasoning_templates()

    def _load_reasoning_templates(self) -> Dict[str, Dict[str, str]]:
        """Load reasoning prompt templates for different domains."""
        return {
            "causal": {
                "zero_shot": """Analyze the following causal relationship question step by step:

Question: {query}

Provide a step-by-step causal analysis. Each step should be a clear, logical reasoning step that builds toward understanding the causal relationship.

Steps:""",
                "few_shot": """Here are examples of causal reasoning:

Example 1:
Question: Why does smoking cause lung cancer?
Steps:
1. Smoking introduces harmful chemicals into the lungs
2. These chemicals damage lung tissue over time
3. Damaged tissue is more susceptible to cancerous mutations
4. Therefore, smoking increases the risk of lung cancer

Example 2:
Question: How does education affect income?
Steps:
1. Education provides knowledge and skills
2. Knowledge and skills increase job market value
3. Higher job market value leads to better job opportunities
4. Better job opportunities result in higher income

Now analyze this question:
Question: {query}
Steps:""",
            },
            "logical": {
                "zero_shot": """Solve this logical reasoning problem step by step:

Problem: {query}

Break down the logical reasoning into clear, sequential steps. Each step should follow logically from the previous ones.

Steps:""",
                "few_shot": """Here are examples of logical reasoning:

Example 1:
Problem: If all birds can fly and penguins are birds, can penguins fly?
Steps:
1. Premise: All birds can fly
2. Premise: Penguins are birds
3. By universal instantiation: Penguins can fly
4. However, this contradicts known facts about penguins
5. Therefore, the first premise is false

Example 2:
Problem: If A implies B and B implies C, what can we conclude about A and C?
Steps:
1. Given: A → B
2. Given: B → C
3. By hypothetical syllogism: A → C
4. Therefore, A implies C

Now solve this problem:
Problem: {query}
Steps:""",
            },
            "general": {
                "zero_shot": """Think through this problem step by step:

Problem: {query}

Provide a clear, logical sequence of reasoning steps that lead to a solution.

Steps:""",
                "few_shot": """Here are examples of step-by-step reasoning:

Example 1:
Problem: What are the main causes of climate change?
Steps:
1. Identify greenhouse gases as primary contributors
2. Consider human activities that increase these gases
3. Analyze natural processes that affect climate
4. Synthesize the interactions between these factors

Example 2:
Problem: How can we improve team productivity?
Steps:
1. Identify current productivity bottlenecks
2. Analyze team dynamics and communication patterns
3. Consider available tools and resources
4. Develop specific improvement strategies

Now think through this problem:
Problem: {query}
Steps:""",
            },
        }

    def sample_reasons(
        self,
        query: str,
        num_samples: int = 5,
        domain: Optional[str] = None,
        strategy: SamplingStrategy = SamplingStrategy.ZERO_SHOT,
        retrieved_knowledge: Optional[List[Any]] = None,
        **kwargs,
    ) -> List[ReasoningStep]:
        """Generate candidate reasoning steps using Hugging Face models."""
        import time

        start_time = time.time()

        # Select template based on domain
        domain = domain or "general"
        template_key = strategy.value

        if domain not in self.templates or template_key not in self.templates[domain]:
            domain = "general"
            template_key = "zero_shot"

        template = self.templates[domain][template_key]

        # Prepare prompt
        prompt = template.format(query=query)

        # Add retrieved knowledge if available
        if retrieved_knowledge:
            knowledge_text = "\n".join([doc.content for doc in retrieved_knowledge[:3]])
            prompt += f"\n\nRelevant knowledge:\n{knowledge_text}\n"

        # Generate multiple samples
        reasoning_steps = []
        for i in range(num_samples):
            try:
                # Generate text using the pipeline
                result = self.pipeline(
                    prompt,
                    max_length=len(prompt.split())
                    + 200,  # Add some length for generation
                    num_return_sequences=1,
                    temperature=kwargs.get("temperature", 0.7),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

                content = result[0]["generated_text"]

                # Extract the generated part (remove the original prompt)
                generated_text = content[len(prompt) :].strip()

                # Parse reasoning steps from response
                steps = self._parse_reasoning_steps(generated_text, i)
                reasoning_steps.extend(steps)

            except Exception as e:
                logger.warning(f"Error generating reasoning step {i}: {e}")
                continue

        sampling_time = time.time() - start_time
        logger.debug(
            f"Generated {len(reasoning_steps)} reasoning steps in {sampling_time:.2f}s"
        )

        return reasoning_steps

    def _parse_reasoning_steps(
        self, content: str, sample_id: int
    ) -> List[ReasoningStep]:
        """Parse reasoning steps from model response."""
        steps = []
        lines = content.split("\n")

        step_counter = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Look for numbered steps or bullet points
            if (
                line.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."))
                or line.startswith(("•", "-", "*"))
                or line.startswith("Step")
            ):

                # Clean up the step text
                step_text = line
                if line.startswith(
                    ("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")
                ):
                    step_text = line[2:].strip()
                elif line.startswith(("•", "-", "*")):
                    step_text = line[1:].strip()
                elif line.startswith("Step"):
                    step_text = line.split(":", 1)[1].strip() if ":" in line else line

                if step_text:
                    step_counter += 1
                    step_id = f"hf_sample_{sample_id}_step_{step_counter}"

                    # Determine reasoning type based on content
                    reasoning_type = self._classify_reasoning_type(step_text)

                    # Estimate confidence (simple heuristic)
                    confidence = self._estimate_confidence(step_text)

                    steps.append(
                        ReasoningStep(
                            step_id=step_id,
                            content=step_text,
                            reasoning_type=reasoning_type,
                            confidence=confidence,
                            metadata={
                                "sample_id": sample_id,
                                "step_number": step_counter,
                                "original_line": line,
                                "model": self.model_name,
                            },
                        )
                    )

        return steps

    def _classify_reasoning_type(self, step_text: str) -> str:
        """Classify the type of reasoning in a step."""
        step_lower = step_text.lower()

        if any(
            word in step_lower
            for word in ["because", "since", "due to", "causes", "leads to"]
        ):
            return "causal"
        elif any(
            word in step_lower
            for word in ["therefore", "thus", "hence", "follows", "implies"]
        ):
            return "logical"
        elif any(
            word in step_lower for word in ["calculate", "compute", "solve", "equation"]
        ):
            return "computational"
        elif any(word in step_lower for word in ["assume", "suppose", "hypothesis"]):
            return "assumption"
        elif any(word in step_lower for word in ["consider", "analyze", "examine"]):
            return "analysis"
        else:
            return "general"

    def _estimate_confidence(self, step_text: str) -> float:
        """Estimate confidence in a reasoning step."""
        # Simple heuristic based on step characteristics
        confidence = 0.5  # Base confidence

        # Increase confidence for longer, more detailed steps
        if len(step_text.split()) > 10:
            confidence += 0.2

        # Increase confidence for steps with specific language
        if any(
            word in step_text.lower()
            for word in ["clearly", "obviously", "definitely", "certainly"]
        ):
            confidence += 0.1

        # Decrease confidence for uncertain language
        if any(
            word in step_text.lower()
            for word in ["maybe", "perhaps", "might", "could", "possibly"]
        ):
            confidence -= 0.1

        return max(0.0, min(1.0, confidence))


class ReasonSampler:
    """
    Main reason sampling module that coordinates reasoning step generation.

    This module provides a unified interface for generating candidate reasoning
    steps using various LLM-based sampling strategies.
    """

    def __init__(
        self,
        sampler: Optional[BaseReasonSampler] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the reason sampler.

        Args:
            sampler: Reason sampler implementation (defaults to OpenAISampler)
            config: Configuration dictionary
        """
        self.config = config or {}

        # Determine which sampler to use based on config
        if sampler is None:
            sampler_type = self.config.get("type", "openai")

            if sampler_type == "openai":
                sampler_config = self.config.get("openai", {})
                self.sampler = OpenAISampler(**sampler_config)
            elif sampler_type == "huggingface":
                sampler_config = self.config.get("huggingface", {})
                self.sampler = HuggingFaceSampler(**sampler_config)
            else:
                raise ValueError(f"Unknown sampler type: {sampler_type}")
        else:
            self.sampler = sampler

        # Load domain-specific configurations
        self.domain_configs = self._load_domain_configs()

    def _load_domain_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load domain-specific sampling configurations."""
        return {
            "causal": {
                "num_samples": 8,
                "strategy": SamplingStrategy.FEW_SHOT,
                "temperature": 0.8,
                "max_tokens": 1200,
            },
            "logical": {
                "num_samples": 6,
                "strategy": SamplingStrategy.ZERO_SHOT,
                "temperature": 0.5,
                "max_tokens": 1000,
            },
            "arithmetic": {
                "num_samples": 4,
                "strategy": SamplingStrategy.FEW_SHOT,
                "temperature": 0.3,
                "max_tokens": 800,
            },
            "general": {
                "num_samples": 5,
                "strategy": SamplingStrategy.ZERO_SHOT,
                "temperature": 0.7,
                "max_tokens": 1000,
            },
        }

    def sample_reasons(
        self,
        query: str,
        domain: Optional[str] = None,
        retrieved_knowledge: Optional[List[Any]] = None,
        **kwargs,
    ) -> List[ReasoningStep]:
        """
        Generate candidate reasoning steps for a query.

        Args:
            query: Input query
            domain: Reasoning domain
            retrieved_knowledge: Optional retrieved knowledge for context
            **kwargs: Additional parameters

        Returns:
            List of ReasoningStep objects
        """
        domain = domain or "general"

        # Get domain-specific configuration
        domain_config = self.domain_configs.get(domain, self.domain_configs["general"])

        # Merge with provided kwargs
        sampling_params = {**domain_config, **kwargs}

        # Generate reasoning steps
        reasoning_steps = self.sampler.sample_reasons(
            query=query,
            domain=domain,
            retrieved_knowledge=retrieved_knowledge,
            **sampling_params,
        )

        return reasoning_steps

    def sample_with_strategy(
        self,
        query: str,
        strategy: SamplingStrategy,
        domain: Optional[str] = None,
        **kwargs,
    ) -> List[ReasoningStep]:
        """Sample reasons using a specific strategy."""
        return self.sampler.sample_reasons(
            query=query, domain=domain, strategy=strategy, **kwargs
        )

    def get_sampling_stats(self) -> Dict[str, Any]:
        """Get statistics about the sampling process."""
        return {
            "sampler_type": type(self.sampler).__name__,
            "available_strategies": [strategy.value for strategy in SamplingStrategy],
            "domain_configs": list(self.domain_configs.keys()),
            "config": self.config,
        }
