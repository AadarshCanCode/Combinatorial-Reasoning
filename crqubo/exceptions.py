"""
Custom exception hierarchy for CRQUBO framework.

This module defines all custom exceptions used throughout CRQUBO,
providing clear error messages and recovery hints.
"""


class CRQUBOError(Exception):
    """Base exception for all CRQUBO errors."""

    def __init__(self, message: str, recovery_hint: str = None, **context):
        """
        Initialize CRQUBO error.

        Args:
            message: Human-readable error message
            recovery_hint: Optional suggestion for recovering from error
            **context: Additional context information
        """
        self.message = message
        self.recovery_hint = recovery_hint
        self.context = context
        super().__init__(self.format_message())

    def format_message(self) -> str:
        """Format error message with context and recovery hint."""
        msg = self.message
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            msg += f" [Context: {context_str}]"
        if self.recovery_hint:
            msg += f" [Hint: {self.recovery_hint}]"
        return msg


# Configuration Errors
class ConfigurationError(CRQUBOError):
    """Raised when there's an issue with configuration."""
    pass


class InvalidConfigError(ConfigurationError):
    """Raised when configuration is invalid."""
    pass


class MissingConfigError(ConfigurationError):
    """Raised when required configuration is missing."""
    pass


# API and Authentication Errors
class APIError(CRQUBOError):
    """Base class for API-related errors."""
    pass


class APIKeyError(APIError):
    """Raised when API key is missing or invalid."""
    pass


class APIRateLimitError(APIError):
    """Raised when API rate limit is exceeded."""
    pass


class APITimeoutError(APIError):
    """Raised when API request times out."""
    pass


class APIQuotaExceededError(APIError):
    """Raised when API quota is exceeded."""
    pass


# Module Errors
class ModuleError(CRQUBOError):
    """Base class for module-specific errors."""
    pass


class SamplingError(ModuleError):
    """Raised when reason sampling fails."""
    pass


class FilteringError(ModuleError):
    """Raised when semantic filtering fails."""
    pass


class OptimizationError(ModuleError):
    """Raised when combinatorial optimization fails."""
    pass


class OrderingError(ModuleError):
    """Raised when reason ordering fails."""
    pass


class VerificationError(ModuleError):
    """Raised when reason verification fails."""
    pass


class InferenceError(ModuleError):
    """Raised when final inference fails."""
    pass


class RetrievalError(ModuleError):
    """Raised when knowledge retrieval fails."""
    pass


# Data Validation Errors
class ValidationError(CRQUBOError):
    """Raised when data validation fails."""
    pass


class InvalidInputError(ValidationError):
    """Raised when input is invalid."""
    pass


class InvalidOutputError(ValidationError):
    """Raised when output is invalid."""
    pass


# Resource Errors
class ResourceError(CRQUBOError):
    """Base class for resource-related errors."""
    pass


class ResourceLimitError(ResourceError):
    """Raised when resource limit is exceeded."""
    pass


class MemoryError(ResourceError):
    """Raised when memory limit is exceeded."""
    pass


class TimeoutError(ResourceError):
    """Raised when operation times out."""
    pass


# Pipeline Errors
class PipelineError(CRQUBOError):
    """Raised when pipeline processing fails."""
    pass


class PipelineConfigurationError(PipelineError):
    """Raised when pipeline configuration is invalid."""
    pass


class PipelineExecutionError(PipelineError):
    """Raised when pipeline execution fails."""
    pass


# Solver Errors
class SolverError(CRQUBOError):
    """Base class for solver-related errors."""
    pass


class SolverNotAvailableError(SolverError):
    """Raised when requested solver is not available."""
    pass


class SolverFailedError(SolverError):
    """Raised when solver fails to find solution."""
    pass


class SolverTimeoutError(SolverError):
    """Raised when solver exceeds time limit."""
    pass


# Model Errors  
class ModelError(CRQUBOError):
    """Base class for model-related errors."""
    pass


class ModelNotFoundError(ModelError):
    """Raised when model is not found."""
    pass


class ModelLoadError(ModelError):
    """Raised when model fails to load."""
    pass


def handle_external_exception(e: Exception, context: str = "") -> CRQUBOError:
    """
    Convert external exceptions to CRQUBO exceptions.

    Args:
        e: External exception
        context: Context in which exception occurred

    Returns:
        Appropriate CRQUBO exception
    """
    # OpenAI exceptions
    if "openai" in str(type(e).__module__):
        if "rate_limit" in str(e).lower() or "ratelimit" in str(e).lower():
            return APIRateLimitError(
                f"OpenAI API rate limit exceeded: {str(e)}",
                recovery_hint="Wait before retrying or reduce request rate",
                context=context,
                original_error=str(e)
            )
        elif "timeout" in str(e).lower():
            return APITimeoutError(
                f"OpenAI API request timed out: {str(e)}",
                recovery_hint="Retry with exponential backoff",
                context=context,
                original_error=str(e)
            )
        elif "quota" in str(e).lower():
            return APIQuotaExceededError(
                f"OpenAI API quota exceeded: {str(e)}",
                recovery_hint="Check your API quota and billing status",
                context=context,
                original_error=str(e)
            )
        elif "api_key" in str(e).lower() or "authentication" in str(e).lower():
            return APIKeyError(
                f"OpenAI API key error: {str(e)}",
                recovery_hint="Check that OPENAI_API_KEY environment variable is set correctly",
                context=context,
                original_error=str(e)
            )
        else:
            return APIError(
                f"OpenAI API error: {str(e)}",
                recovery_hint="Check OpenAI API status and retry",
                context=context,
                original_error=str(e)
            )

    # Import errors
    if isinstance(e, ImportError):
        missing_module = str(e).split("'")[1] if "'" in str(e) else "unknown"
        return ModuleError(
            f"Required module not found: {missing_module}",
            recovery_hint=f"Install missing dependency: pip install {missing_module}",
            context=context,
            original_error=str(e)
        )

    # Timeout errors
    if isinstance(e, TimeoutError) or "timeout" in str(e).lower():
        return TimeoutError(
            f"Operation timed out: {str(e)}",
            recovery_hint="Increase timeout or reduce problem complexity",
            context=context,
            original_error=str(e)
        )

    # Memory errors
    if isinstance(e, MemoryError) or "memory" in str(e).lower():
        return MemoryError(
            f"Memory limit exceeded: {str(e)}",
            recovery_hint="Reduce batch size or enable streaming",
            context=context,
            original_error=str(e)
        )

    # Generic fallback
    return CRQUBOError(
        f"Unexpected error: {str(e)}",
        recovery_hint="Check logs for details and retry",
        context=context,
        original_error=str(e),
        error_type=type(e).__name__
    )
