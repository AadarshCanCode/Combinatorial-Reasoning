"""
Retry and resource management utilities for CRQUBO framework.

Provides decorators and functions for handling retries, timeouts,
circuit breakers, and resource limits.
"""

import functools
import logging
import time
from typing import Any, Callable, Optional, Tuple, Type

from .exceptions import (
    APIRateLimitError,
    APITimeoutError,
    ResourceLimitError,
    TimeoutError,
    handle_external_exception,
)

logger = logging.getLogger(__name__)

# Constants
DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_DELAY = 1.0
DEFAULT_MAX_DELAY = 60.0
DEFAULT_BACKOFF_FACTOR = 2.0
DEFAULT_TIMEOUT = 30.0


def retry_with_exponential_backoff(
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    logger_instance: Optional[logging.Logger] = None,
):
    """
    Decorator to retry a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Multiplier for delay after each retry
        retryable_exceptions: Tuple of exception types that should trigger retry
        logger_instance: Optional logger to use for logging retry attempts

    Returns:
        Decorated function that retries on failure
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    # Don't retry on last attempt
                    if attempt == max_retries:
                        break

                    # Check if this is a non-retryable error
                    if _is_non_retryable_error(e):
                        logger.warning(
                            f"{func.__name__} failed with non-retryable error: {e}"
                        )
                        raise

                    # Log retry attempt
                    log_func = logger_instance.warning if logger_instance else logger.warning
                    log_func(
                        f"{func.__name__} attempt {attempt + 1}/{max_retries + 1} "
                        f"failed: {e}. Retrying in {delay:.2f}s..."
                    )

                    # Wait before retry
                    time.sleep(delay)

                    # Increase delay for next retry (exponential backoff)
                    delay = min(delay * backoff_factor, max_delay)

            # All retries exhausted
            logger.error(
                f"{func.__name__} failed after {max_retries + 1} attempts"
            )
            # Convert to CRQUBO exception if needed
            if not isinstance(last_exception, Exception):
                raise last_exception
            crqubo_error = handle_external_exception(
                last_exception, context=f"{func.__name__}"
            )
            raise crqubo_error from last_exception

        return wrapper

    return decorator


def _is_non_retryable_error(error: Exception) -> bool:
    """
    Check if an error should not be retried.

    Args:
        error: Exception to check

    Returns:
        True if error should not be retried
    """
    error_str = str(error).lower()

    # Authentication/authorization errors - don't retry
    if any(
        keyword in error_str
        for keyword in ["authentication", "unauthorized", "forbidden", "api_key"]
    ):
        return True

    # Validation errors - don't retry
    if any(
        keyword in error_str
        for keyword in ["invalid", "validation", "bad request", "malformed"]
    ):
        return True

    # Quota exceeded - don't retry (different from rate limit)
    if "quota exceeded" in error_str:
        return True

    return False


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for API calls.

    Opens circuit after a threshold of failures, preventing further calls
    until a timeout period has passed.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_duration: float = 60.0,
        expected_exception: Type[Exception] = Exception,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            timeout_duration: How long to wait before trying again (seconds)
            expected_exception: Exception type to track
        """
        self.failure_threshold = failure_threshold
        self.timeout_duration = timeout_duration
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call a function through the circuit breaker.

        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            ResourceLimitError: If circuit is open
        """
        if self.state == "open":
            if time.time() - self.last_failure_time < self.timeout_duration:
                raise ResourceLimitError(
                    f"Circuit breaker is open for {func.__name__}",
                    recovery_hint=f"Wait {self.timeout_duration}s before retrying",
                    failure_count=self.failure_count,
                )
            else:
                self.state = "half_open"
                logger.info(f"Circuit breaker entering half-open state for {func.__name__}")

        try:
            result = func(*args, **kwargs)
            # Success - reset circuit breaker
            if self.state == "half_open":
                logger.info(f"Circuit breaker closing after successful call to {func.__name__}")
            self.failure_count = 0
            self.state = "closed"
            return result
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.error(
                    f"Circuit breaker opened for {func.__name__} "
                    f"after {self.failure_count} failures"
                )

            raise


def with_timeout(timeout: float = DEFAULT_TIMEOUT):
    """
    Decorator to enforce timeout on function execution.

    Args:
        timeout: Maximum execution time in seconds

    Returns:
        Decorated function with timeout
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError(
                    f"Function {func.__name__} exceeded timeout of {timeout}s",
                    recovery_hint="Increase timeout or reduce problem complexity",
                )

            # Set timeout alarm (Unix-like systems only)
            if hasattr(signal, "SIGALRM"):
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(timeout))
                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                return result
            else:
                # Fallback for systems without signal support (e.g., Windows)
                # Just run without timeout enforcement
                logger.warning(
                    f"Timeout enforcement not available on this platform for {func.__name__}"
                )
                return func(*args, **kwargs)

        return wrapper

    return decorator


class ResourceTracker:
    """
    Track resource usage (API calls, tokens, etc.) with limits.
    """

    def __init__(
        self,
        max_api_calls: Optional[int] = None,
        max_tokens: Optional[int] = None,
        tracking_window: float = 3600.0,  # 1 hour
    ):
        """
        Initialize resource tracker.

        Args:
            max_api_calls: Maximum API calls in tracking window
            max_tokens: Maximum tokens in tracking window
            tracking_window: Time window for tracking (seconds)
        """
        self.max_api_calls = max_api_calls
        self.max_tokens = max_tokens
        self.tracking_window = tracking_window
        self.api_calls = []
        self.tokens = []

    def _prune_old_entries(self) -> None:
        """Remove entries outside of the tracking window."""
        current_time = time.time()
        self.api_calls = [t for t in self.api_calls if current_time - t < self.tracking_window]
        self.tokens = [
            (t, tok) for t, tok in self.tokens if current_time - t < self.tracking_window
        ]

    def ensure_call_allowed(self) -> None:
        """Ensure another API call is allowed within limits."""
        self._prune_old_entries()
        if self.max_api_calls and len(self.api_calls) >= self.max_api_calls:
            raise ResourceLimitError(
                f"API call limit of {self.max_api_calls} exceeded in {self.tracking_window}s",
                recovery_hint="Wait before making more API calls or increase limit",
                api_calls_made=len(self.api_calls),
            )

    def track_api_call(self, tokens_used: int = 0) -> None:
        """
        Track an API call.

        Args:
            tokens_used: Number of tokens used in the call

        Raises:
            ResourceLimitError: If limits are exceeded
        """
        self._prune_old_entries()

        # Check limits before recording tokens
        if self.max_api_calls and len(self.api_calls) >= self.max_api_calls:
            raise ResourceLimitError(
                f"API call limit of {self.max_api_calls} exceeded in {self.tracking_window}s",
                recovery_hint="Wait before making more API calls or increase limit",
                api_calls_made=len(self.api_calls),
            )

        total_tokens = sum(tok for _, tok in self.tokens)
        if self.max_tokens and total_tokens + tokens_used > self.max_tokens:
            raise ResourceLimitError(
                f"Token limit of {self.max_tokens} exceeded in {self.tracking_window}s",
                recovery_hint="Wait before using more tokens or increase limit",
                tokens_used=total_tokens + tokens_used,
            )

        # Track this call
        self.api_calls.append(time.time())
        if tokens_used > 0:
            self.tokens.append((time.time(), tokens_used))

    def get_usage_stats(self) -> dict:
        """Get current resource usage statistics."""
        current_time = time.time()

        # Clean old entries
        self.api_calls = [t for t in self.api_calls if current_time - t < self.tracking_window]
        self.tokens = [
            (t, tok) for t, tok in self.tokens if current_time - t < self.tracking_window
        ]

        return {
            "api_calls": len(self.api_calls),
            "max_api_calls": self.max_api_calls,
            "tokens_used": sum(tok for _, tok in self.tokens),
            "max_tokens": self.max_tokens,
            "tracking_window": self.tracking_window,
        }
