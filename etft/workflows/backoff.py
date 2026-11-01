import logging
import random
import time

logger = logging.getLogger(__name__)

def calculate_jittered_backoff(attempt: int, min_jitter: float = 0.5, max_jitter: float = 1.5) -> float:
    """
    Calculates the sleep time for a jittered exponential backoff.
    Based on the formula: T_sleep = 2^N + Random(min_jitter, max_jitter)

    Args:
        attempt (int): The current attempt number (0-indexed or 1-indexed, usually N in the formula).
        min_jitter (float): Minimum jitter to add in seconds.
        max_jitter (float): Maximum jitter to add in seconds.

    Returns:
        float: The sleep time in seconds.
    """
    exponential_part = 2 ** attempt
    jitter_part = random.uniform(min_jitter, max_jitter)
    sleep_time = exponential_part + jitter_part
    return sleep_time

def execute_with_backoff(func, max_attempts: int = 5, *args, **kwargs):
    """
    Executes a function with jittered exponential backoff on failure.

    Args:
        func: The function to execute.
        max_attempts: Maximum number of attempts before giving up.
        *args, **kwargs: Arguments to pass to the function.

    Raises:
        Exception: The last exception encountered if all attempts fail.
    """
    for attempt in range(max_attempts):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # You might want to filter specific exceptions here (e.g. RateLimitError/HTTP 429)
            if attempt == max_attempts - 1:
                logger.error(f"Max attempts reached. Final failure for {func.__name__}: {e}")
                raise

            sleep_time = calculate_jittered_backoff(attempt)
            logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
