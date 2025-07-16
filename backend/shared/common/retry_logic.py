# backend/services/shared/common/retry_logic.py
"""
Minimal Retry Logic for Advanced Feature Stability
Critical for Week 2 safe feature activation
"""

import asyncio
import random
import time
import logging
from typing import Callable, Any, List
from enum import Enum

logger = logging.getLogger(__name__)

class RetryStrategy(Enum):
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"

class RetryManager:
    """Simple retry manager for advanced features"""
    
    def __init__(self):
        self.default_config = {
            "max_retries": 2,
            "base_delay": 0.5,
            "max_delay": 5.0,
            "jitter": True
        }
    
    async def retry_with_strategy(
        self,
        func: Callable,
        *args,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
        max_retries: int = 2,
        base_delay: float = 0.5,
        max_delay: float = 5.0,
        jitter: bool = True,
        retry_on: List[Exception] = None,
        **kwargs
    ) -> Any:
        """Execute function with retry logic"""
        
        retry_on = retry_on or [Exception]
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if we should retry this exception
                if not any(isinstance(e, exc_type) for exc_type in retry_on):
                    raise e
                
                if attempt >= max_retries:
                    raise e
                
                # Calculate delay
                if strategy == RetryStrategy.EXPONENTIAL:
                    delay = base_delay * (2 ** attempt)
                elif strategy == RetryStrategy.LINEAR:
                    delay = base_delay * (attempt + 1)
                else:  # FIXED
                    delay = base_delay
                
                # Apply jitter
                if jitter:
                    delay += random.uniform(0, delay * 0.1)
                
                delay = min(delay, max_delay)
                
                logger.warning(f"Retry {attempt + 1}/{max_retries} for {func.__name__} after {delay:.2f}s. Error: {str(e)}")
                await asyncio.sleep(delay)
        
        raise last_exception

# Global instance
retry_manager = RetryManager()

# Convenience decorator
def retry(max_retries: int = 2, base_delay: float = 0.5):
    """Simple retry decorator"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            return await retry_manager.retry_with_strategy(
                func, *args,
                max_retries=max_retries,
                base_delay=base_delay,
                **kwargs
            )
        return wrapper
    return decorator