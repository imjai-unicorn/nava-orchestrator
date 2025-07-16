# backend/services/03-external-ai/gpt-client/app/timeout_handler.py
"""
GPT Service Timeout Handler - แก้ปัญหา AI response >5s
Week 1: Critical Fix for Performance Issues
"""

import asyncio
import time
from typing import Optional, Dict, Any, List
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class TimeoutStrategy(Enum):
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    IMMEDIATE_FALLBACK = "immediate_fallback"

class GPTTimeoutHandler:
    """Enhanced timeout handling for GPT service"""
    
    def __init__(self):
        self.timeout_settings = {
            'default_timeout': 8.0,  # 15 seconds default
            'max_retries': 2,
            'backoff_factor': 1.5,
            'jitter_max': 0.5,
            'circuit_breaker_threshold': 3,
            'recovery_time': 30.0  # 1 minute recovery
        }
        
        self.failure_count = 0
        self.last_failure_time = 0
        self.circuit_open = False
        
        # Performance tracking
        self.response_times = []
        self.success_count = 0
        self.total_requests = 0
        
    def calculate_timeout(self, complexity_level: str = "medium") -> float:
        """Calculate dynamic timeout based on request complexity"""
        base_timeouts = {
            "simple": 5.0,      # Simple chat: 8s
            "medium": 8.0,     # Normal requests: 15s
            "complex": 15.0,    # Complex analysis: 25s
            "critical": 20.0    # Critical workflows: 35s
        }
        return base_timeouts.get(complexity_level, 15.0)
    
    def should_use_circuit_breaker(self) -> bool:
        """Check if circuit breaker should be activated"""
        if self.circuit_open:
            # Check if recovery time has passed
            if time.time() - self.last_failure_time > self.timeout_settings['recovery_time']:
                logger.info("Circuit breaker recovery period completed, attempting to close")
                self.circuit_open = False
                self.failure_count = 0
                return False
            return True
        
        return self.failure_count >= self.timeout_settings['circuit_breaker_threshold']
    
    async def call_with_timeout(
        self, 
        api_call_func, 
        *args, 
        complexity_level: str = "medium",
        fallback_func=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute API call with intelligent timeout handling
        
        Args:
            api_call_func: The actual GPT API call function
            complexity_level: Request complexity for timeout calculation
            fallback_func: Fallback function if all retries fail
        """
        self.total_requests += 1
        start_time = time.time()
        
        # Check circuit breaker
        if self.should_use_circuit_breaker():
            if self.circuit_open:
                logger.warning("Circuit breaker is OPEN - using fallback immediately")
                return await self._handle_circuit_breaker_fallback(fallback_func, *args, **kwargs)
            else:
                # Open circuit breaker
                logger.error(f"Opening circuit breaker after {self.failure_count} failures")
                self.circuit_open = True
                self.last_failure_time = time.time()
                return await self._handle_circuit_breaker_fallback(fallback_func, *args, **kwargs)
        
        timeout = self.calculate_timeout(complexity_level)
        max_retries = self.timeout_settings['max_retries']
        
        for attempt in range(max_retries):
            try:
                logger.info(f"GPT API call attempt {attempt + 1}/{max_retries}, timeout: {timeout}s")
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    api_call_func(*args, **kwargs),
                    timeout=timeout
                )
                
                # Success - record metrics
                response_time = time.time() - start_time
                self._record_success(response_time)
                
                logger.info(f"GPT API call successful in {response_time:.2f}s")
                return {
                    "success": True,
                    "response": result,
                    "response_time": response_time,
                    "attempt": attempt + 1,
                    "timeout_used": timeout
                }
                
            except asyncio.TimeoutError:
                response_time = time.time() - start_time
                logger.warning(f"GPT API timeout after {response_time:.2f}s (attempt {attempt + 1})")
                
                if attempt < max_retries - 1:
                    # Calculate backoff delay
                    delay = self._calculate_backoff_delay(attempt)
                    logger.info(f"Retrying after {delay:.2f}s delay...")
                    await asyncio.sleep(delay)
                    
                    # Increase timeout for next attempt
                    timeout = min(timeout * 1.5, 45.0)  # Max 45s timeout
                else:
                    # Final timeout - record failure
                    self._record_failure()
                    logger.error(f"GPT API final timeout after {max_retries} attempts")
                    
                    if fallback_func:
                        return await self._handle_fallback(fallback_func, *args, **kwargs)
                    
                    return {
                        "success": False,
                        "error": "timeout",
                        "message": f"GPT API timeout after {max_retries} attempts",
                        "total_time": response_time,
                        "max_timeout": timeout
                    }
                    
            except Exception as e:
                response_time = time.time() - start_time
                logger.error(f"GPT API error: {str(e)} (attempt {attempt + 1})")
                
                if attempt < max_retries - 1:
                    delay = self._calculate_backoff_delay(attempt)
                    await asyncio.sleep(delay)
                else:
                    self._record_failure()
                    
                    if fallback_func:
                        return await self._handle_fallback(fallback_func, *args, **kwargs)
                    
                    return {
                        "success": False,
                        "error": "api_error",
                        "message": str(e),
                        "total_time": response_time
                    }
        
        # Should never reach here
        return {
            "success": False,
            "error": "unknown",
            "message": "Unexpected error in timeout handler"
        }
    
    def _calculate_backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter"""
        base_delay = self.timeout_settings['backoff_factor'] ** attempt
        jitter = __import__('random').uniform(0, self.timeout_settings['jitter_max'])
        return base_delay + jitter
    
    def _record_success(self, response_time: float):
        """Record successful API call metrics"""
        self.success_count += 1
        self.response_times.append(response_time)
        
        # Reset failure count on success
        if self.failure_count > 0:
            logger.info(f"GPT service recovered after {self.failure_count} failures")
            self.failure_count = 0
        
        # Keep only recent response times (last 100)
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]
    
    def _record_failure(self):
        """Record API call failure"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        logger.warning(f"GPT service failure count: {self.failure_count}")
    
    async def _handle_fallback(self, fallback_func, *args, **kwargs) -> Dict[str, Any]:
        """Handle fallback when all retries fail"""
        try:
            logger.info("Attempting GPT fallback function...")
            fallback_result = await fallback_func(*args, **kwargs)
            return {
                "success": True,
                "response": fallback_result,
                "fallback_used": True,
                "message": "Fallback function executed successfully"
            }
        except Exception as e:
            logger.error(f"Fallback function failed: {str(e)}")
            return {
                "success": False,
                "error": "fallback_failed",
                "message": f"Both main and fallback functions failed: {str(e)}"
            }
    
    async def _handle_circuit_breaker_fallback(self, fallback_func, *args, **kwargs) -> Dict[str, Any]:
        """Handle circuit breaker state with immediate fallback"""
        if fallback_func:
            return await self._handle_fallback(fallback_func, *args, **kwargs)
        
        return {
            "success": False,
            "error": "circuit_breaker_open",
            "message": "GPT service circuit breaker is open, no fallback available",
            "retry_after": self.timeout_settings['recovery_time']
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status and metrics"""
        avg_response_time = (
            sum(self.response_times) / len(self.response_times) 
            if self.response_times else 0
        )
        
        success_rate = (
            (self.success_count / self.total_requests * 100) 
            if self.total_requests > 0 else 100
        )
        
        return {
            "service": "gpt",
            "status": "circuit_open" if self.circuit_open else "healthy",
            "circuit_breaker_open": self.circuit_open,
            "failure_count": self.failure_count,
            "success_rate": round(success_rate, 2),
            "avg_response_time": round(avg_response_time, 2),
            "total_requests": self.total_requests,
            "last_failure_time": self.last_failure_time,
            "current_timeout_settings": self.timeout_settings
        }
    
    def reset_circuit_breaker(self):
        """Manually reset circuit breaker (for admin use)"""
        logger.info("Manually resetting GPT circuit breaker")
        self.circuit_open = False
        self.failure_count = 0
        self.last_failure_time = 0

# Global instance
gpt_timeout_handler = GPTTimeoutHandler()

# Convenience functions
async def call_gpt_with_timeout(api_func, *args, complexity="medium", fallback=None, **kwargs):
    """Convenience function for GPT API calls with timeout handling"""
    return await gpt_timeout_handler.call_with_timeout(
        api_func, *args, 
        complexity_level=complexity, 
        fallback_func=fallback, 
        **kwargs
    )

def get_gpt_health():
    """Get GPT service health status"""
    return gpt_timeout_handler.get_health_status()

def reset_gpt_circuit_breaker():
    """Reset GPT circuit breaker"""
    gpt_timeout_handler.reset_circuit_breaker()