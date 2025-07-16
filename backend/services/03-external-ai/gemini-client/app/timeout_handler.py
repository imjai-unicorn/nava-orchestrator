# backend/services/03-external-ai/gemini-client/app/timeout_handler.py
"""
Gemini Service Timeout Handler - แก้ปัญหา AI response >5s
Week 1: Critical Fix for Performance Issues
Optimized for Gemini's multimodal and search capabilities
"""

import asyncio
import time
from typing import Optional, Dict, Any, List
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class GeminiTimeoutHandler:
    """Enhanced timeout handling for Gemini service - optimized for multimodal and search tasks"""
    
    def __init__(self):
        self.timeout_settings = {
            'default_timeout': 18.0,  # Gemini balance between speed and capability
            'max_retries': 3,
            'backoff_factor': 2.2,   # Slightly more aggressive for search tasks
            'jitter_max': 1.2,
            'circuit_breaker_threshold': 5,
            'recovery_time': 75.0    # 75 seconds recovery
        }
        
        self.failure_count = 0
        self.last_failure_time = 0
        self.circuit_open = False
        
        # Gemini-specific performance tracking
        self.response_times = []
        self.multimodal_performance = []
        self.search_performance = []
        self.success_count = 0
        self.total_requests = 0
        
    def calculate_timeout(
        self, 
        complexity_level: str = "medium", 
        has_multimodal: bool = False,
        needs_search: bool = False
    ) -> float:
        """Calculate dynamic timeout for Gemini based on multimodal and search requirements"""
        base_timeouts = {
            "simple": 10.0,     # Simple text: 10s
            "medium": 18.0,     # Normal requests: 18s 
            "complex": 30.0,    # Complex analysis: 30s
            "critical": 45.0    # Critical workflows: 45s
        }
        
        base_timeout = base_timeouts.get(complexity_level, 18.0)
        
        # Multimodal processing needs more time
        if has_multimodal:
            base_timeout *= 1.8  # 80% more time for image/video processing
        
        # Search integration needs additional time
        if needs_search:
            base_timeout *= 1.4  # 40% more time for search integration
        
        # If both multimodal and search, cap the maximum
        if has_multimodal and needs_search:
            base_timeout = min(base_timeout, 90.0)  # Max 90s for complex multimodal+search
        
        return base_timeout
    
    def should_use_circuit_breaker(self) -> bool:
        """Check if circuit breaker should be activated for Gemini"""
        if self.circuit_open:
            # Check if recovery time has passed
            if time.time() - self.last_failure_time > self.timeout_settings['recovery_time']:
                logger.info("Gemini circuit breaker recovery period completed")
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
        has_multimodal: bool = False,
        needs_search: bool = False,
        fallback_func=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute Gemini API call with multimodal-optimized timeout handling
        
        Args:
            api_call_func: The actual Gemini API call function
            complexity_level: Request complexity for timeout calculation
            has_multimodal: Whether request includes images/video
            needs_search: Whether request needs search integration
            fallback_func: Fallback function if all retries fail
        """
        self.total_requests += 1
        start_time = time.time()
        
        # Check circuit breaker
        if self.should_use_circuit_breaker():
            if self.circuit_open:
                logger.warning("Gemini circuit breaker is OPEN - using fallback")
                return await self._handle_circuit_breaker_fallback(fallback_func, *args, **kwargs)
            else:
                # Open circuit breaker
                logger.error(f"Opening Gemini circuit breaker after {self.failure_count} failures")
                self.circuit_open = True
                self.last_failure_time = time.time()
                return await self._handle_circuit_breaker_fallback(fallback_func, *args, **kwargs)
        
        timeout = self.calculate_timeout(complexity_level, has_multimodal, needs_search)
        max_retries = self.timeout_settings['max_retries']
        
        features = []
        if has_multimodal:
            features.append("multimodal")
        if needs_search:
            features.append("search")
        features_str = "+".join(features) if features else "text-only"
        
        logger.info(f"Gemini API call starting - complexity: {complexity_level}, features: {features_str}, timeout: {timeout}s")
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Gemini API attempt {attempt + 1}/{max_retries}, timeout: {timeout}s")
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    api_call_func(*args, **kwargs),
                    timeout=timeout
                )
                
                # Success - record metrics with feature-specific assessment
                response_time = time.time() - start_time
                self._record_success(response_time, has_multimodal, needs_search, result)
                
                logger.info(f"Gemini API successful - {response_time:.2f}s, features: {features_str}")
                return {
                    "success": True,
                    "response": result,
                    "response_time": response_time,
                    "attempt": attempt + 1,
                    "timeout_used": timeout,
                    "features_used": features,
                    "service": "gemini"
                }
                
            except asyncio.TimeoutError:
                response_time = time.time() - start_time
                logger.warning(f"Gemini API timeout after {response_time:.2f}s (attempt {attempt + 1}, features: {features_str})")
                
                if attempt < max_retries - 1:
                    delay = self._calculate_backoff_delay(attempt)
                    logger.info(f"Gemini retrying after {delay:.2f}s delay...")
                    await asyncio.sleep(delay)
                    
                    # Increase timeout based on features
                    if has_multimodal:
                        timeout = min(timeout * 1.6, 120.0)  # More generous for multimodal
                    elif needs_search:
                        timeout = min(timeout * 1.5, 80.0)   # Moderate increase for search
                    else:
                        timeout = min(timeout * 1.3, 50.0)   # Standard increase for text
                else:
                    # Final timeout
                    self._record_failure(has_multimodal, needs_search)
                    logger.error(f"Gemini API final timeout after {max_retries} attempts")
                    
                    if fallback_func:
                        return await self._handle_fallback(fallback_func, *args, **kwargs)
                    
                    return {
                        "success": False,
                        "error": "timeout",
                        "message": f"Gemini API timeout after {max_retries} attempts",
                        "total_time": response_time,
                        "max_timeout": timeout,
                        "service": "gemini",
                        "features_attempted": features
                    }
                    
            except Exception as e:
                response_time = time.time() - start_time
                logger.error(f"Gemini API error: {str(e)} (attempt {attempt + 1})")
                
                if attempt < max_retries - 1:
                    delay = self._calculate_backoff_delay(attempt)
                    await asyncio.sleep(delay)
                else:
                    self._record_failure(has_multimodal, needs_search)
                    
                    if fallback_func:
                        return await self._handle_fallback(fallback_func, *args, **kwargs)
                    
                    return {
                        "success": False,
                        "error": "api_error",
                        "message": str(e),
                        "total_time": response_time,
                        "service": "gemini"
                    }
        
        return {
            "success": False,
            "error": "unknown",
            "message": "Unexpected error in Gemini timeout handler",
            "service": "gemini"
        }
    
    def _calculate_backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter for Gemini"""
        base_delay = self.timeout_settings['backoff_factor'] ** attempt
        jitter = __import__('random').uniform(0, self.timeout_settings['jitter_max'])
        return base_delay + jitter
    
    def _record_success(
        self, 
        response_time: float, 
        has_multimodal: bool, 
        needs_search: bool, 
        response_content: Any = None
    ):
        """Record successful Gemini API call with feature-specific metrics"""
        self.success_count += 1
        self.response_times.append(response_time)
        
        # Track feature-specific performance
        if has_multimodal:
            self.multimodal_performance.append(response_time)
        if needs_search:
            self.search_performance.append(response_time)
        
        # Reset failure count on success
        if self.failure_count > 0:
            logger.info(f"Gemini service recovered after {self.failure_count} failures")
            self.failure_count = 0
        
        # Keep only recent metrics (last 100)
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]
        if len(self.multimodal_performance) > 50:
            self.multimodal_performance = self.multimodal_performance[-50:]
        if len(self.search_performance) > 50:
            self.search_performance = self.search_performance[-50:]
    
    def _record_failure(self, has_multimodal: bool = False, needs_search: bool = False):
        """Record Gemini API call failure with feature context"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        features = []
        if has_multimodal:
            features.append("multimodal")
        if needs_search:
            features.append("search")
        features_str = "+".join(features) if features else "text-only"
        
        logger.warning(f"Gemini service failure count: {self.failure_count} (features: {features_str})")
    
    async def _handle_fallback(self, fallback_func, *args, **kwargs) -> Dict[str, Any]:
        """Handle fallback when Gemini retries fail"""
        try:
            logger.info("Attempting Gemini fallback function...")
            fallback_result = await fallback_func(*args, **kwargs)
            return {
                "success": True,
                "response": fallback_result,
                "fallback_used": True,
                "message": "Gemini fallback function executed successfully",
                "service": "gemini_fallback"
            }
        except Exception as e:
            logger.error(f"Gemini fallback function failed: {str(e)}")
            return {
                "success": False,
                "error": "fallback_failed",
                "message": f"Gemini main and fallback functions failed: {str(e)}",
                "service": "gemini"
            }
    
    async def _handle_circuit_breaker_fallback(self, fallback_func, *args, **kwargs) -> Dict[str, Any]:
        """Handle Gemini circuit breaker state"""
        if fallback_func:
            return await self._handle_fallback(fallback_func, *args, **kwargs)
        
        return {
            "success": False,
            "error": "circuit_breaker_open",
            "message": "Gemini service circuit breaker is open, no fallback available",
            "retry_after": self.timeout_settings['recovery_time'],
            "service": "gemini"
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current Gemini health status and feature-specific metrics"""
        avg_response_time = (
            sum(self.response_times) / len(self.response_times) 
            if self.response_times else 0
        )
        
        avg_multimodal_time = (
            sum(self.multimodal_performance) / len(self.multimodal_performance)
            if self.multimodal_performance else 0
        )
        
        avg_search_time = (
            sum(self.search_performance) / len(self.search_performance)
            if self.search_performance else 0
        )
        
        success_rate = (
            (self.success_count / self.total_requests * 100) 
            if self.total_requests > 0 else 100
        )
        
        return {
            "service": "gemini",
            "status": "circuit_open" if self.circuit_open else "healthy",
            "circuit_breaker_open": self.circuit_open,
            "failure_count": self.failure_count,
            "success_rate": round(success_rate, 2),
            "avg_response_time": round(avg_response_time, 2),
            "avg_multimodal_time": round(avg_multimodal_time, 2),
            "avg_search_time": round(avg_search_time, 2),
            "multimodal_requests": len(self.multimodal_performance),
            "search_requests": len(self.search_performance),
            "total_requests": self.total_requests,
            "last_failure_time": self.last_failure_time,
            "current_timeout_settings": self.timeout_settings,
            "features_supported": ["multimodal", "search", "text"]
        }
    
    def reset_circuit_breaker(self):
        """Manually reset Gemini circuit breaker"""
        logger.info("Manually resetting Gemini circuit breaker")
        self.circuit_open = False
        self.failure_count = 0
        self.last_failure_time = 0

# Global instance
gemini_timeout_handler = GeminiTimeoutHandler()

# Convenience functions
async def call_gemini_with_timeout(
    api_func, *args, 
    complexity="medium", 
    multimodal=False,
    search=False,
    fallback=None, 
    **kwargs
):
    """Convenience function for Gemini API calls with multimodal-optimized timeout handling"""
    return await gemini_timeout_handler.call_with_timeout(
        api_func, *args, 
        complexity_level=complexity, 
        has_multimodal=multimodal,
        needs_search=search,
        fallback_func=fallback, 
        **kwargs
    )

def get_gemini_health():
    """Get Gemini service health status"""
    return gemini_timeout_handler.get_health_status()

def reset_gemini_circuit_breaker():
    """Reset Gemini circuit breaker"""
    gemini_timeout_handler.reset_circuit_breaker()