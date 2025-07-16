# backend/services/03-external-ai/claude-client/app/timeout_handler.py
"""
Claude Service Timeout Handler - แก้ปัญหา AI response >5s
Week 1: Critical Fix for Performance Issues
Optimized for Claude's reasoning capabilities
"""

import asyncio
import time
from typing import Optional, Dict, Any, List
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class ClaudeTimeoutHandler:
    """Enhanced timeout handling for Claude service - optimized for reasoning tasks"""
    
    def __init__(self):
        self.timeout_settings = {
            'default_timeout': 20.0,  # Claude needs more time for reasoning
            'max_retries': 3,
            'backoff_factor': 1.8,   # Gentler backoff for Claude
            'jitter_max': 0.8,
            'circuit_breaker_threshold': 4,  # More tolerant for reasoning tasks
            'recovery_time': 90.0    # Longer recovery for complex reasoning
        }
        
        self.failure_count = 0
        self.last_failure_time = 0
        self.circuit_open = False
        
        # Claude-specific performance tracking
        self.response_times = []
        self.reasoning_quality_scores = []
        self.success_count = 0
        self.total_requests = 0
        
    def calculate_timeout(self, complexity_level: str = "medium", reasoning_depth: str = "standard") -> float:
        """Calculate dynamic timeout for Claude based on reasoning requirements"""
        base_timeouts = {
            "simple": 12.0,     # Simple chat: 12s (Claude needs time for quality)
            "medium": 20.0,     # Normal requests: 20s 
            "complex": 35.0,    # Complex analysis: 35s
            "critical": 50.0    # Deep reasoning: 50s
        }
        
        reasoning_multipliers = {
            "quick": 0.8,       # Quick responses
            "standard": 1.0,    # Standard reasoning
            "deep": 1.5,        # Deep analysis
            "comprehensive": 2.0 # Comprehensive reasoning
        }
        
        base_timeout = base_timeouts.get(complexity_level, 20.0)
        multiplier = reasoning_multipliers.get(reasoning_depth, 1.0)
        
        return base_timeout * multiplier
    
    def should_use_circuit_breaker(self) -> bool:
        """Check if circuit breaker should be activated for Claude"""
        if self.circuit_open:
            # Check if recovery time has passed
            if time.time() - self.last_failure_time > self.timeout_settings['recovery_time']:
                logger.info("Claude circuit breaker recovery period completed")
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
        reasoning_depth: str = "standard",
        fallback_func=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute Claude API call with reasoning-optimized timeout handling
        
        Args:
            api_call_func: The actual Claude API call function
            complexity_level: Request complexity for timeout calculation
            reasoning_depth: Reasoning depth requirement
            fallback_func: Fallback function if all retries fail
        """
        self.total_requests += 1
        start_time = time.time()
        
        # Check circuit breaker
        if self.should_use_circuit_breaker():
            if self.circuit_open:
                logger.warning("Claude circuit breaker is OPEN - using fallback")
                return await self._handle_circuit_breaker_fallback(fallback_func, *args, **kwargs)
            else:
                # Open circuit breaker
                logger.error(f"Opening Claude circuit breaker after {self.failure_count} failures")
                self.circuit_open = True
                self.last_failure_time = time.time()
                return await self._handle_circuit_breaker_fallback(fallback_func, *args, **kwargs)
        
        timeout = self.calculate_timeout(complexity_level, reasoning_depth)
        max_retries = self.timeout_settings['max_retries']
        
        logger.info(f"Claude API call starting - complexity: {complexity_level}, reasoning: {reasoning_depth}, timeout: {timeout}s")
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Claude API attempt {attempt + 1}/{max_retries}, timeout: {timeout}s")
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    api_call_func(*args, **kwargs),
                    timeout=timeout
                )
                
                # Success - record metrics with quality assessment
                response_time = time.time() - start_time
                self._record_success(response_time, result)
                
                logger.info(f"Claude API successful - {response_time:.2f}s, reasoning quality assessed")
                return {
                    "success": True,
                    "response": result,
                    "response_time": response_time,
                    "attempt": attempt + 1,
                    "timeout_used": timeout,
                    "reasoning_depth": reasoning_depth,
                    "service": "claude"
                }
                
            except asyncio.TimeoutError:
                response_time = time.time() - start_time
                logger.warning(f"Claude API timeout after {response_time:.2f}s (attempt {attempt + 1})")
                
                if attempt < max_retries - 1:
                    # For Claude, we increase timeout more aggressively for reasoning tasks
                    delay = self._calculate_backoff_delay(attempt)
                    logger.info(f"Claude retrying after {delay:.2f}s delay...")
                    await asyncio.sleep(delay)
                    
                    # Increase timeout for reasoning tasks
                    if reasoning_depth in ["deep", "comprehensive"]:
                        timeout = min(timeout * 1.8, 80.0)  # Max 80s for deep reasoning
                    else:
                        timeout = min(timeout * 1.4, 45.0)  # Max 45s for normal tasks
                else:
                    # Final timeout
                    self._record_failure()
                    logger.error(f"Claude API final timeout after {max_retries} attempts")
                    
                    if fallback_func:
                        return await self._handle_fallback(fallback_func, *args, **kwargs)
                    
                    return {
                        "success": False,
                        "error": "timeout",
                        "message": f"Claude API timeout after {max_retries} attempts",
                        "total_time": response_time,
                        "max_timeout": timeout,
                        "service": "claude",
                        "reasoning_depth": reasoning_depth
                    }
                    
            except Exception as e:
                response_time = time.time() - start_time
                logger.error(f"Claude API error: {str(e)} (attempt {attempt + 1})")
                
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
                        "total_time": response_time,
                        "service": "claude"
                    }
        
        return {
            "success": False,
            "error": "unknown",
            "message": "Unexpected error in Claude timeout handler",
            "service": "claude"
        }
    
    def _calculate_backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter for Claude"""
        base_delay = self.timeout_settings['backoff_factor'] ** attempt
        jitter = __import__('random').uniform(0, self.timeout_settings['jitter_max'])
        return base_delay + jitter
    
    def _record_success(self, response_time: float, response_content: Any = None):
        """Record successful Claude API call with reasoning quality assessment"""
        self.success_count += 1
        self.response_times.append(response_time)
        
        # Assess reasoning quality (basic heuristic)
        quality_score = self._assess_reasoning_quality(response_content, response_time)
        self.reasoning_quality_scores.append(quality_score)
        
        # Reset failure count on success
        if self.failure_count > 0:
            logger.info(f"Claude service recovered after {self.failure_count} failures")
            self.failure_count = 0
        
        # Keep only recent metrics (last 100)
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]
            self.reasoning_quality_scores = self.reasoning_quality_scores[-100:]
    
    def _assess_reasoning_quality(self, response_content: Any, response_time: float) -> float:
        """Basic reasoning quality assessment for Claude responses"""
        if not response_content:
            return 0.5
        
        # Convert response to string for analysis
        response_text = str(response_content)
        
        quality_score = 0.0
        
        # Length factor (good reasoning needs adequate length)
        if len(response_text) > 100:
            quality_score += 0.3
        if len(response_text) > 500:
            quality_score += 0.2
        
        # Reasoning indicators
        reasoning_indicators = [
            "because", "therefore", "however", "although", "considering",
            "analysis", "conclusion", "evidence", "reasoning", "logic"
        ]
        
        indicator_count = sum(1 for indicator in reasoning_indicators if indicator in response_text.lower())
        quality_score += min(indicator_count * 0.1, 0.3)
        
        # Time efficiency (balance quality vs speed)
        if response_time < 10:
            quality_score += 0.1  # Fast responses get bonus
        elif response_time > 40:
            quality_score -= 0.1  # Very slow responses lose points
        
        return min(max(quality_score, 0.0), 1.0)
    
    def _record_failure(self):
        """Record Claude API call failure"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        logger.warning(f"Claude service failure count: {self.failure_count}")
    
    async def _handle_fallback(self, fallback_func, *args, **kwargs) -> Dict[str, Any]:
        """Handle fallback when Claude retries fail"""
        try:
            logger.info("Attempting Claude fallback function...")
            fallback_result = await fallback_func(*args, **kwargs)
            return {
                "success": True,
                "response": fallback_result,
                "fallback_used": True,
                "message": "Claude fallback function executed successfully",
                "service": "claude_fallback"
            }
        except Exception as e:
            logger.error(f"Claude fallback function failed: {str(e)}")
            return {
                "success": False,
                "error": "fallback_failed",
                "message": f"Claude main and fallback functions failed: {str(e)}",
                "service": "claude"
            }
    
    async def _handle_circuit_breaker_fallback(self, fallback_func, *args, **kwargs) -> Dict[str, Any]:
        """Handle Claude circuit breaker state"""
        if fallback_func:
            return await self._handle_fallback(fallback_func, *args, **kwargs)
        
        return {
            "success": False,
            "error": "circuit_breaker_open",
            "message": "Claude service circuit breaker is open, no fallback available",
            "retry_after": self.timeout_settings['recovery_time'],
            "service": "claude"
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current Claude health status and reasoning metrics"""
        avg_response_time = (
            sum(self.response_times) / len(self.response_times) 
            if self.response_times else 0
        )
        
        avg_reasoning_quality = (
            sum(self.reasoning_quality_scores) / len(self.reasoning_quality_scores)
            if self.reasoning_quality_scores else 0
        )
        
        success_rate = (
            (self.success_count / self.total_requests * 100) 
            if self.total_requests > 0 else 100
        )
        
        return {
            "service": "claude",
            "status": "circuit_open" if self.circuit_open else "healthy",
            "circuit_breaker_open": self.circuit_open,
            "failure_count": self.failure_count,
            "success_rate": round(success_rate, 2),
            "avg_response_time": round(avg_response_time, 2),
            "avg_reasoning_quality": round(avg_reasoning_quality, 2),
            "total_requests": self.total_requests,
            "last_failure_time": self.last_failure_time,
            "current_timeout_settings": self.timeout_settings,
            "reasoning_optimized": True
        }
    
    def reset_circuit_breaker(self):
        """Manually reset Claude circuit breaker"""
        logger.info("Manually resetting Claude circuit breaker")
        self.circuit_open = False
        self.failure_count = 0
        self.last_failure_time = 0

# Global instance
claude_timeout_handler = ClaudeTimeoutHandler()

# Convenience functions
async def call_claude_with_timeout(
    api_func, *args, 
    complexity="medium", 
    reasoning="standard", 
    fallback=None, 
    **kwargs
):
    """Convenience function for Claude API calls with reasoning-optimized timeout handling"""
    return await claude_timeout_handler.call_with_timeout(
        api_func, *args, 
        complexity_level=complexity, 
        reasoning_depth=reasoning,
        fallback_func=fallback, 
        **kwargs
    )

def get_claude_health():
    """Get Claude service health status"""
    return claude_timeout_handler.get_health_status()

def reset_claude_circuit_breaker():
    """Reset Claude circuit breaker"""
    claude_timeout_handler.reset_circuit_breaker()