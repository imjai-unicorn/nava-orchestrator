"""
Enhanced Circuit Breaker for NAVA AI Services
Fixes timeout issues and provides intelligent failover
File: backend/services/shared/common/enhanced_circuit_breaker.py
"""

import time
import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import random
import json

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"     # Normal operation
    OPEN = "open"         # Circuit breaker activated
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class ServiceConfig:
    """Configuration for each AI service"""
    name: str
    timeout: float
    max_failures: int = 5
    failure_threshold: float = 0.5  # 50% failure rate
    recovery_timeout: float = 60.0  # 1 minute
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter: bool = True

@dataclass
class ServiceHealth:
    """Health tracking for each service"""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0
    last_success_time: float = 0
    total_requests: int = 0
    consecutive_failures: int = 0
    recovery_attempts: int = 0

class EnhancedCircuitBreaker:
    """
    Enhanced Circuit Breaker with per-service configuration
    Handles GPT, Claude, Gemini with different timeout settings
    """
    
    def __init__(self):
        self.services: Dict[str, ServiceConfig] = {}
        self.health: Dict[str, ServiceHealth] = {}
        self.failover_chain: List[str] = []
        self.setup_default_services()
        
    def setup_default_services(self):
        """Setup default configurations for AI services"""
        # GPT Configuration
        self.services["gpt"] = ServiceConfig(
            name="gpt",
            timeout=15.0,
            max_failures=5,
            failure_threshold=0.3,
            recovery_timeout=30.0,
            max_retries=3,
            base_delay=1.0,
            max_delay=10.0
        )
        
        # Claude Configuration
        self.services["claude"] = ServiceConfig(
            name="claude",
            timeout=20.0,
            max_failures=3,
            failure_threshold=0.2,
            recovery_timeout=45.0,
            max_retries=2,
            base_delay=2.0,
            max_delay=15.0
        )
        
        # Gemini Configuration
        self.services["gemini"] = ServiceConfig(
            name="gemini",
            timeout=18.0,
            max_failures=4,
            failure_threshold=0.25,
            recovery_timeout=40.0,
            max_retries=3,
            base_delay=1.5,
            max_delay=12.0
        )
        
        # Local AI Configuration
        self.services["local"] = ServiceConfig(
            name="local",
            timeout=5.0,
            max_failures=2,
            failure_threshold=0.1,
            recovery_timeout=10.0,
            max_retries=1,
            base_delay=0.5,
            max_delay=2.0
        )
        
        # Initialize health tracking
        for service_name in self.services.keys():
            self.health[service_name] = ServiceHealth()
            
        # Setup failover chain
        self.failover_chain = ["gpt", "claude", "gemini", "local", "cache"]
    
    def get_service_config(self, service_name: str) -> ServiceConfig:
        """Get configuration for a service"""
        return self.services.get(service_name.lower(), self.services["gpt"])
    
    def get_service_health(self, service_name: str) -> ServiceHealth:
        """Get health status for a service"""
        service_key = service_name.lower()
        if service_key not in self.health:
            self.health[service_key] = ServiceHealth()
        return self.health[service_key]
    
    def calculate_delay(self, attempt: int, config: ServiceConfig) -> float:
        """Calculate exponential backoff delay with jitter"""
        delay = min(config.base_delay * (2 ** attempt), config.max_delay)
        
        if config.jitter:
            # Add random jitter (±25%)
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
            
        return max(0.1, delay)  # Minimum 100ms delay
    
    def should_open_circuit(self, service_name: str) -> bool:
        """Determine if circuit should be opened"""
        health = self.get_service_health(service_name)
        config = self.get_service_config(service_name)
        
        # Check consecutive failures
        if health.consecutive_failures >= config.max_failures:
            return True
            
        # Check failure rate
        if health.total_requests >= 10:  # Minimum sample size
            failure_rate = health.failure_count / health.total_requests
            if failure_rate >= config.failure_threshold:
                return True
                
        return False
    
    def should_attempt_recovery(self, service_name: str) -> bool:
        """Determine if we should attempt recovery"""
        health = self.get_service_health(service_name)
        config = self.get_service_config(service_name)
        
        if health.state != CircuitState.OPEN:
            return False
            
        time_since_failure = time.time() - health.last_failure_time
        return time_since_failure >= config.recovery_timeout
    
    def record_success(self, service_name: str):
        """Record successful operation"""
        health = self.get_service_health(service_name)
        health.success_count += 1
        health.total_requests += 1
        health.consecutive_failures = 0
        health.last_success_time = time.time()
        
        # Recovery logic
        if health.state == CircuitState.HALF_OPEN:
            health.state = CircuitState.CLOSED
            health.recovery_attempts = 0
            logger.info(f"Circuit breaker for {service_name} restored to CLOSED")
    
    def record_failure(self, service_name: str, error: Exception):
        """Record failed operation"""
        health = self.get_service_health(service_name)
        health.failure_count += 1
        health.total_requests += 1
        health.consecutive_failures += 1
        health.last_failure_time = time.time()
        
        # Check if circuit should be opened
        if self.should_open_circuit(service_name):
            health.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker OPENED for {service_name}: {error}")
        
        logger.error(f"Service {service_name} failed: {error}")
    
    def get_next_available_service(self, current_service: str) -> Optional[str]:
        """Get next available service in failover chain"""
        try:
            current_index = self.failover_chain.index(current_service.lower())
            
            # Try services after current one
            for i in range(current_index + 1, len(self.failover_chain)):
                service = self.failover_chain[i]
                if service == "cache":
                    return "cache"  # Cache is always available
                    
                health = self.get_service_health(service)
                if health.state != CircuitState.OPEN:
                    return service
                    
        except ValueError:
            # Current service not in chain, start from beginning
            pass
        
        # If no services available, return cache
        return "cache"
    
    @asynccontextmanager
    async def execute_with_circuit_breaker(self, service_name: str, operation: Callable):
        """Execute operation with circuit breaker protection"""
        service_key = service_name.lower()
        health = self.get_service_health(service_key)
        config = self.get_service_config(service_key)
        
        # Check circuit state
        if health.state == CircuitState.OPEN:
            if self.should_attempt_recovery(service_key):
                health.state = CircuitState.HALF_OPEN
                health.recovery_attempts += 1
                logger.info(f"Circuit breaker for {service_key} moved to HALF_OPEN")
            else:
                raise Exception(f"Circuit breaker OPEN for {service_key}")
        
        # Execute with timeout and retries
        last_exception = None
        
        for attempt in range(config.max_retries + 1):
            try:
                # Add delay for retries
                if attempt > 0:
                    delay = self.calculate_delay(attempt - 1, config)
                    await asyncio.sleep(delay)
                    logger.info(f"Retrying {service_key} (attempt {attempt + 1}/{config.max_retries + 1}) after {delay:.2f}s")
                
                # Execute with timeout
                result = await asyncio.wait_for(operation(), timeout=config.timeout)
                
                # Record success
                self.record_success(service_key)
                yield result
                return
                
            except asyncio.TimeoutError as e:
                last_exception = e
                logger.warning(f"Timeout for {service_key} (attempt {attempt + 1}): {config.timeout}s")
                
            except Exception as e:
                last_exception = e
                logger.error(f"Error in {service_key} (attempt {attempt + 1}): {e}")
        
        # All retries failed
        self.record_failure(service_key, last_exception)
        raise last_exception
    
    async def execute_with_failover(self, service_name: str, operation_factory: Callable[[str], Callable]):
        """Execute operation with automatic failover"""
        current_service = service_name.lower()
        attempt_count = 0
        max_attempts = len(self.failover_chain)
        
        while attempt_count < max_attempts:
            try:
                if current_service == "cache":
                    # Cache operation (always succeeds)
                    operation = operation_factory("cache")
                    result = await operation()
                    logger.info(f"Served from cache after failover chain")
                    return result
                
                # Try current service
                operation = operation_factory(current_service)
                async with self.execute_with_circuit_breaker(current_service, operation) as result:
                    if attempt_count > 0:
                        logger.info(f"Successful failover to {current_service}")
                    return result
                    
            except Exception as e:
                logger.warning(f"Service {current_service} failed in failover: {e}")
                
                # Get next service
                next_service = self.get_next_available_service(current_service)
                if next_service == current_service:
                    # No more services available
                    break
                    
                current_service = next_service
                attempt_count += 1
        
        # All services failed
        raise Exception(f"All services in failover chain failed for original service: {service_name}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        health_report = {
            "timestamp": time.time(),
            "services": {},
            "overall_status": "healthy"
        }
        
        unhealthy_services = 0
        
        for service_name, health in self.health.items():
            service_report = {
                "state": health.state.value,
                "failure_count": health.failure_count,
                "success_count": health.success_count,
                "total_requests": health.total_requests,
                "consecutive_failures": health.consecutive_failures,
                "success_rate": health.success_count / max(health.total_requests, 1),
                "last_failure_time": health.last_failure_time,
                "last_success_time": health.last_success_time
            }
            
            if health.state == CircuitState.OPEN:
                unhealthy_services += 1
                
            health_report["services"][service_name] = service_report
        
        # Overall system status
        if unhealthy_services == 0:
            health_report["overall_status"] = "healthy"
        elif unhealthy_services < len(self.services):
            health_report["overall_status"] = "degraded"
        else:
            health_report["overall_status"] = "critical"
        
        return health_report
    
    def reset_service_health(self, service_name: str):
        """Reset health statistics for a service"""
        service_key = service_name.lower()
        if service_key in self.health:
            self.health[service_key] = ServiceHealth()
            logger.info(f"Reset health statistics for {service_key}")
    
    def force_circuit_state(self, service_name: str, state: CircuitState):
        """Force circuit breaker to specific state (for testing/admin)"""
        service_key = service_name.lower()
        health = self.get_service_health(service_key)
        health.state = state
        logger.info(f"Forced circuit breaker for {service_key} to {state.value}")

# Global circuit breaker instance
circuit_breaker = EnhancedCircuitBreaker()

# Convenience functions
async def execute_with_circuit_breaker(service_name: str, operation: Callable):
    """Execute operation with circuit breaker protection"""
    async with circuit_breaker.execute_with_circuit_breaker(service_name, operation) as result:
        return result

async def execute_with_failover(service_name: str, operation_factory: Callable[[str], Callable]):
    """Execute operation with automatic failover"""
    return await circuit_breaker.execute_with_failover(service_name, operation_factory)

def get_system_health() -> Dict[str, Any]:
    """Get system health status"""
    return circuit_breaker.get_system_health()

def reset_service_health(service_name: str):
    """Reset service health"""
    circuit_breaker.reset_service_health(service_name)

def force_circuit_state(service_name: str, state: CircuitState):
    """Force circuit state"""
    circuit_breaker.force_circuit_state(service_name, state)

# Example usage in AI services
async def example_ai_call(service_name: str, prompt: str):
    """Example of how to use circuit breaker in AI services"""
    
    def create_ai_operation(actual_service: str):
        async def ai_operation():
            # This would be replaced with actual AI service call
            if actual_service == "gpt":
                # GPT API call
                return f"GPT response to: {prompt}"
            elif actual_service == "claude":
                # Claude API call
                return f"Claude response to: {prompt}"
            elif actual_service == "gemini":
                # Gemini API call
                return f"Gemini response to: {prompt}"
            elif actual_service == "local":
                # Local AI call
                return f"Local AI response to: {prompt}"
            elif actual_service == "cache":
                # Cache lookup
                return f"Cached response to: {prompt}"
            else:
                raise Exception(f"Unknown service: {actual_service}")
        
        return ai_operation
    
    # Execute with automatic failover
    try:
        result = await execute_with_failover(service_name, create_ai_operation)
        return result
    except Exception as e:
        logger.error(f"All AI services failed: {e}")
        return f"Error: Unable to process request - {e}"

if __name__ == "__main__":
    # Test the circuit breaker
    async def test_circuit_breaker():
        result = await example_ai_call("gpt", "Hello, world!")
        print(f"Result: {result}")
        
        health = get_system_health()
        print(f"System health: {json.dumps(health, indent=2)}")
    
    # Run test
    asyncio.run(test_circuit_breaker())
