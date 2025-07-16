# backend/shared/common/circuit_breaker.py

import asyncio
import time
import logging
from typing import Dict, Any, Optional, Callable
from enum import Enum

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"     # Normal operation
    OPEN = "open"         # Circuit breaker triggered
    HALF_OPEN = "half_open"  # Testing if service recovered

class ServiceHealth:
    """Track health metrics for a service"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.last_success_time = 0
        self.response_times = []
        self.state = CircuitState.CLOSED
        
    def record_success(self, response_time: float):
        """Record successful request"""
        self.success_count += 1
        self.last_success_time = time.time()
        self.response_times.append(response_time)
        
        # Keep only recent response times (last 100)
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]
            
        # Reset failure count on success
        if self.state == CircuitState.HALF_OPEN:
            self.failure_count = 0
            self.state = CircuitState.CLOSED
            logger.info(f"✅ Circuit breaker for {self.service_name} closed - service recovered")
    
    def record_failure(self):
        """Record failed request"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        logger.warning(f"❌ Service failure recorded for {self.service_name} (count: {self.failure_count})")
    
    @property
    def average_response_time(self) -> float:
        """Get average response time"""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
    
    @property
    def failure_rate(self) -> float:
        """Get failure rate (0.0 to 1.0)"""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.failure_count / total

class EnhancedCircuitBreaker:
    """Enhanced circuit breaker with intelligent failover"""
    
    def __init__(self):
        self.timeout_settings = {
            'gpt': {'timeout': 2.5, 'retry': 2, 'backoff': 1.2},     # ← ลดจาก 15→5
            'claude': {'timeout': 3, 'retry': 2, 'backoff': 1.2},  # ← ลดจาก 20→7
            'gemini': {'timeout': 3, 'retry': 2, 'backoff': 1.2},  # ← ลดจาก 18→6
            'local': {'timeout': 2, 'retry': 2, 'backoff': 1}      # ← ลดจาก 5→3
        }
        
        self.failure_thresholds = {
            'consecutive_failures': 5,
            'failure_rate_percentage': 20,
            'circuit_open_duration': 60  # seconds
        }
        
        self.fallback_chain = ['gpt', 'claude', 'gemini', 'local', 'cache']
        
        self.concurrent_limits = {
            'gpt': 15,      # จำกัด GPT ไม่เกิน 10 concurrent requests
            'claude': 20,   # Claude รับได้มากกว่า
            'gemini': 25,   # Gemini รับได้มากที่สุด
            'local': 15     # Local AI รับได้มากที่สุด
        }
        
        self.active_requests = {
            'gpt': 0,
            'claude': 0, 
            'gemini': 0,
            'local': 0
        }
        
        self.request_lock = asyncio.Lock()
        self.service_health: Dict[str, ServiceHealth] = {}        
        
        # Initialize health tracking
        for service in self.timeout_settings.keys():
            self.service_health[service] = ServiceHealth(service)
            
        logger.info("🔧 Enhanced Circuit Breaker initialized")
    
    def _get_service_health(self, service_name: str) -> ServiceHealth:
        """Get or create service health tracker"""
        if service_name not in self.service_health:
            self.service_health[service_name] = ServiceHealth(service_name)
        return self.service_health[service_name]
    
    def _should_circuit_open(self, health: ServiceHealth) -> bool:
        """Determine if circuit should open"""
        
        # Check consecutive failures
        if health.failure_count >= self.failure_thresholds['consecutive_failures']:
            return True
            
        # Check failure rate
        if health.failure_rate >= (self.failure_thresholds['failure_rate_percentage'] / 100.0):
            return True
            
        return False
    
    def _should_circuit_close(self, health: ServiceHealth) -> bool:
        """Determine if circuit should close (move from OPEN to HALF_OPEN)"""
        
        if health.state != CircuitState.OPEN:
            return False
            
        # Check if enough time has passed since circuit opened
        time_since_failure = time.time() - health.last_failure_time
        return time_since_failure >= self.failure_thresholds['circuit_open_duration']
    
    async def call_with_timeout(self, service_name: str, request_func: Callable, *args, **kwargs):
        """Call service with circuit breaker protection"""
        
        # 🆕 NEW: Check concurrent limits and route if necessary
        async with self.request_lock:
            if service_name in self.concurrent_limits:
                if self.active_requests[service_name] >= self.concurrent_limits[service_name]:
                    # Route to alternative service
                    alternatives = ['claude', 'gemini'] if service_name == 'gpt' else ['gpt', 'gemini']
                    original_service = service_name
                    
                    for alt in alternatives:
                        if alt in self.active_requests and self.active_requests[alt] < self.concurrent_limits[alt]:
                            service_name = alt
                            logger.info(f"🔄 Routing from {original_service} to {alt} (concurrent limit)")
                            break
                    else:
                        # All services busy, brief wait
                        logger.warning(f"⚠️ All services busy, waiting briefly...")
                        await asyncio.sleep(0.1)
                
                self.active_requests[service_name] += 1        
               
        health = self._get_service_health(service_name)
        
        # Check circuit state
        if health.state == CircuitState.OPEN:
            if self._should_circuit_close(health):
                health.state = CircuitState.HALF_OPEN
                logger.info(f"🔄 Circuit breaker for {service_name} half-open - testing service")
            else:
                logger.warning(f"⚡ Circuit breaker OPEN for {service_name} - using fallback")
                raise Exception(f"Circuit breaker open for {service_name}")
        
        settings = self.timeout_settings.get(service_name, self.timeout_settings['gpt'])
        
        for attempt in range(settings['retry']):
            start_time = time.time()
            
            try:
                # Make the actual API call with timeout
                response = await asyncio.wait_for(
                    request_func(*args, **kwargs),
                    timeout=settings['timeout']
                )
                
                # Record success
                response_time = time.time() - start_time
                health.record_success(response_time)
                
                logger.info(f"✅ {service_name} call successful ({response_time:.2f}s)")
                return response
                
            except asyncio.TimeoutError:
                logger.warning(f"⏰ {service_name} timeout on attempt {attempt + 1}")
                health.record_failure()
                
                if attempt < settings['retry'] - 1:
                    # Wait before retry with exponential backoff + jitter
                    delay = settings['backoff'] ** attempt + (time.time() % 1)  # Add jitter
                    await asyncio.sleep(delay)
                    continue
                else:
                    # Check if circuit should open
                    if self._should_circuit_open(health):
                        health.state = CircuitState.OPEN
                        logger.error(f"⚡ Circuit breaker OPENED for {service_name}")
                    
                    raise Exception(f"Service {service_name} timeout after {settings['retry']} attempts")
                    
            except Exception as e:
                logger.error(f"❌ {service_name} call failed: {e}")
                health.record_failure()
                
                if attempt < settings['retry'] - 1:
                    delay = settings['backoff'] ** attempt
                    await asyncio.sleep(delay)
                    continue
                else:
                    # Check if circuit should open
                    if self._should_circuit_open(health):
                        health.state = CircuitState.OPEN
                        logger.error(f"⚡ Circuit breaker OPENED for {service_name}")
                    
                    raise
            finally:
                # 🆕 NEW: Cleanup active request count
                if service_name in self.active_requests:
                    async with self.request_lock:
                        self.active_requests[service_name] = max(0, self.active_requests[service_name] - 1)
    
    async def call_with_fallback(self, request_func: Callable, fallback_chain: Optional[list] = None, *args, **kwargs):
        """Call service with automatic fallback chain"""
        
        chain = fallback_chain or self.fallback_chain
        last_exception = None
        
        for service_name in chain:
            if service_name == 'cache':
                # Handle cache fallback (would need cache manager)
                logger.info("💾 Attempting cache fallback")
                continue
                
            try:
                logger.info(f"🔄 Trying {service_name}")
                response = await self.call_with_timeout(service_name, request_func, *args, **kwargs)
                return {
                    'response': response,
                    'service_used': service_name,
                    'fallback_chain': chain[:chain.index(service_name) + 1]
                }
                
            except Exception as e:
                last_exception = e
                logger.warning(f"⚠️ {service_name} failed, trying next in chain")
                continue
        
        # All services in chain failed
        logger.error(f"💥 All services in fallback chain failed")
        raise Exception(f"All services failed. Last error: {last_exception}")
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all services"""
        
        status = {}
        
        for service_name, health in self.service_health.items():
            status[service_name] = {
                'state': health.state.value,
                'failure_count': health.failure_count,
                'success_count': health.success_count,
                'failure_rate': health.failure_rate,
                'average_response_time': health.average_response_time,
                'last_success': health.last_success_time,
                'last_failure': health.last_failure_time
            }
        
        return status
    
    def reset_service(self, service_name: str):
        """Reset circuit breaker for a service"""
        if service_name in self.service_health:
            health = self.service_health[service_name]
            health.failure_count = 0
            health.state = CircuitState.CLOSED
            logger.info(f"🔄 Circuit breaker reset for {service_name}")
    
    def reset_all_services(self):
        """Reset all circuit breakers"""
        for service_name in self.service_health:
            self.reset_service(service_name)
        logger.info("🔄 All circuit breakers reset")
    
    def get_concurrent_status(self) -> Dict[str, Any]:
        """Get current concurrent request status"""
        return {
            'active_requests': dict(self.active_requests),
            'concurrent_limits': dict(self.concurrent_limits),
            'utilization': {
                service: f"{self.active_requests.get(service, 0)}/{limit}" 
                for service, limit in self.concurrent_limits.items()
            }
        }

# Global circuit breaker instance
circuit_breaker = EnhancedCircuitBreaker()

# Export both for compatibility
enhanced_circuit_breaker = circuit_breaker

# Convenience functions for easy integration
async def call_ai_with_protection(service: str, call_func: Callable, *args, **kwargs):
    """Convenience function for AI calls with circuit breaker protection"""
    return await circuit_breaker.call_with_circuit_breaker(service, call_func, *args, **kwargs)

def get_service_health(service: str = None):
    """Convenience function to get service health"""
    return circuit_breaker.get_service_status()

def reset_service_circuit(service: str):
    """Convenience function to reset service circuit"""
    circuit_breaker.reset_service(service)

def get_concurrent_stats():
    """Convenience function to get concurrent request stats"""
    return circuit_breaker.get_concurrent_status()