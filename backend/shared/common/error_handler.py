# backend/services/shared/common/error_handler.py
"""
Global Error Handler - Week 1 Shared Utility
Centralized error handling across all NAVA services
"""

import logging
import traceback
import time
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    TIMEOUT = "timeout"
    API_ERROR = "api_error"
    NETWORK_ERROR = "network_error"
    VALIDATION_ERROR = "validation_error"
    SYSTEM_ERROR = "system_error"
    USER_ERROR = "user_error"

@dataclass
class ErrorInfo:
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    service: str
    timestamp: float
    stack_trace: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class GlobalErrorHandler:
    """Centralized error handling for all NAVA services"""
    
    def __init__(self):
        self.error_history = []
        self.error_counts = {}
        self.circuit_breaker_thresholds = {
            ErrorSeverity.CRITICAL: 3,
            ErrorSeverity.HIGH: 5,
            ErrorSeverity.MEDIUM: 10,
            ErrorSeverity.LOW: 20
        }
    
    def handle_error(
        self, 
        error: Exception, 
        service: str,
        context: Dict[str, Any] = None,
        category: ErrorCategory = ErrorCategory.SYSTEM_ERROR
    ) -> ErrorInfo:
        """Handle and categorize errors"""
        
        error_id = f"{service}_{int(time.time())}_{id(error)}"
        severity = self._determine_severity(error, category)
        
        error_info = ErrorInfo(
            error_id=error_id,
            category=category,
            severity=severity,
            message=str(error),
            service=service,
            timestamp=time.time(),
            stack_trace=traceback.format_exc(),
            context=context
        )
        
        # Record error
        self._record_error(error_info)
        
        # Log appropriately
        self._log_error(error_info)
        
        # Check if circuit breaker should trigger
        self._check_circuit_breaker(service, severity)
        
        return error_info
    
    def _determine_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Determine error severity based on type and category"""
        
        # Critical errors
        if isinstance(error, (SystemExit, KeyboardInterrupt, MemoryError)):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if category == ErrorCategory.TIMEOUT and "circuit_breaker" in str(error).lower():
            return ErrorSeverity.HIGH
        
        if isinstance(error, (ConnectionError, OSError)) or category == ErrorCategory.NETWORK_ERROR:
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if category in [ErrorCategory.API_ERROR, ErrorCategory.TIMEOUT]:
            return ErrorSeverity.MEDIUM
        
        # Low severity errors
        if category in [ErrorCategory.VALIDATION_ERROR, ErrorCategory.USER_ERROR]:
            return ErrorSeverity.LOW
        
        return ErrorSeverity.MEDIUM  # Default
    
    def _record_error(self, error_info: ErrorInfo):
        """Record error in history and counts"""
        
        self.error_history.append(error_info)
        
        # Keep only recent errors (last 1000)
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
        
        # Update error counts
        service_key = error_info.service
        severity_key = f"{service_key}_{error_info.severity.value}"
        
        self.error_counts[service_key] = self.error_counts.get(service_key, 0) + 1
        self.error_counts[severity_key] = self.error_counts.get(severity_key, 0) + 1
    
    def _log_error(self, error_info: ErrorInfo):
        """Log error with appropriate level"""
        
        log_message = f"[{error_info.service}] {error_info.category.value}: {error_info.message}"
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message, extra={"error_info": error_info})
        elif error_info.severity == ErrorSeverity.HIGH:
            logger.error(log_message, extra={"error_info": error_info})
        elif error_info.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message, extra={"error_info": error_info})
        else:
            logger.info(log_message, extra={"error_info": error_info})
    
    def _check_circuit_breaker(self, service: str, severity: ErrorSeverity):
        """Check if circuit breaker should trigger for service"""
        
        severity_key = f"{service}_{severity.value}"
        error_count = self.error_counts.get(severity_key, 0)
        threshold = self.circuit_breaker_thresholds[severity]
        
        if error_count >= threshold:
            logger.error(f"Circuit breaker threshold reached for {service}: {error_count} {severity.value} errors")
            # Trigger circuit breaker (integrate with circuit breaker system)
            self._trigger_circuit_breaker(service, severity)
    
    def _trigger_circuit_breaker(self, service: str, severity: ErrorSeverity):
        """Trigger circuit breaker for service"""
        try:
            # Import and trigger appropriate circuit breaker
            if service == "gpt":
                from .circuit_breaker import gpt_timeout_handler
                gpt_timeout_handler.failure_count += 1
            elif service == "claude":
                from .circuit_breaker import claude_timeout_handler
                claude_timeout_handler.failure_count += 1
            elif service == "gemini":
                from .circuit_breaker import gemini_timeout_handler
                gemini_timeout_handler.failure_count += 1
        except ImportError:
            logger.warning(f"Could not trigger circuit breaker for {service}")
    
    def get_error_summary(self, service: str = None) -> Dict[str, Any]:
        """Get error summary for service or all services"""
        
        if service:
            service_errors = [e for e in self.error_history if e.service == service]
            error_count = self.error_counts.get(service, 0)
        else:
            service_errors = self.error_history
            error_count = sum(self.error_counts.values())
        
        # Group by category and severity
        by_category = {}
        by_severity = {}
        
        for error in service_errors:
            by_category[error.category.value] = by_category.get(error.category.value, 0) + 1
            by_severity[error.severity.value] = by_severity.get(error.severity.value, 0) + 1
        
        return {
            "service": service or "all",
            "total_errors": error_count,
            "recent_errors": len(service_errors),
            "by_category": by_category,
            "by_severity": by_severity,
            "last_error_time": service_errors[-1].timestamp if service_errors else None
        }

# Global instance
global_error_handler = GlobalErrorHandler()

# Convenience functions
def handle_error(error: Exception, service: str, context: Dict[str, Any] = None, category: ErrorCategory = ErrorCategory.SYSTEM_ERROR) -> ErrorInfo:
    """Handle error globally"""
    return global_error_handler.handle_error(error, service, context, category)

def get_error_summary(service: str = None) -> Dict[str, Any]:
    """Get error summary"""
    return global_error_handler.get_error_summary(service)

# ---
# backend/services/shared/common/retry_logic.py
"""
Intelligent Retry Logic - Week 1 Shared Utility
Smart retry strategies for all NAVA services
"""

import asyncio
import random
import time
from typing import Callable, Any, Dict, Optional, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class RetryStrategy(Enum):
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIBONACCI = "fibonacci"
    CUSTOM = "custom"

class RetryManager:
    """Intelligent retry management with multiple strategies"""
    
    def __init__(self):
        self.retry_history = {}
        self.default_config = {
            "max_retries": 3,
            "base_delay": 1.0,
            "max_delay": 30.0,
            "jitter": True,
            "backoff_multiplier": 2.0
        }
    
    async def retry_with_strategy(
        self,
        func: Callable,
        *args,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        jitter: bool = True,
        backoff_multiplier: float = 2.0,
        retry_on: List[Exception] = None,
        **kwargs
    ) -> Any:
        """Execute function with intelligent retry logic"""
        
        retry_on = retry_on or [Exception]
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                
                # Record successful retry
                if attempt > 0:
                    self._record_retry_success(func.__name__, attempt)
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if we should retry this exception
                if not any(isinstance(e, exc_type) for exc_type in retry_on):
                    raise e
                
                if attempt >= max_retries:
                    self._record_retry_failure(func.__name__, attempt)
                    raise e
                
                # Calculate delay
                delay = self._calculate_delay(
                    attempt, strategy, base_delay, max_delay, 
                    backoff_multiplier, jitter
                )
                
                logger.warning(f"Retry {attempt + 1}/{max_retries} for {func.__name__} after {delay:.2f}s delay. Error: {str(e)}")
                
                await asyncio.sleep(delay)
        
        # Should never reach here, but just in case
        raise last_exception
    
    def _calculate_delay(
        self, 
        attempt: int, 
        strategy: RetryStrategy, 
        base_delay: float, 
        max_delay: float,
        backoff_multiplier: float, 
        jitter: bool
    ) -> float:
        """Calculate retry delay based on strategy"""
        
        if strategy == RetryStrategy.EXPONENTIAL:
            delay = base_delay * (backoff_multiplier ** attempt)
        elif strategy == RetryStrategy.LINEAR:
            delay = base_delay * (attempt + 1)
        elif strategy == RetryStrategy.FIBONACCI:
            delay = base_delay * self._fibonacci(attempt + 1)
        else:  # CUSTOM or default
            delay = base_delay * (backoff_multiplier ** attempt)
        
        # Apply jitter to avoid thundering herd
        if jitter:
            jitter_amount = delay * 0.1 * random.uniform(-1, 1)
            delay += jitter_amount
        
        return min(delay, max_delay)
    
    def _fibonacci(self, n: int) -> int:
        """Calculate fibonacci number"""
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
    
    def _record_retry_success(self, func_name: str, attempts: int):
        """Record successful retry"""
        if func_name not in self.retry_history:
            self.retry_history[func_name] = {"successes": 0, "failures": 0, "attempts": []}
        
        self.retry_history[func_name]["successes"] += 1
        self.retry_history[func_name]["attempts"].append(attempts)
    
    def _record_retry_failure(self, func_name: str, attempts: int):
        """Record failed retry"""
        if func_name not in self.retry_history:
            self.retry_history[func_name] = {"successes": 0, "failures": 0, "attempts": []}
        
        self.retry_history[func_name]["failures"] += 1

# Global instance
retry_manager = RetryManager()

# Convenience decorator
def retry(
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_on: List[Exception] = None
):
    """Retry decorator for functions"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            return await retry_manager.retry_with_strategy(
                func, *args,
                strategy=strategy,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                retry_on=retry_on,
                **kwargs
            )
        return wrapper
    return decorator

# ---
# backend/services/shared/common/monitoring.py  
"""
System Monitoring Utilities - Week 1 Shared Utility
Real-time monitoring and health tracking
"""

import time
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_percent: float
    network_io: Dict[str, int]
    timestamp: float

@dataclass
class ServiceHealth:
    service_name: str
    status: str
    response_time: float
    error_rate: float
    last_check: float
    details: Dict[str, Any]

class SystemMonitor:
    """Real-time system monitoring"""
    
    def __init__(self):
        self.metrics_history = []
        self.service_health = {}
        self.alert_thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_percent": 90.0,
            "response_time": 5.0,
            "error_rate": 10.0
        }
        self.monitoring_active = False
    
    def get_current_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        
        try:
            # Try to use psutil if available
            try:
                import psutil
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Network I/O
                net_io = psutil.net_io_counters()
                network_io = {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv
                }
                
                metrics = SystemMetrics(
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_used_mb=memory.used / 1024 / 1024,
                    disk_percent=disk.percent,
                    network_io=network_io,
                    timestamp=time.time()
                )
                
            except ImportError:
                # Fallback to basic metrics if psutil not available
                metrics = SystemMetrics(
                    cpu_percent=0.0,
                    memory_percent=0.0,
                    memory_used_mb=0.0,
                    disk_percent=0.0,
                    network_io={},
                    timestamp=time.time()
                )
            
            # Store in history
            self.metrics_history.append(metrics)
            
            # Keep only recent metrics (last 1000)
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return SystemMetrics(0, 0, 0, 0, {}, time.time())
    
    def update_service_health(
        self, 
        service_name: str, 
        status: str, 
        response_time: float = 0.0,
        error_rate: float = 0.0,
        details: Dict[str, Any] = None
    ):
        """Update service health status"""
        
        health = ServiceHealth(
            service_name=service_name,
            status=status,
            response_time=response_time,
            error_rate=error_rate,
            last_check=time.time(),
            details=details or {}
        )
        
        self.service_health[service_name] = health
        
        # Check for alerts
        self._check_service_alerts(health)
    
    def _check_service_alerts(self, health: ServiceHealth):
        """Check if service health triggers any alerts"""
        
        alerts = []
        
        if health.response_time > self.alert_thresholds["response_time"]:
            alerts.append(f"High response time: {health.response_time:.2f}s")
        
        if health.error_rate > self.alert_thresholds["error_rate"]:
            alerts.append(f"High error rate: {health.error_rate:.1f}%")
        
        if health.status not in ["healthy", "operational"]:
            alerts.append(f"Service unhealthy: {health.status}")
        
        if alerts:
            logger.warning(f"Service alerts for {health.service_name}: {', '.join(alerts)}")
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary"""
        
        current_metrics = self.get_current_metrics()
        
        # Calculate averages for recent metrics
        recent_metrics = self.metrics_history[-10:] if len(self.metrics_history) >= 10 else self.metrics_history
        
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0
        
        # Service health summary
        healthy_services = [s for s in self.service_health.values() if s.status in ["healthy", "operational"]]
        total_services = len(self.service_health)
        
        overall_health = "healthy"
        if current_metrics.cpu_percent > 90 or current_metrics.memory_percent > 90:
            overall_health = "critical"
        elif current_metrics.cpu_percent > 80 or current_metrics.memory_percent > 80:
            overall_health = "warning"
        elif total_services > 0 and len(healthy_services) < total_services * 0.8:
            overall_health = "degraded"
        
        return {
            "overall_health": overall_health,
            "system_metrics": {
                "cpu_percent": current_metrics.cpu_percent,
                "memory_percent": current_metrics.memory_percent,
                "memory_used_mb": current_metrics.memory_used_mb,
                "disk_percent": current_metrics.disk_percent,
                "avg_cpu_10min": round(avg_cpu, 2),
                "avg_memory_10min": round(avg_memory, 2)
            },
            "service_health": {
                "total_services": total_services,
                "healthy_services": len(healthy_services),
                "service_details": {name: {
                    "status": health.status,
                    "response_time": health.response_time,
                    "error_rate": health.error_rate,
                    "last_check": health.last_check
                } for name, health in self.service_health.items()}
            },
            "timestamp": current_metrics.timestamp
        }
    
    async def start_monitoring(self, interval: int = 30):
        """Start continuous monitoring"""
        self.monitoring_active = True
        logger.info(f"Starting system monitoring with {interval}s interval")
        
        while self.monitoring_active:
            try:
                self.get_current_metrics()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval)
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        logger.info("Stopping system monitoring")

# Global instance
system_monitor = SystemMonitor()

# Convenience functions
def get_system_health() -> Dict[str, Any]:
    """Get current system health"""
    return system_monitor.get_system_health_summary()

def update_service_health(service_name: str, status: str, response_time: float = 0.0, error_rate: float = 0.0, details: Dict[str, Any] = None):
    """Update service health status"""
    system_monitor.update_service_health(service_name, status, response_time, error_rate, details)

async def start_system_monitoring(interval: int = 30):
    """Start system monitoring"""
    await system_monitor.start_monitoring(interval)

def stop_system_monitoring():
    """Stop system monitoring"""
    system_monitor.stop_monitoring()