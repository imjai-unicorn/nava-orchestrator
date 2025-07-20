"""
Performance Utilities for NAVA System
File: backend/services/shared/common/performance_utils.py
"""

import time
import asyncio
import logging
import psutil
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import functools

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: float
    response_time: float
    cpu_usage: float
    memory_usage: float
    request_count: int
    error_count: int
    success_rate: float
    throughput: float
    service_name: str
    endpoint: str = ""

@dataclass
class SystemMetrics:
    """System-wide metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    network_io: Dict[str, int]
    process_count: int
    load_average: List[float]

class PerformanceMonitor:
    """
    Performance monitoring and metrics collection
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.counters: Dict[str, int] = defaultdict(int)
        self.timers: Dict[str, float] = {}
        self.lock = threading.Lock()
        
    def record_request(self, service_name: str, endpoint: str, response_time: float, success: bool = True):
        """Record a request metric"""
        with self.lock:
            timestamp = time.time()
            
            # Update counters
            key = f"{service_name}:{endpoint}"
            self.counters[f"{key}:total"] += 1
            if success:
                self.counters[f"{key}:success"] += 1
            else:
                self.counters[f"{key}:error"] += 1
            
            # Calculate success rate
            total = self.counters[f"{key}:total"]
            success_count = self.counters[f"{key}:success"]
            success_rate = success_count / total if total > 0 else 0
            
            # Get system metrics
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            
            # Create metric
            metric = PerformanceMetrics(
                timestamp=timestamp,
                response_time=response_time,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                request_count=total,
                error_count=self.counters[f"{key}:error"],
                success_rate=success_rate,
                throughput=self.calculate_throughput(key),
                service_name=service_name,
                endpoint=endpoint
            )
            
            # Store metric
            self.metrics[key].append(metric)
    
    def calculate_throughput(self, key: str) -> float:
        """Calculate requests per second"""
        if key not in self.metrics or len(self.metrics[key]) < 2:
            return 0.0
        
        recent_metrics = list(self.metrics[key])[-60:]  # Last 60 requests
        if len(recent_metrics) < 2:
            return 0.0
        
        time_span = recent_metrics[-1].timestamp - recent_metrics[0].timestamp
        if time_span <= 0:
            return 0.0
        
        return len(recent_metrics) / time_span
    
    def get_metrics(self, service_name: str, endpoint: str = "") -> Dict[str, Any]:
        """Get metrics for a service/endpoint"""
        key = f"{service_name}:{endpoint}"
        
        with self.lock:
            if key not in self.metrics or not self.metrics[key]:
                return {
                    "service": service_name,
                    "endpoint": endpoint,
                    "metrics": {
                        "avg_response_time": 0,
                        "p95_response_time": 0,
                        "p99_response_time": 0,
                        "total_requests": 0,
                        "success_rate": 0,
                        "throughput": 0,
                        "error_count": 0
                    }
                }
            
            recent_metrics = list(self.metrics[key])
            response_times = [m.response_time for m in recent_metrics]
            
            # Calculate percentiles
            response_times.sort()
            count = len(response_times)
            p95_index = int(count * 0.95)
            p99_index = int(count * 0.99)
            
            return {
                "service": service_name,
                "endpoint": endpoint,
                "metrics": {
                    "avg_response_time": sum(response_times) / count,
                    "p95_response_time": response_times[p95_index] if p95_index < count else 0,
                    "p99_response_time": response_times[p99_index] if p99_index < count else 0,
                    "total_requests": self.counters[f"{key}:total"],
                    "success_rate": recent_metrics[-1].success_rate,
                    "throughput": recent_metrics[-1].throughput,
                    "error_count": self.counters[f"{key}:error"],
                    "last_updated": recent_metrics[-1].timestamp
                }
            }
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=psutil.cpu_percent(),
            memory_percent=psutil.virtual_memory().percent,
            disk_usage=psutil.disk_usage('/').percent,
            network_io=dict(psutil.net_io_counters()._asdict()),
            process_count=len(psutil.pids()),
            load_average=list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else [0, 0, 0]
        )
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics"""
        all_metrics = {}
        
        with self.lock:
            for key in self.metrics:
                service_name, endpoint = key.split(':', 1)
                all_metrics[key] = self.get_metrics(service_name, endpoint)
        
        return {
            "timestamp": time.time(),
            "system": self.get_system_metrics().__dict__,
            "services": all_metrics
        }
    
    def reset_metrics(self, service_name: str = None, endpoint: str = None):
        """Reset metrics for service/endpoint or all metrics"""
        with self.lock:
            if service_name:
                key = f"{service_name}:{endpoint or ''}"
                if key in self.metrics:
                    self.metrics[key].clear()
                # Reset counters
                for counter_key in list(self.counters.keys()):
                    if counter_key.startswith(key):
                        self.counters[counter_key] = 0
            else:
                # Reset all metrics
                self.metrics.clear()
                self.counters.clear()

class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, service_name: str, endpoint: str = "", monitor: Optional[PerformanceMonitor] = None):
        self.service_name = service_name
        self.endpoint = endpoint
        self.monitor = monitor or performance_monitor
        self.start_time = None
        self.end_time = None
        self.success = True
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        response_time = self.end_time - self.start_time
        
        # Mark as failure if exception occurred
        if exc_type is not None:
            self.success = False
            
        # Record metric
        self.monitor.record_request(
            self.service_name,
            self.endpoint,
            response_time,
            self.success
        )
        
        return False  # Don't suppress exceptions

def performance_timer(service_name: str, endpoint: str = ""):
    """Decorator for timing function performance"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            with PerformanceTimer(service_name, endpoint):
                return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            with PerformanceTimer(service_name, endpoint):
                return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

class PerformanceOptimizer:
    """
    Performance optimization utilities
    """
    
    def __init__(self):
        self.optimizations = {}
        self.cache = {}
        
    def memoize(self, ttl: int = 300):
        """Memoization decorator with TTL"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Create cache key
                key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
                
                # Check cache
                if key in self.cache:
                    result, timestamp = self.cache[key]
                    if time.time() - timestamp < ttl:
                        return result
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Cache result
                self.cache[key] = (result, time.time())
                
                return result
            return wrapper
        return decorator
    
    def async_memoize(self, ttl: int = 300):
        """Async memoization decorator with TTL"""
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                # Create cache key
                key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
                
                # Check cache
                if key in self.cache:
                    result, timestamp = self.cache[key]
                    if time.time() - timestamp < ttl:
                        return result
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Cache result
                self.cache[key] = (result, time.time())
                
                return result
            return wrapper
        return decorator
    
    def rate_limit(self, max_calls: int, time_window: int = 60):
        """Rate limiting decorator"""
        def decorator(func):
            calls = deque()
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                now = time.time()
                
                # Remove old calls
                while calls and calls[0] < now - time_window:
                    calls.popleft()
                
                # Check rate limit
                if len(calls) >= max_calls:
                    raise Exception(f"Rate limit exceeded: {max_calls} calls per {time_window} seconds")
                
                # Record call
                calls.append(now)
                
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def async_rate_limit(self, max_calls: int, time_window: int = 60):
        """Async rate limiting decorator"""
        def decorator(func):
            calls = deque()
            
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                now = time.time()
                
                # Remove old calls
                while calls and calls[0] < now - time_window:
                    calls.popleft()
                
                # Check rate limit
                if len(calls) >= max_calls:
                    raise Exception(f"Rate limit exceeded: {max_calls} calls per {time_window} seconds")
                
                # Record call
                calls.append(now)
                
                return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    def clear_cache(self):
        """Clear all cached results"""
        self.cache.clear()

class PerformanceAnalyzer:
    """
    Performance analysis and reporting
    """
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        
    def analyze_performance(self, service_name: str, time_window: int = 3600) -> Dict[str, Any]:
        """Analyze performance over time window"""
        cutoff_time = time.time() - time_window
        
        # Get all metrics for service
        all_metrics = {}
        for key in self.monitor.metrics:
            if key.startswith(f"{service_name}:"):
                service, endpoint = key.split(':', 1)
                recent_metrics = [m for m in self.monitor.metrics[key] if m.timestamp > cutoff_time]
                
                if recent_metrics:
                    response_times = [m.response_time for m in recent_metrics]
                    
                    all_metrics[endpoint] = {
                        "count": len(recent_metrics),
                        "avg_response_time": sum(response_times) / len(response_times),
                        "min_response_time": min(response_times),
                        "max_response_time": max(response_times),
                        "success_rate": recent_metrics[-1].success_rate,
                        "throughput": recent_metrics[-1].throughput
                    }
        
        return {
            "service": service_name,
            "analysis_window": time_window,
            "endpoints": all_metrics,
            "recommendations": self.generate_recommendations(all_metrics)
        }
    
    def generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        for endpoint, data in metrics.items():
            if data["avg_response_time"] > 3.0:
                recommendations.append(f"Endpoint '{endpoint}' has high response time ({data['avg_response_time']:.2f}s)")
            
            if data["success_rate"] < 0.95:
                recommendations.append(f"Endpoint '{endpoint}' has low success rate ({data['success_rate']:.2%})")
            
            if data["throughput"] < 1.0:
                recommendations.append(f"Endpoint '{endpoint}' has low throughput ({data['throughput']:.2f} req/s)")
        
        return recommendations
    
    def generate_report(self, service_name: str = None) -> str:
        """Generate performance report"""
        if service_name:
            analysis = self.analyze_performance(service_name)
            return json.dumps(analysis, indent=2)
        else:
            all_metrics = self.monitor.get_all_metrics()
            return json.dumps(all_metrics, indent=2)

# Global instances
performance_monitor = PerformanceMonitor()
performance_optimizer = PerformanceOptimizer()
performance_analyzer = PerformanceAnalyzer(performance_monitor)

# Convenience functions
def record_request(service_name: str, endpoint: str, response_time: float, success: bool = True):
    """Record a request metric"""
    performance_monitor.record_request(service_name, endpoint, response_time, success)

def get_metrics(service_name: str, endpoint: str = "") -> Dict[str, Any]:
    """Get metrics for service/endpoint"""
    return performance_monitor.get_metrics(service_name, endpoint)

def get_system_metrics() -> SystemMetrics:
    """Get current system metrics"""
    return performance_monitor.get_system_metrics()

def get_all_metrics() -> Dict[str, Any]:
    """Get all metrics"""
    return performance_monitor.get_all_metrics()

def analyze_performance(service_name: str, time_window: int = 3600) -> Dict[str, Any]:
    """Analyze performance"""
    return performance_analyzer.analyze_performance(service_name, time_window)

def generate_performance_report(service_name: str = None) -> str:
    """Generate performance report"""
    return performance_analyzer.generate_report(service_name)

# Example usage
if __name__ == "__main__":
    # Test performance monitoring
    @performance_timer("test_service", "test_endpoint")
    async def test_function():
        await asyncio.sleep(0.1)
        return "test result"
    
    async def test_performance():
        # Run test function multiple times
        for i in range(10):
            result = await test_function()
            print(f"Test {i+1}: {result}")
        
        # Get metrics
        metrics = get_metrics("test_service", "test_endpoint")
        print(f"Metrics: {json.dumps(metrics, indent=2)}")
        
        # Generate report
        report = generate_performance_report("test_service")
        print(f"Report: {report}")
    
    # Run test
    asyncio.run(test_performance())
