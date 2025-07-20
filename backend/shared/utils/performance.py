# backend/shared/utils/performance.py
"""
Performance Monitoring and Optimization Utilities for NAVA
Response time tracking, bottleneck detection, and performance optimization
"""

import time
import logging
import statistics
import threading
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from functools import wraps
import psutil

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Individual performance measurement"""
    name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TimingResult:
    """Result of timing measurement"""
    operation: str
    duration_ms: float
    start_time: datetime
    end_time: datetime
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class PerformanceTracker:
    """
    Performance tracking and monitoring
    Tracks response times, throughput, and system performance
    """
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        
        # Timing data
        self.operation_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.operation_counts: Dict[str, int] = defaultdict(int)
        self.operation_errors: Dict[str, int] = defaultdict(int)
        
        # System metrics
        self.system_metrics: deque = deque(maxlen=max_history)
        
        # Performance baselines
        self.baselines: Dict[str, float] = {}
        self.thresholds: Dict[str, float] = {
            'response_time_warning': 3000,  # 3 seconds
            'response_time_critical': 5000,  # 5 seconds
            'cpu_warning': 70.0,
            'memory_warning': 80.0,
            'error_rate_warning': 5.0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("✅ Performance Tracker initialized")
    
    def record_timing(self, operation: str, duration_ms: float, 
                     success: bool = True, metadata: Dict[str, Any] = None):
        """Record timing for an operation"""
        with self._lock:
            timing = TimingResult(
                operation=operation,
                duration_ms=duration_ms,
                start_time=datetime.now() - timedelta(milliseconds=duration_ms),
                end_time=datetime.now(),
                success=success,
                metadata=metadata or {}
            )
            
            self.operation_times[operation].append(timing)
            self.operation_counts[operation] += 1
            
            if not success:
                self.operation_errors[operation] += 1
            
            # Check for performance issues
            self._check_performance_thresholds(operation, duration_ms)
            
        logger.debug(f"📊 Recorded timing: {operation} = {duration_ms:.2f}ms")
    
    def record_system_metrics(self):
        """Record current system metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            metric = PerformanceMetric(
                name="system_metrics",
                value=0,  # Composite metric
                unit="composite",
                metadata={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            with self._lock:
                self.system_metrics.append(metric)
            
            # Check system thresholds
            if cpu_percent > self.thresholds['cpu_warning']:
                logger.warning(f"⚠️ High CPU usage: {cpu_percent:.1f}%")
            
            if memory.percent > self.thresholds['memory_warning']:
                logger.warning(f"⚠️ High memory usage: {memory.percent:.1f}%")
                
        except Exception as e:
            logger.error(f"❌ Failed to record system metrics: {e}")
    
    def get_operation_stats(self, operation: str) -> Dict[str, Any]:
        """Get statistics for a specific operation"""
        with self._lock:
            if operation not in self.operation_times:
                return {"error": "Operation not found"}
            
            timings = [t.duration_ms for t in self.operation_times[operation]]
            
            if not timings:
                return {"error": "No timing data available"}
            
            return {
                "operation": operation,
                "count": self.operation_counts[operation],
                "errors": self.operation_errors[operation],
                "error_rate_percent": (self.operation_errors[operation] / self.operation_counts[operation]) * 100,
                "response_times": {
                    "min_ms": min(timings),
                    "max_ms": max(timings),
                    "avg_ms": statistics.mean(timings),
                    "median_ms": statistics.median(timings),
                    "p95_ms": self._calculate_percentile(timings, 95),
                    "p99_ms": self._calculate_percentile(timings, 99)
                },
                "recent_timings": timings[-10:],  # Last 10 measurements
                "baseline_ms": self.baselines.get(operation),
                "performance_trend": self._calculate_trend(timings)
            }
    
    def get_all_operation_stats(self) -> Dict[str, Any]:
        """Get statistics for all operations"""
        stats = {}
        for operation in self.operation_times.keys():
            stats[operation] = self.get_operation_stats(operation)
        return stats
    
    def get_system_performance_summary(self) -> Dict[str, Any]:
        """Get overall system performance summary"""
        with self._lock:
            if not self.system_metrics:
                self.record_system_metrics()
            
            # Get recent system metrics
            recent_metrics = list(self.system_metrics)[-10:]
            
            if recent_metrics:
                latest = recent_metrics[-1]
                cpu_values = [m.metadata.get('cpu_percent', 0) for m in recent_metrics]
                memory_values = [m.metadata.get('memory_percent', 0) for m in recent_metrics]
                
                return {
                    "timestamp": datetime.now().isoformat(),
                    "current_metrics": {
                        "cpu_percent": latest.metadata.get('cpu_percent', 0),
                        "memory_percent": latest.metadata.get('memory_percent', 0),
                        "memory_available_gb": latest.metadata.get('memory_available_gb', 0)
                    },
                    "averages_last_10": {
                        "cpu_percent": statistics.mean(cpu_values) if cpu_values else 0,
                        "memory_percent": statistics.mean(memory_values) if memory_values else 0
                    },
                    "total_operations": sum(self.operation_counts.values()),
                    "total_errors": sum(self.operation_errors.values()),
                    "overall_error_rate": self._calculate_overall_error_rate(),
                    "performance_status": self._assess_performance_status()
                }
        
        return {"error": "No system metrics available"}
    
    def set_baseline(self, operation: str, baseline_ms: float):
        """Set performance baseline for operation"""
        self.baselines[operation] = baseline_ms
        logger.info(f"📊 Set baseline for {operation}: {baseline_ms:.2f}ms")
    
    def auto_calculate_baselines(self):
        """Automatically calculate baselines from current data"""
        with self._lock:
            for operation, timings_data in self.operation_times.items():
                if len(timings_data) >= 10:  # Need at least 10 measurements
                    timings = [t.duration_ms for t in timings_data]
                    # Use median as baseline (more stable than average)
                    baseline = statistics.median(timings)
                    self.baselines[operation] = baseline
                    logger.info(f"📊 Auto-calculated baseline for {operation}: {baseline:.2f}ms")
    
    def detect_performance_degradation(self) -> List[Dict[str, Any]]:
        """Detect operations with performance degradation"""
        issues = []
        
        with self._lock:
            for operation, timings_data in self.operation_times.items():
                if len(timings_data) < 5:  # Need at least 5 measurements
                    continue
                
                recent_timings = [t.duration_ms for t in list(timings_data)[-5:]]
                recent_avg = statistics.mean(recent_timings)
                
                baseline = self.baselines.get(operation)
                if baseline and recent_avg > baseline * 1.5:  # 50% degradation
                    issues.append({
                        "operation": operation,
                        "baseline_ms": baseline,
                        "recent_avg_ms": recent_avg,
                        "degradation_percent": ((recent_avg - baseline) / baseline) * 100,
                        "severity": "critical" if recent_avg > baseline * 2 else "warning"
                    })
        
        return issues
    
    def get_slowest_operations(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get slowest operations by average response time"""
        operation_avgs = []
        
        with self._lock:
            for operation, timings_data in self.operation_times.items():
                if timings_data:
                    timings = [t.duration_ms for t in timings_data]
                    avg_time = statistics.mean(timings)
                    operation_avgs.append({
                        "operation": operation,
                        "avg_response_time_ms": avg_time,
                        "count": len(timings),
                        "error_rate": (self.operation_errors[operation] / self.operation_counts[operation]) * 100
                    })
        
        # Sort by average response time (descending)
        operation_avgs.sort(key=lambda x: x["avg_response_time_ms"], reverse=True)
        
        return operation_avgs[:limit]
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile value"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = (percentile / 100) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower = sorted_values[int(index)]
            upper = sorted_values[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def _calculate_trend(self, timings: List[float]) -> str:
        """Calculate performance trend"""
        if len(timings) < 5:
            return "insufficient_data"
        
        # Compare first half vs second half
        mid = len(timings) // 2
        first_half_avg = statistics.mean(timings[:mid])
        second_half_avg = statistics.mean(timings[mid:])
        
        diff_percent = ((second_half_avg - first_half_avg) / first_half_avg) * 100
        
        if diff_percent > 10:
            return "degrading"
        elif diff_percent < -10:
            return "improving"
        else:
            return "stable"
    
    def _calculate_overall_error_rate(self) -> float:
        """Calculate overall error rate across all operations"""
        total_operations = sum(self.operation_counts.values())
        total_errors = sum(self.operation_errors.values())
        
        if total_operations == 0:
            return 0.0
        
        return (total_errors / total_operations) * 100
    
    def _assess_performance_status(self) -> str:
        """Assess overall performance status"""
        error_rate = self._calculate_overall_error_rate()
        
        # Check for performance issues
        if error_rate > self.thresholds['error_rate_warning']:
            return "critical"
        
        # Check recent system metrics
        if self.system_metrics:
            latest = self.system_metrics[-1]
            cpu = latest.metadata.get('cpu_percent', 0)
            memory = latest.metadata.get('memory_percent', 0)
            
            if cpu > self.thresholds['cpu_warning'] or memory > self.thresholds['memory_warning']:
                return "warning"
        
        # Check for slow operations
        slowest = self.get_slowest_operations(1)
        if slowest and slowest[0]['avg_response_time_ms'] > self.thresholds['response_time_warning']:
            return "warning"
        
        return "healthy"
    
    def _check_performance_thresholds(self, operation: str, duration_ms: float):
        """Check if operation exceeds performance thresholds"""
        if duration_ms > self.thresholds['response_time_critical']:
            logger.error(f"🚨 CRITICAL: {operation} took {duration_ms:.2f}ms (threshold: {self.thresholds['response_time_critical']}ms)")
        elif duration_ms > self.thresholds['response_time_warning']:
            logger.warning(f"⚠️ SLOW: {operation} took {duration_ms:.2f}ms (threshold: {self.thresholds['response_time_warning']}ms)")

class PerformanceOptimizer:
    """Performance optimization utilities"""
    
    @staticmethod
    def optimize_memory_usage():
        """Optimize memory usage"""
        import gc
        
        # Force garbage collection
        collected = gc.collect()
        
        # Get memory info
        memory = psutil.virtual_memory()
        
        logger.info(f"🧹 Memory optimization: collected {collected} objects")
        logger.info(f"💾 Memory usage: {memory.percent:.1f}% ({memory.available / (1024**3):.1f}GB available)")
        
        return {
            "objects_collected": collected,
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3)
        }
    
    @staticmethod
    def analyze_cpu_usage(duration_seconds: int = 5) -> Dict[str, Any]:
        """Analyze CPU usage over time"""
        logger.info(f"📊 Analyzing CPU usage for {duration_seconds} seconds...")
        
        # Sample CPU usage
        samples = []
        interval = 0.5
        iterations = int(duration_seconds / interval)
        
        for _ in range(iterations):
            cpu_percent = psutil.cpu_percent(interval=interval)
            samples.append(cpu_percent)
        
        if not samples:
            return {"error": "No CPU samples collected"}
        
        return {
            "duration_seconds": duration_seconds,
            "samples": len(samples),
            "cpu_usage": {
                "min_percent": min(samples),
                "max_percent": max(samples),
                "avg_percent": statistics.mean(samples),
                "median_percent": statistics.median(samples)
            },
            "cpu_info": {
                "logical_cores": psutil.cpu_count(),
                "physical_cores": psutil.cpu_count(logical=False)
            },
            "recommendations": PerformanceOptimizer._generate_cpu_recommendations(samples)
        }
    
    @staticmethod
    def _generate_cpu_recommendations(cpu_samples: List[float]) -> List[str]:
        """Generate CPU optimization recommendations"""
        recommendations = []
        avg_cpu = statistics.mean(cpu_samples)
        max_cpu = max(cpu_samples)
        
        if avg_cpu > 80:
            recommendations.append("High average CPU usage - consider scaling up or optimizing algorithms")
        elif avg_cpu > 60:
            recommendations.append("Moderate CPU usage - monitor for sustained load")
        
        if max_cpu > 95:
            recommendations.append("CPU spikes detected - investigate intensive operations")
        
        # Check for high variability
        if len(cpu_samples) > 1:
            cpu_std = statistics.stdev(cpu_samples)
            if cpu_std > 20:
                recommendations.append("High CPU variability - consider load balancing")
        
        if not recommendations:
            recommendations.append("CPU usage is within normal parameters")
        
        return recommendations

# Performance decorators
def measure_performance(operation_name: str = None):
    """Decorator to measure function performance"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                success = True
                error_msg = None
            except Exception as e:
                result = None
                success = False
                error_msg = str(e)
                raise
            finally:
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000
                
                # Record timing
                performance_tracker.record_timing(
                    operation=op_name,
                    duration_ms=duration_ms,
                    success=success,
                    metadata={"error": error_msg} if error_msg else {}
                )
            
            return result
        
        return wrapper
    return decorator

def measure_time(operation: str):
    """Context manager for measuring execution time"""
    class TimeContext:
        def __init__(self, operation_name: str):
            self.operation = operation_name
            self.start_time = None
            
        def __enter__(self):
            self.start_time = time.time()
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            end_time = time.time()
            duration_ms = (end_time - self.start_time) * 1000
            success = exc_type is None
            
            performance_tracker.record_timing(
                operation=self.operation,
                duration_ms=duration_ms,
                success=success,
                metadata={"exception": str(exc_val)} if exc_val else {}
            )
    
    return TimeContext(operation)

# Global performance tracker instance
performance_tracker = PerformanceTracker()

# Convenience functions
def record_operation_time(operation: str, duration_ms: float, success: bool = True):
    """Record timing for an operation"""
    performance_tracker.record_timing(operation, duration_ms, success)

def get_performance_stats(operation: str = None) -> Dict[str, Any]:
    """Get performance statistics"""
    if operation:
        return performance_tracker.get_operation_stats(operation)
    else:
        return performance_tracker.get_all_operation_stats()

def get_system_performance() -> Dict[str, Any]:
    """Get system performance summary"""
    return performance_tracker.get_system_performance_summary()

def detect_slow_operations() -> List[Dict[str, Any]]:
    """Detect slow operations"""
    return performance_tracker.get_slowest_operations()

def check_performance_health() -> Dict[str, Any]:
    """Check overall performance health"""
    summary = performance_tracker.get_system_performance_summary()
    degradation = performance_tracker.detect_performance_degradation()
    slow_ops = performance_tracker.get_slowest_operations(3)
    
    return {
        "status": summary.get("performance_status", "unknown"),
        "summary": summary,
        "degraded_operations": degradation,
        "slowest_operations": slow_ops,
        "recommendations": _generate_performance_recommendations(summary, degradation, slow_ops)
    }

def _generate_performance_recommendations(summary: Dict[str, Any], 
                                        degradation: List[Dict[str, Any]], 
                                        slow_ops: List[Dict[str, Any]]) -> List[str]:
    """Generate performance recommendations"""
    recommendations = []
    
    if summary.get("overall_error_rate", 0) > 5:
        recommendations.append("High error rate detected - investigate failing operations")
    
    if degradation:
        recommendations.append(f"Performance degradation in {len(degradation)} operations - review recent changes")
    
    if slow_ops and slow_ops[0].get("avg_response_time_ms", 0) > 3000:
        recommendations.append("Slow operations detected - consider optimization or caching")
    
    current_metrics = summary.get("current_metrics", {})
    if current_metrics.get("cpu_percent", 0) > 70:
        recommendations.append("High CPU usage - consider scaling or algorithm optimization")
    
    if current_metrics.get("memory_percent", 0) > 80:
        recommendations.append("High memory usage - investigate memory leaks or increase capacity")
    
    if not recommendations:
        recommendations.append("Performance is within acceptable parameters")
    
    return recommendations

# Auto-start system metrics collection
def start_performance_monitoring(interval_seconds: int = 60):
    """Start automatic performance monitoring"""
    def monitor_loop():
        while True:
            try:
                performance_tracker.record_system_metrics()
                time.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"❌ Performance monitoring error: {e}")
                time.sleep(interval_seconds)
    
    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()
    logger.info(f"🔍 Performance monitoring started (interval: {interval_seconds}s)")

# Export main classes and functions
__all__ = [
    'PerformanceMetric', 'TimingResult', 'PerformanceTracker', 'PerformanceOptimizer',
    'performance_tracker', 'measure_performance', 'measure_time',
    'record_operation_time', 'get_performance_stats', 'get_system_performance',
    'detect_slow_operations', 'check_performance_health', 'start_performance_monitoring'
]
