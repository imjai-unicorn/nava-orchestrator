"""
Performance Tracker - Week 4 Performance Tracking System
Real-time performance monitoring and optimization for NAVA system
"""

import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import json

@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    timestamp: datetime
    service_name: str
    operation: str
    duration_ms: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ServiceHealth:
    """Service health status"""
    service_name: str
    avg_response_time: float
    success_rate: float
    error_count: int
    last_error: Optional[str]
    status: str  # 'healthy', 'degraded', 'critical'
    updated_at: datetime

class PerformanceTracker:
    """
    Comprehensive performance tracking system for NAVA
    Monitors response times, success rates, and system health
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Performance data storage
        self.metrics = deque(maxlen=10000)  # Last 10k metrics
        self.service_metrics = defaultdict(lambda: deque(maxlen=1000))
        
        # Health monitoring
        self.service_health = {}
        self.system_health_score = 1.0
        
        # Performance thresholds
        self.thresholds = {
            'response_time_warning': 2000,  # 2s
            'response_time_critical': 5000,  # 5s
            'success_rate_warning': 0.95,   # 95%
            'success_rate_critical': 0.90,  # 90%
            'error_rate_warning': 0.05,     # 5%
            'error_rate_critical': 0.10     # 10%
        }
        
        # Real-time tracking
        self.active_requests = {}
        self.performance_trends = defaultdict(list)
        
        # Background monitoring
        self._monitoring_task = None
        self._is_monitoring = False

    async def start_monitoring(self):
        """Start background performance monitoring"""
        if self._is_monitoring:
            return
            
        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitor_loop())
        self.logger.info("Performance monitoring started")

    async def stop_monitoring(self):
        """Stop background performance monitoring"""
        self._is_monitoring = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Performance monitoring stopped")

    def start_request_tracking(self, request_id: str, service_name: str, operation: str) -> str:
        """Start tracking a request"""
        tracking_data = {
            'request_id': request_id,
            'service_name': service_name,
            'operation': operation,
            'start_time': time.time(),
            'timestamp': datetime.now()
        }
        
        self.active_requests[request_id] = tracking_data
        return request_id

    def end_request_tracking(self, request_id: str, success: bool = True, 
                           error_message: Optional[str] = None, 
                           metadata: Optional[Dict[str, Any]] = None) -> PerformanceMetric:
        """End tracking a request and record metrics"""
        if request_id not in self.active_requests:
            self.logger.warning(f"Request {request_id} not found in active tracking")
            return None

        tracking_data = self.active_requests.pop(request_id)
        end_time = time.time()
        duration_ms = (end_time - tracking_data['start_time']) * 1000

        metric = PerformanceMetric(
            timestamp=datetime.now(),
            service_name=tracking_data['service_name'],
            operation=tracking_data['operation'],
            duration_ms=duration_ms,
            success=success,
            error_message=error_message,
            metadata=metadata or {}
        )

        # Store metrics
        self.metrics.append(metric)
        self.service_metrics[tracking_data['service_name']].append(metric)

        # Update health status
        self._update_service_health(tracking_data['service_name'])
        
        # Log performance issues
        if duration_ms > self.thresholds['response_time_critical']:
            self.logger.warning(f"Critical response time: {duration_ms:.2f}ms for {tracking_data['service_name']}.{tracking_data['operation']}")
        elif duration_ms > self.thresholds['response_time_warning']:
            self.logger.info(f"Slow response time: {duration_ms:.2f}ms for {tracking_data['service_name']}.{tracking_data['operation']}")

        return metric

    def _update_service_health(self, service_name: str):
        """Update health status for a service"""
        service_metrics = list(self.service_metrics[service_name])
        if not service_metrics:
            return

        # Calculate metrics for last 5 minutes
        five_minutes_ago = datetime.now() - timedelta(minutes=5)
        recent_metrics = [m for m in service_metrics if m.timestamp > five_minutes_ago]
        
        if not recent_metrics:
            return

        # Calculate health indicators
        avg_response_time = statistics.mean([m.duration_ms for m in recent_metrics])
        success_rate = sum(1 for m in recent_metrics if m.success) / len(recent_metrics)
        error_count = sum(1 for m in recent_metrics if not m.success)
        
        # Determine status
        status = 'healthy'
        if (avg_response_time > self.thresholds['response_time_critical'] or 
            success_rate < self.thresholds['success_rate_critical']):
            status = 'critical'
        elif (avg_response_time > self.thresholds['response_time_warning'] or 
              success_rate < self.thresholds['success_rate_warning']):
            status = 'degraded'

        # Get last error
        last_error = None
        for metric in reversed(recent_metrics):
            if not metric.success and metric.error_message:
                last_error = metric.error_message
                break

        self.service_health[service_name] = ServiceHealth(
            service_name=service_name,
            avg_response_time=avg_response_time,
            success_rate=success_rate,
            error_count=error_count,
            last_error=last_error,
            status=status,
            updated_at=datetime.now()
        )

    def get_service_health(self, service_name: Optional[str] = None) -> Dict[str, ServiceHealth]:
        """Get health status for services"""
        if service_name:
            return {service_name: self.service_health.get(service_name)}
        return dict(self.service_health)

    def get_system_health_score(self) -> float:
        """Calculate overall system health score (0.0 to 1.0)"""
        if not self.service_health:
            return 1.0

        health_scores = []
        for health in self.service_health.values():
            # Score based on success rate and response time
            success_score = health.success_rate
            
            # Response time score (1.0 if under warning, decreases linearly)
            if health.avg_response_time <= self.thresholds['response_time_warning']:
                time_score = 1.0
            elif health.avg_response_time >= self.thresholds['response_time_critical']:
                time_score = 0.0
            else:
                time_score = 1.0 - ((health.avg_response_time - self.thresholds['response_time_warning']) / 
                                  (self.thresholds['response_time_critical'] - self.thresholds['response_time_warning']))
            
            # Combined score (weighted average)
            service_score = (success_score * 0.7) + (time_score * 0.3)
            health_scores.append(service_score)

        return statistics.mean(health_scores) if health_scores else 1.0

    def get_performance_summary(self, time_window_minutes: int = 30) -> Dict[str, Any]:
        """Get performance summary for specified time window"""
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        recent_metrics = [m for m in self.metrics if m.timestamp > cutoff_time]

        if not recent_metrics:
            return {
                'total_requests': 0,
                'avg_response_time': 0,
                'success_rate': 1.0,
                'error_count': 0,
                'system_health_score': self.get_system_health_score()
            }

        # Calculate summary statistics
        total_requests = len(recent_metrics)
        avg_response_time = statistics.mean([m.duration_ms for m in recent_metrics])
        success_count = sum(1 for m in recent_metrics if m.success)
        success_rate = success_count / total_requests
        error_count = total_requests - success_count

        # Response time percentiles
        response_times = sorted([m.duration_ms for m in recent_metrics])
        p50 = response_times[int(len(response_times) * 0.5)] if response_times else 0
        p95 = response_times[int(len(response_times) * 0.95)] if response_times else 0
        p99 = response_times[int(len(response_times) * 0.99)] if response_times else 0

        return {
            'time_window_minutes': time_window_minutes,
            'total_requests': total_requests,
            'avg_response_time': avg_response_time,
            'response_time_p50': p50,
            'response_time_p95': p95,
            'response_time_p99': p99,
            'success_rate': success_rate,
            'error_count': error_count,
            'requests_per_minute': total_requests / time_window_minutes,
            'system_health_score': self.get_system_health_score(),
            'service_breakdown': self._get_service_breakdown(recent_metrics)
        }

    def _get_service_breakdown(self, metrics: List[PerformanceMetric]) -> Dict[str, Dict[str, Any]]:
        """Get performance breakdown by service"""
        service_data = defaultdict(list)
        for metric in metrics:
            service_data[metric.service_name].append(metric)

        breakdown = {}
        for service_name, service_metrics in service_data.items():
            breakdown[service_name] = {
                'total_requests': len(service_metrics),
                'avg_response_time': statistics.mean([m.duration_ms for m in service_metrics]),
                'success_rate': sum(1 for m in service_metrics if m.success) / len(service_metrics),
                'error_count': sum(1 for m in service_metrics if not m.success)
            }

        return breakdown

    def get_performance_trends(self, hours: int = 24) -> Dict[str, List[Dict[str, Any]]]:
        """Get performance trends over time"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics if m.timestamp > cutoff_time]

        # Group by hour
        hourly_data = defaultdict(list)
        for metric in recent_metrics:
            hour_key = metric.timestamp.replace(minute=0, second=0, microsecond=0)
            hourly_data[hour_key].append(metric)

        trends = {}
        for hour, hour_metrics in hourly_data.items():
            avg_response_time = statistics.mean([m.duration_ms for m in hour_metrics])
            success_rate = sum(1 for m in hour_metrics if m.success) / len(hour_metrics)
            
            trends[hour.isoformat()] = {
                'timestamp': hour.isoformat(),
                'total_requests': len(hour_metrics),
                'avg_response_time': avg_response_time,
                'success_rate': success_rate,
                'error_count': sum(1 for m in hour_metrics if not m.success)
            }

        return dict(sorted(trends.items()))

    async def _monitor_loop(self):
        """Background monitoring loop"""
        while self._is_monitoring:
            try:
                # Update system health
                system_health = self.get_system_health_score()
                
                # Log system status periodically
                if int(time.time()) % 300 == 0:  # Every 5 minutes
                    summary = self.get_performance_summary(5)
                    self.logger.info(f"System health: {system_health:.2f}, "
                                   f"Avg response: {summary['avg_response_time']:.2f}ms, "
                                   f"Success rate: {summary['success_rate']:.2%}")

                # Check for performance issues
                for service_name, health in self.service_health.items():
                    if health.status == 'critical':
                        self.logger.error(f"Service {service_name} is in critical state: "
                                        f"avg_response={health.avg_response_time:.2f}ms, "
                                        f"success_rate={health.success_rate:.2%}")
                    elif health.status == 'degraded':
                        self.logger.warning(f"Service {service_name} is degraded: "
                                          f"avg_response={health.avg_response_time:.2f}ms, "
                                          f"success_rate={health.success_rate:.2%}")

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    def record_custom_metric(self, service_name: str, operation: str, 
                           value: float, success: bool = True, 
                           metadata: Optional[Dict[str, Any]] = None):
        """Record a custom performance metric"""
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            service_name=service_name,
            operation=operation,
            duration_ms=value,
            success=success,
            metadata=metadata or {}
        )
        
        self.metrics.append(metric)
        self.service_metrics[service_name].append(metric)
        self._update_service_health(service_name)

    def get_slow_operations(self, threshold_ms: float = 2000, 
                          limit: int = 10) -> List[Dict[str, Any]]:
        """Get slowest operations above threshold"""
        slow_metrics = [
            m for m in self.metrics 
            if m.duration_ms > threshold_ms
        ]
        
        # Sort by duration descending
        slow_metrics.sort(key=lambda x: x.duration_ms, reverse=True)
        
        return [
            {
                'service': m.service_name,
                'operation': m.operation,
                'duration_ms': m.duration_ms,
                'timestamp': m.timestamp.isoformat(),
                'success': m.success,
                'error': m.error_message
            }
            for m in slow_metrics[:limit]
        ]

    def export_metrics(self, format: str = 'json') -> str:
        """Export performance metrics"""
        if format == 'json':
            data = {
                'summary': self.get_performance_summary(),
                'service_health': {
                    name: {
                        'avg_response_time': health.avg_response_time,
                        'success_rate': health.success_rate,
                        'error_count': health.error_count,
                        'status': health.status,
                        'updated_at': health.updated_at.isoformat()
                    }
                    for name, health in self.service_health.items()
                },
                'trends': self.get_performance_trends(24),
                'slow_operations': self.get_slow_operations()
            }
            return json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")

# Global performance tracker instance
performance_tracker = PerformanceTracker()

# Decorator for automatic performance tracking
def track_performance(service_name: str, operation: str = None):
    """Decorator to automatically track function performance"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            operation_name = operation or func.__name__
            request_id = f"{service_name}_{operation_name}_{int(time.time() * 1000)}"
            
            performance_tracker.start_request_tracking(request_id, service_name, operation_name)
            
            try:
                result = await func(*args, **kwargs)
                performance_tracker.end_request_tracking(request_id, success=True)
                return result
            except Exception as e:
                performance_tracker.end_request_tracking(
                    request_id, 
                    success=False, 
                    error_message=str(e)
                )
                raise
        
        def sync_wrapper(*args, **kwargs):
            operation_name = operation or func.__name__
            request_id = f"{service_name}_{operation_name}_{int(time.time() * 1000)}"
            
            performance_tracker.start_request_tracking(request_id, service_name, operation_name)
            
            try:
                result = func(*args, **kwargs)
                performance_tracker.end_request_tracking(request_id, success=True)
                return result
            except Exception as e:
                performance_tracker.end_request_tracking(
                    request_id, 
                    success=False, 
                    error_message=str(e)
                )
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator
