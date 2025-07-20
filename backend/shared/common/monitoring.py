# backend/shared/common/monitoring.py
"""
Shared Monitoring System for NAVA
Comprehensive monitoring, metrics collection, and alerting
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Deque
from collections import defaultdict, deque
from dataclasses import dataclass, field
import json
import threading
from uuid import uuid4

logger = logging.getLogger(__name__)

@dataclass
class MetricPoint:
    """Individual metric data point"""
    timestamp: datetime
    value: float
    tags: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass  
class ServiceHealth:
    """Service health status"""
    service_name: str
    status: str  # healthy, degraded, unhealthy
    last_check: datetime
    response_time: Optional[float] = None
    error_rate: Optional[float] = None
    uptime_percentage: Optional[float] = None
    issues: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class SystemMonitor:
    """
    Comprehensive system monitoring and metrics collection
    Handles metrics, health checks, alerting, and performance tracking
    """
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.start_time = datetime.now()
        
        # Metrics storage - using deque for efficient rotation
        self.metrics: Dict[str, Deque[MetricPoint]] = defaultdict(lambda: deque(maxlen=10000))
        
        # Service health tracking
        self.service_health: Dict[str, ServiceHealth] = {}
        
        # Request tracking
        self.request_counters: Dict[str, int] = defaultdict(int)
        self.response_times: Dict[str, List[float]] = defaultdict(list)
        self.error_counts: Dict[str, int] = defaultdict(int)
        
        # Alerting system
        self.alerts: List[Dict[str, Any]] = []
        self.alert_rules: List[Dict[str, Any]] = []
        
        # Health check functions
        self.health_checks: Dict[str, Callable] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("✅ System Monitor initialized")
    
    def record_metric(self, name: str, value: float, 
                     tags: Optional[Dict[str, Any]] = None,
                     metadata: Optional[Dict[str, Any]] = None):
        """Record a metric data point"""
        with self._lock:
            point = MetricPoint(
                timestamp=datetime.now(),
                value=value,
                tags=tags or {},
                metadata=metadata or {}
            )
            self.metrics[name].append(point)
            
        logger.debug(f"📊 Metric recorded: {name} = {value}")
    
    def record_request(self, service: str, endpoint: str, 
                      response_time: float, status_code: int):
        """Record API request metrics"""
        with self._lock:
            metric_key = f"{service}.{endpoint}"
            
            self.request_counters[metric_key] += 1
            self.response_times[metric_key].append(response_time)
            
            # Keep only recent response times (last 100)
            if len(self.response_times[metric_key]) > 100:
                self.response_times[metric_key] = self.response_times[metric_key][-100:]
            
            # Track errors (4xx, 5xx status codes)
            if status_code >= 400:
                self.error_counts[metric_key] += 1
            
            # Record as metrics
            self.record_metric(f"{metric_key}.requests", 1)
            self.record_metric(f"{metric_key}.response_time", response_time)
            
            if status_code >= 400:
                self.record_metric(f"{metric_key}.errors", 1)
        
        logger.debug(f"🌐 Request recorded: {service}.{endpoint} - {status_code} in {response_time}s")
    
    def record_ai_request(self, model: str, response_time: float, 
                         success: bool, quality_score: Optional[float] = None):
        """Record AI model request metrics"""
        with self._lock:
            metric_key = f"ai.{model}"
            
            self.request_counters[metric_key] += 1
            self.response_times[metric_key].append(response_time)
            
            if not success:
                self.error_counts[metric_key] += 1
            
            # Record detailed metrics
            self.record_metric(f"ai.{model}.requests", 1)
            self.record_metric(f"ai.{model}.response_time", response_time)
            
            if quality_score is not None:
                self.record_metric(f"ai.{model}.quality_score", quality_score)
            
            if not success:
                self.record_metric(f"ai.{model}.errors", 1)
        
        logger.debug(f"🤖 AI request recorded: {model} - {'success' if success else 'failure'} in {response_time}s")
    
    def update_service_health(self, service_name: str, status: str,
                             response_time: Optional[float] = None,
                             error_rate: Optional[float] = None,
                             issues: Optional[List[str]] = None,
                             metadata: Optional[Dict[str, Any]] = None):
        """Update service health status"""
        with self._lock:
            now = datetime.now()
            
            # Calculate uptime percentage
            uptime_percentage = None
            if service_name in self.service_health:
                # Simple uptime calculation (weighted average)
                previous_health = self.service_health[service_name]
                if previous_health.uptime_percentage is not None:
                    # Weight: 80% previous, 20% current
                    current_weight = 1.0 if status == "healthy" else 0.5 if status == "degraded" else 0.0
                    uptime_percentage = (previous_health.uptime_percentage * 0.8) + (current_weight * 0.2)
                else:
                    uptime_percentage = 1.0 if status == "healthy" else 0.5 if status == "degraded" else 0.0
            else:
                uptime_percentage = 1.0 if status == "healthy" else 0.5 if status == "degraded" else 0.0
            
            self.service_health[service_name] = ServiceHealth(
                service_name=service_name,
                status=status,
                last_check=now,
                response_time=response_time,
                error_rate=error_rate,
                uptime_percentage=uptime_percentage,
                issues=issues or [],
                metadata=metadata or {}
            )
            
            # Record health as metric
            health_score = 1.0 if status == "healthy" else 0.5 if status == "degraded" else 0.0
            self.record_metric(f"service.{service_name}.health", health_score)
        
        logger.debug(f"🏥 Service health updated: {service_name} = {status}")
    
    def register_health_check(self, service_name: str, check_function: Callable):
        """Register a health check function for a service"""
        self.health_checks[service_name] = check_function
        logger.info(f"✅ Health check registered for {service_name}")
    
    def run_health_checks(self):
        """Run all registered health checks"""
        for service_name, check_function in self.health_checks.items():
            try:
                result = check_function()
                
                if isinstance(result, dict):
                    status = result.get("status", "unknown")
                    response_time = result.get("response_time")
                    error_rate = result.get("error_rate")
                    issues = result.get("issues", [])
                    metadata = result.get("metadata", {})
                elif isinstance(result, bool):
                    status = "healthy" if result else "unhealthy"
                    response_time = None
                    error_rate = None
                    issues = [] if result else ["Health check failed"]
                    metadata = {}
                else:
                    status = "unhealthy"
                    response_time = None
                    error_rate = None
                    issues = ["Invalid health check response"]
                    metadata = {}
                
                self.update_service_health(
                    service_name, status, response_time, error_rate, issues, metadata
                )
                
            except Exception as e:
                logger.error(f"❌ Health check failed for {service_name}: {e}")
                self.update_service_health(
                    service_name, "unhealthy", 
                    issues=[f"Health check exception: {str(e)}"]
                )
    
    def create_alert(self, alert_type: str, message: str,
                    severity: str = "info", service: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new alert"""
        alert_id = str(uuid4())
        
        alert = {
            "id": alert_id,
            "type": alert_type,
            "message": message,
            "severity": severity,  # info, warning, error, critical
            "service": service,
            "status": "active",
            "created_at": datetime.now().isoformat(),
            "resolved_at": None,
            "metadata": metadata or {}
        }
        
        with self._lock:
            self.alerts.append(alert)
        
        logger.warning(f"🚨 Alert created: {alert_type} - {message}")
        return alert_id
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert by ID"""
        with self._lock:
            for alert in self.alerts:
                if alert["id"] == alert_id and alert["status"] == "active":
                    alert["status"] = "resolved"
                    alert["resolved_at"] = datetime.now().isoformat()
                    logger.info(f"✅ Alert resolved: {alert_id}")
                    return True
        return False
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts"""
        with self._lock:
            return [alert for alert in self.alerts if alert["status"] == "active"]
    
    def add_alert_rule(self, name: str, condition: str, 
                      threshold: float, severity: str = "warning",
                      enabled: bool = True):
        """Add an alert rule"""
        rule = {
            "name": name,
            "condition": condition,
            "threshold": threshold,
            "severity": severity,
            "enabled": enabled,
            "created_at": datetime.now().isoformat()
        }
        
        self.alert_rules.append(rule)
        logger.info(f"📋 Alert rule added: {name}")
    
    def check_alert_rules(self):
        """Check all alert rules and trigger alerts if needed"""
        for rule in self.alert_rules:
            if not rule["enabled"]:
                continue
                
            try:
                # Simple rule checking (can be extended)
                if rule["condition"] == "error_rate > threshold":
                    self._check_error_rate_rule(rule)
                elif rule["condition"] == "response_time > threshold":
                    self._check_response_time_rule(rule)
                    
            except Exception as e:
                logger.error(f"❌ Alert rule check failed for {rule['name']}: {e}")
    
    def _check_error_rate_rule(self, rule: Dict[str, Any]):
        """Check error rate alert rule"""
        for metric_key, error_count in self.error_counts.items():
            request_count = self.request_counters.get(metric_key, 0)
            if request_count > 0:
                error_rate = (error_count / request_count) * 100
                if error_rate > rule["threshold"]:
                    self.create_alert(
                        rule["name"],
                        f"High error rate for {metric_key}: {error_rate:.1f}%",
                        rule["severity"],
                        metadata={"error_rate": error_rate, "threshold": rule["threshold"]}
                    )
    
    def _check_response_time_rule(self, rule: Dict[str, Any]):
        """Check response time alert rule"""
        for metric_key, response_times in self.response_times.items():
            if response_times:
                avg_response_time = sum(response_times) / len(response_times) * 1000  # Convert to ms
                if avg_response_time > rule["threshold"]:
                    self.create_alert(
                        rule["name"],
                        f"High response time for {metric_key}: {avg_response_time:.1f}ms",
                        rule["severity"],
                        metadata={"response_time": avg_response_time, "threshold": rule["threshold"]}
                    )
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system overview"""
        with self._lock:
            # Service summary
            total_services = len(self.service_health)
            healthy_services = len([s for s in self.service_health.values() if s.status == "healthy"])
            degraded_services = len([s for s in self.service_health.values() if s.status == "degraded"])
            unhealthy_services = len([s for s in self.service_health.values() if s.status == "unhealthy"])
            
            # Request summary
            total_requests = sum(self.request_counters.values())
            total_errors = sum(self.error_counts.values())
            error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0.0
            
            # Overall status
            overall_status = self._determine_overall_status()
            
            return {
                "status": overall_status,
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                "services": {
                    "total": total_services,
                    "healthy": healthy_services,
                    "degraded": degraded_services,
                    "unhealthy": unhealthy_services
                },
                "requests": {
                    "total": total_requests,
                    "error_count": total_errors,
                    "error_rate_percent": round(error_rate, 2)
                },
                "alerts": {
                    "active": len(self.get_active_alerts()),
                    "total": len(self.alerts)
                }
            }
    
    def _determine_overall_status(self) -> str:
        """Determine overall system status"""
        if not self.service_health:
            return "unknown"
        
        statuses = [service.status for service in self.service_health.values()]
        
        if all(status == "healthy" for status in statuses):
            return "healthy"
        elif any(status == "unhealthy" for status in statuses):
            if len([s for s in statuses if s == "unhealthy"]) > len(statuses) / 2:
                return "critical"
            else:
                return "degraded"
        elif any(status == "degraded" for status in statuses):
            return "warning"
        else:
            return "healthy"
    
    def get_service_metrics(self, service_name: str, 
                           period_hours: int = 1) -> Dict[str, Any]:
        """Get metrics for a specific service"""
        cutoff_time = datetime.now() - timedelta(hours=period_hours)
        
        # Get service health
        health = self.service_health.get(service_name)
        
        # Get service metrics
        service_metrics = {}
        for metric_name, points in self.metrics.items():
            if metric_name.startswith(f"service.{service_name}.") or metric_name.startswith(f"{service_name}."):
                recent_points = [p for p in points if p.timestamp > cutoff_time]
                if recent_points:
                    values = [p.value for p in recent_points]
                    service_metrics[metric_name] = {
                        "current": values[-1] if values else None,
                        "average": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "count": len(values)
                    }
        
        # Get request metrics
        request_metrics = {}
        for metric_key in self.request_counters:
            if metric_key.startswith(service_name):
                request_count = self.request_counters.get(metric_key, 0)
                error_count = self.error_counts.get(metric_key, 0)
                response_times = self.response_times.get(metric_key, [])
                
                request_metrics[metric_key] = {
                    "requests": request_count,
                    "errors": error_count,
                    "error_rate": (error_count / request_count * 100) if request_count > 0 else 0,
                    "avg_response_time": sum(response_times) / len(response_times) if response_times else 0
                }
        
        return {
            "service_name": service_name,
            "health": health.__dict__ if health else None,
            "metrics": service_metrics,
            "requests": request_metrics,
            "period_hours": period_hours,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_ai_model_metrics(self, period_hours: int = 1) -> Dict[str, Any]:
        """Get AI model performance metrics"""
        cutoff_time = datetime.now() - timedelta(hours=period_hours)
        
        ai_metrics = {}
        
        # Find all AI models
        ai_models = set()
        for metric_key in self.request_counters:
            if metric_key.startswith("ai."):
                model = metric_key.split(".")[1]
                ai_models.add(model)
        
        for model in ai_models:
            metric_key = f"ai.{model}"
            request_count = self.request_counters.get(metric_key, 0)
            error_count = self.error_counts.get(metric_key, 0)
            response_times = self.response_times.get(metric_key, [])
            
            # Get quality scores from metrics
            quality_points = [p for p in self.metrics.get(f"ai.{model}.quality_score", []) 
                            if p.timestamp > cutoff_time]
            quality_scores = [p.value for p in quality_points]
            
            ai_metrics[model] = {
                "requests": {
                    "count": request_count,
                    "errors": error_count,
                    "success_rate": ((request_count - error_count) / request_count * 100) if request_count > 0 else 0
                },
                "performance": {
                    "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
                    "min_response_time": min(response_times) if response_times else 0,
                    "max_response_time": max(response_times) if response_times else 0
                },
                "quality": {
                    "avg_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
                    "min_score": min(quality_scores) if quality_scores else 0,
                    "max_score": max(quality_scores) if quality_scores else 0,
                    "sample_count": len(quality_scores)
                }
            }
        
        return {
            "models": ai_metrics,
            "period_hours": period_hours,
            "timestamp": datetime.now().isoformat()
        }
    
    def cleanup_old_data(self):
        """Clean up old metrics and alerts"""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        
        with self._lock:
            # Clean old metrics
            for metric_name in list(self.metrics.keys()):
                # Filter out old points
                recent_points = [p for p in self.metrics[metric_name] if p.timestamp > cutoff_time]
                self.metrics[metric_name] = deque(recent_points, maxlen=10000)
                
                # Remove empty metrics
                if not self.metrics[metric_name]:
                    del self.metrics[metric_name]
            
            # Clean old alerts (keep resolved alerts for 24 hours)
            alert_cutoff = datetime.now() - timedelta(hours=24)
            self.alerts = [
                alert for alert in self.alerts 
                if alert["status"] == "active" or 
                (alert["resolved_at"] and 
                 datetime.fromisoformat(alert["resolved_at"]) > alert_cutoff)
            ]
        
        logger.info(f"🧹 Cleaned up old monitoring data (retention: {self.retention_hours}h)")
    
    def export_metrics(self, format_type: str = "json") -> str:
        """Export metrics in specified format"""
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "system_overview": self.get_system_overview(),
            "service_health": {name: health.__dict__ for name, health in self.service_health.items()},
            "active_alerts": self.get_active_alerts(),
            "uptime_hours": (datetime.now() - self.start_time).total_seconds() / 3600
        }
        
        if format_type.lower() == "json":
            return json.dumps(export_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def get_performance_summary(self, period_hours: int = 1) -> Dict[str, Any]:
        """Get overall performance summary"""
        cutoff_time = datetime.now() - timedelta(hours=period_hours)
        
        # Calculate totals
        total_requests = sum(self.request_counters.values())
        total_errors = sum(self.error_counts.values())
        
        # Calculate average response time
        all_response_times = []
        for times in self.response_times.values():
            all_response_times.extend(times)
        
        avg_response_time = sum(all_response_times) / len(all_response_times) if all_response_times else 0
        
        # Service availability
        total_services = len(self.service_health)
        available_services = len([s for s in self.service_health.values() if s.status in ["healthy", "degraded"]])
        availability = (available_services / total_services * 100) if total_services > 0 else 100
        
        return {
            "period_hours": period_hours,
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": (total_errors / total_requests * 100) if total_requests > 0 else 0,
            "avg_response_time": avg_response_time,
            "service_availability": availability,
            "timestamp": datetime.now().isoformat()
        }

# Global system monitor instance
system_monitor = SystemMonitor()

# Convenience functions
def record_metric(name: str, value: float, tags: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
    """Record a metric data point"""
    system_monitor.record_metric(name, value, tags, metadata)

def record_request(service: str, endpoint: str, response_time: float, status_code: int):
    """Record API request metrics"""
    system_monitor.record_request(service, endpoint, response_time, status_code)

def record_ai_request(model: str, response_time: float, success: bool, 
                     quality_score: Optional[float] = None):
    """Record AI model request metrics"""
    system_monitor.record_ai_request(model, response_time, success, quality_score)

def update_service_health(service_name: str, status: str, **kwargs):
    """Update service health status"""
    system_monitor.update_service_health(service_name, status, **kwargs)

def get_system_status() -> Dict[str, Any]:
    """Get current system status"""
    return system_monitor.get_system_overview()

def create_alert(alert_type: str, message: str, severity: str = "info", **kwargs) -> str:
    """Create a new alert"""
    return system_monitor.create_alert(alert_type, message, severity, **kwargs)

def run_health_checks():
    """Run all registered health checks"""
    system_monitor.run_health_checks()

def cleanup_monitoring_data():
    """Clean up old monitoring data"""
    system_monitor.cleanup_old_data()

# Initialize logging
logger.info("✅ Shared monitoring system initialized")