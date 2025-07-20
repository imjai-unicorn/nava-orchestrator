# backend/shared/common/alert_system.py
"""
Shared Alert System for NAVA
Comprehensive alerting, notifications, and escalation management
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
from uuid import uuid4

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    type: str
    message: str
    severity: AlertSeverity
    status: AlertStatus
    service: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    condition: str
    threshold: float
    severity: AlertSeverity
    enabled: bool = True
    service: Optional[str] = None
    cooldown_minutes: int = 5
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_triggered: Optional[datetime] = None

class AlertManager:
    """
    Comprehensive alert management system
    Handles alert creation, escalation, notifications, and lifecycle management
    """
    
    def __init__(self, max_alerts: int = 10000):
        self.max_alerts = max_alerts
        
        # Alert storage
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        
        # Notification handlers
        self.notification_handlers: Dict[str, Callable] = {}
        
        # Alert statistics
        self.alert_stats = {
            "total_created": 0,
            "total_resolved": 0,
            "total_acknowledged": 0,
            "by_severity": {severity.value: 0 for severity in AlertSeverity},
            "by_service": {},
            "response_times": []
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Default alert rules
        self._setup_default_rules()
        
        logger.info("✅ Alert Manager initialized")
    
    def _setup_default_rules(self):
        """Setup default alert rules"""
        default_rules = [
            AlertRule(
                name="high_error_rate",
                condition="error_rate > threshold",
                threshold=10.0,  # 10% error rate
                severity=AlertSeverity.WARNING,
                description="Trigger when error rate exceeds 10%"
            ),
            AlertRule(
                name="very_high_error_rate", 
                condition="error_rate > threshold",
                threshold=25.0,  # 25% error rate
                severity=AlertSeverity.ERROR,
                description="Trigger when error rate exceeds 25%"
            ),
            AlertRule(
                name="critical_error_rate",
                condition="error_rate > threshold", 
                threshold=50.0,  # 50% error rate
                severity=AlertSeverity.CRITICAL,
                description="Trigger when error rate exceeds 50%"
            ),
            AlertRule(
                name="high_response_time",
                condition="response_time > threshold",
                threshold=5000.0,  # 5 seconds
                severity=AlertSeverity.WARNING,
                description="Trigger when response time exceeds 5 seconds"
            ),
            AlertRule(
                name="service_down",
                condition="service_health == unhealthy",
                threshold=0.0,
                severity=AlertSeverity.CRITICAL,
                description="Trigger when service becomes unhealthy"
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.name] = rule
    
    def create_alert(self, alert_type: str, message: str,
                    severity: AlertSeverity = AlertSeverity.INFO,
                    service: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None,
                    tags: Optional[Set[str]] = None) -> str:
        """Create a new alert"""
        
        alert_id = str(uuid4())
        
        alert = Alert(
            id=alert_id,
            type=alert_type,
            message=message,
            severity=severity,
            status=AlertStatus.ACTIVE,
            service=service,
            metadata=metadata or {},
            tags=tags or set()
        )
        
        with self._lock:
            # Check for duplicates (same type, service, and similar message)
            if not self._is_duplicate_alert(alert):
                self.alerts[alert_id] = alert
                
                # Update statistics
                self.alert_stats["total_created"] += 1
                self.alert_stats["by_severity"][severity.value] += 1
                
                if service:
                    if service not in self.alert_stats["by_service"]:
                        self.alert_stats["by_service"][service] = 0
                    self.alert_stats["by_service"][service] += 1
                
                # Cleanup old alerts if needed
                self._cleanup_old_alerts()
                
                # Send notifications
                self._send_notifications(alert)
                
                logger.warning(f"🚨 Alert created: {alert_type} - {message} ({severity.value})")
                return alert_id
            else:
                logger.debug(f"⚠️ Duplicate alert suppressed: {alert_type}")
                return None
    
    def _is_duplicate_alert(self, new_alert: Alert) -> bool:
        """Check if alert is a duplicate of existing active alerts"""
        for alert in self.alerts.values():
            if (alert.status == AlertStatus.ACTIVE and
                alert.type == new_alert.type and
                alert.service == new_alert.service and
                abs((alert.created_at - new_alert.created_at).total_seconds()) < 300):  # 5 minutes
                # Consider it duplicate if messages are very similar
                if self._message_similarity(alert.message, new_alert.message) > 0.8:
                    return True
        return False
    
    def _message_similarity(self, msg1: str, msg2: str) -> float:
        """Calculate message similarity (simple word overlap)"""
        words1 = set(msg1.lower().split())
        words2 = set(msg2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an alert"""
        with self._lock:
            if alert_id in self.alerts:
                alert = self.alerts[alert_id]
                if alert.status == AlertStatus.ACTIVE:
                    alert.status = AlertStatus.ACKNOWLEDGED
                    alert.acknowledged_at = datetime.now()
                    alert.acknowledged_by = acknowledged_by
                    
                    self.alert_stats["total_acknowledged"] += 1
                    
                    logger.info(f"✅ Alert acknowledged: {alert_id} by {acknowledged_by}")
                    return True
        return False
    
    def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Resolve an alert"""
        with self._lock:
            if alert_id in self.alerts:
                alert = self.alerts[alert_id]
                if alert.status in [AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED]:
                    alert.status = AlertStatus.RESOLVED
                    alert.resolved_at = datetime.now()
                    alert.resolved_by = resolved_by
                    
                    # Calculate response time
                    response_time = (alert.resolved_at - alert.created_at).total_seconds()
                    self.alert_stats["response_times"].append(response_time)
                    self.alert_stats["total_resolved"] += 1
                    
                    logger.info(f"✅ Alert resolved: {alert_id} by {resolved_by} (response time: {response_time:.1f}s)")
                    return True
        return False
    
    def suppress_alert(self, alert_id: str) -> bool:
        """Suppress an alert (temporary silence)"""
        with self._lock:
            if alert_id in self.alerts:
                alert = self.alerts[alert_id]
                if alert.status == AlertStatus.ACTIVE:
                    alert.status = AlertStatus.SUPPRESSED
                    logger.info(f"🔇 Alert suppressed: {alert_id}")
                    return True
        return False
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None,
                         service: Optional[str] = None) -> List[Alert]:
        """Get active alerts with optional filtering"""
        with self._lock:
            alerts = [alert for alert in self.alerts.values() 
                     if alert.status == AlertStatus.ACTIVE]
            
            if severity:
                alerts = [alert for alert in alerts if alert.severity == severity]
            
            if service:
                alerts = [alert for alert in alerts if alert.service == service]
            
            # Sort by severity and creation time
            severity_order = {
                AlertSeverity.CRITICAL: 0,
                AlertSeverity.ERROR: 1, 
                AlertSeverity.WARNING: 2,
                AlertSeverity.INFO: 3
            }
            
            alerts.sort(key=lambda a: (severity_order[a.severity], a.created_at), reverse=True)
            return alerts
    
    def get_alert_history(self, hours: int = 24, 
                         service: Optional[str] = None) -> List[Alert]:
        """Get alert history for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            alerts = [alert for alert in self.alerts.values() 
                     if alert.created_at > cutoff_time]
            
            if service:
                alerts = [alert for alert in alerts if alert.service == service]
            
            alerts.sort(key=lambda a: a.created_at, reverse=True)
            return alerts
    
    def add_alert_rule(self, rule: AlertRule):
        """Add or update an alert rule"""
        with self._lock:
            self.alert_rules[rule.name] = rule
            logger.info(f"📋 Alert rule added/updated: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str) -> bool:
        """Remove an alert rule"""
        with self._lock:
            if rule_name in self.alert_rules:
                del self.alert_rules[rule_name]
                logger.info(f"🗑️ Alert rule removed: {rule_name}")
                return True
        return False
    
    def enable_alert_rule(self, rule_name: str) -> bool:
        """Enable an alert rule"""
        with self._lock:
            if rule_name in self.alert_rules:
                self.alert_rules[rule_name].enabled = True
                logger.info(f"✅ Alert rule enabled: {rule_name}")
                return True
        return False
    
    def disable_alert_rule(self, rule_name: str) -> bool:
        """Disable an alert rule"""
        with self._lock:
            if rule_name in self.alert_rules:
                self.alert_rules[rule_name].enabled = False
                logger.info(f"🔇 Alert rule disabled: {rule_name}")
                return True
        return False
    
    def check_alert_rules(self, metrics: Dict[str, Any]):
        """Check all alert rules against current metrics"""
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            # Check cooldown
            if (rule.last_triggered and 
                (datetime.now() - rule.last_triggered).total_seconds() < rule.cooldown_minutes * 60):
                continue
            
            try:
                if self._evaluate_rule_condition(rule, metrics):
                    # Trigger alert
                    alert_id = self.create_alert(
                        alert_type=rule_name,
                        message=f"Alert rule '{rule.name}' triggered: {rule.description or rule.condition}",
                        severity=rule.severity,
                        service=rule.service,
                        metadata={
                            "rule_name": rule.name,
                            "threshold": rule.threshold,
                            "condition": rule.condition,
                            "triggered_value": self._get_metric_value(rule, metrics)
                        }
                    )
                    
                    if alert_id:
                        rule.last_triggered = datetime.now()
                        
            except Exception as e:
                logger.error(f"❌ Error evaluating alert rule {rule_name}: {e}")
    
    def _evaluate_rule_condition(self, rule: AlertRule, metrics: Dict[str, Any]) -> bool:
        """Evaluate if rule condition is met"""
        condition = rule.condition.lower()
        threshold = rule.threshold
        
        if "error_rate > threshold" in condition:
            error_rate = metrics.get("error_rate", 0)
            return error_rate > threshold
            
        elif "response_time > threshold" in condition:
            response_time = metrics.get("avg_response_time", 0) * 1000  # Convert to ms
            return response_time > threshold
            
        elif "service_health == unhealthy" in condition:
            service_health = metrics.get("service_health", "healthy")
            return service_health == "unhealthy"
            
        # Add more condition types as needed
        return False
    
    def _get_metric_value(self, rule: AlertRule, metrics: Dict[str, Any]) -> Any:
        """Get the metric value that triggered the rule"""
        condition = rule.condition.lower()
        
        if "error_rate" in condition:
            return metrics.get("error_rate", 0)
        elif "response_time" in condition:
            return metrics.get("avg_response_time", 0) * 1000
        elif "service_health" in condition:
            return metrics.get("service_health", "unknown")
        
        return None
    
    def register_notification_handler(self, handler_name: str, handler_function: Callable):
        """Register a notification handler"""
        self.notification_handlers[handler_name] = handler_function
        logger.info(f"📬 Notification handler registered: {handler_name}")
    
    def _send_notifications(self, alert: Alert):
        """Send notifications for an alert"""
        for handler_name, handler in self.notification_handlers.items():
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"❌ Notification handler {handler_name} failed: {e}")
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get comprehensive alert statistics"""
        with self._lock:
            active_alerts = self.get_active_alerts()
            
            # Response time statistics
            response_times = self.alert_stats["response_times"]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            
            return {
                "total_alerts": len(self.alerts),
                "active_alerts": len(active_alerts),
                "total_created": self.alert_stats["total_created"],
                "total_resolved": self.alert_stats["total_resolved"],
                "total_acknowledged": self.alert_stats["total_acknowledged"],
                "resolution_rate": (self.alert_stats["total_resolved"] / max(1, self.alert_stats["total_created"]) * 100),
                "avg_response_time_seconds": avg_response_time,
                "by_severity": self.alert_stats["by_severity"].copy(),
                "by_service": self.alert_stats["by_service"].copy(),
                "active_by_severity": {
                    severity.value: len([a for a in active_alerts if a.severity == severity])
                    for severity in AlertSeverity
                },
                "alert_rules": {
                    "total": len(self.alert_rules),
                    "enabled": len([r for r in self.alert_rules.values() if r.enabled]),
                    "disabled": len([r for r in self.alert_rules.values() if not r.enabled])
                }
            }
    
    def _cleanup_old_alerts(self):
        """Clean up old resolved alerts"""
        if len(self.alerts) > self.max_alerts:
            with self._lock:
                # Keep active and recent alerts, remove old resolved ones
                cutoff_time = datetime.now() - timedelta(days=7)  # Keep 7 days of history
                
                alerts_to_remove = []
                for alert_id, alert in self.alerts.items():
                    if (alert.status == AlertStatus.RESOLVED and 
                        alert.resolved_at and alert.resolved_at < cutoff_time):
                        alerts_to_remove.append(alert_id)
                
                # Remove oldest alerts if still over limit
                if len(alerts_to_remove) > 0:
                    for alert_id in alerts_to_remove[:len(self.alerts) - self.max_alerts]:
                        del self.alerts[alert_id]
                
                logger.info(f"🧹 Cleaned up {len(alerts_to_remove)} old alerts")
    
    def export_alerts(self, format_type: str = "json", 
                     hours: int = 24) -> str:
        """Export alerts in specified format"""
        alerts = self.get_alert_history(hours)
        
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "period_hours": hours,
            "total_alerts": len(alerts),
            "statistics": self.get_alert_statistics(),
            "alerts": [
                {
                    "id": alert.id,
                    "type": alert.type,
                    "message": alert.message,
                    "severity": alert.severity.value,
                    "status": alert.status.value,
                    "service": alert.service,
                    "created_at": alert.created_at.isoformat(),
                    "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                    "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
                    "metadata": alert.metadata,
                    "tags": list(alert.tags)
                }
                for alert in alerts
            ]
        }
        
        if format_type.lower() == "json":
            return json.dumps(export_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")

# Global alert manager instance
alert_manager = AlertManager()

# Convenience functions
def create_alert(alert_type: str, message: str, 
                severity: str = "info", **kwargs) -> str:
    """Create a new alert"""
    severity_enum = AlertSeverity(severity.lower())
    return alert_manager.create_alert(alert_type, message, severity_enum, **kwargs)

def acknowledge_alert(alert_id: str, acknowledged_by: str = "system") -> bool:
    """Acknowledge an alert"""
    return alert_manager.acknowledge_alert(alert_id, acknowledged_by)

def resolve_alert(alert_id: str, resolved_by: str = "system") -> bool:
    """Resolve an alert"""
    return alert_manager.resolve_alert(alert_id, resolved_by)

def get_active_alerts(severity: Optional[str] = None, 
                     service: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get active alerts"""
    severity_enum = AlertSeverity(severity.lower()) if severity else None
    alerts = alert_manager.get_active_alerts(severity_enum, service)
    
    return [
        {
            "id": alert.id,
            "type": alert.type,
            "message": alert.message,
            "severity": alert.severity.value,
            "service": alert.service,
            "created_at": alert.created_at.isoformat(),
            "metadata": alert.metadata
        }
        for alert in alerts
    ]

def get_alert_statistics() -> Dict[str, Any]:
    """Get alert statistics"""
    return alert_manager.get_alert_statistics()

def register_notification_handler(handler_name: str, handler_function: Callable):
    """Register a notification handler"""
    alert_manager.register_notification_handler(handler_name, handler_function)

def check_alert_rules(metrics: Dict[str, Any]):
    """Check alert rules against metrics"""
    alert_manager.check_alert_rules(metrics)

# Initialize logging
logger.info("✅ Shared alert system initialized")