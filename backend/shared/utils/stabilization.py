# backend/shared/utils/stabilization.py
"""
System Stabilization Utilities for NAVA
Tools for system stability, recovery, and emergency mode management
"""

import logging
import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class SystemStatus(Enum):
    """System status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class RecoveryAction(Enum):
    """Recovery action types"""
    RESTART_SERVICE = "restart_service"
    CLEAR_CACHE = "clear_cache"
    REDUCE_LOAD = "reduce_load"
    EMERGENCY_MODE = "emergency_mode"
    SHUTDOWN_GRACEFUL = "shutdown_graceful"

@dataclass
class SystemMetrics:
    """Current system metrics"""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    response_time_avg: float
    error_rate_percent: float
    active_connections: int
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class StabilityThresholds:
    """System stability thresholds"""
    cpu_warning: float = 70.0
    cpu_critical: float = 85.0
    memory_warning: float = 75.0
    memory_critical: float = 90.0
    disk_warning: float = 80.0
    disk_critical: float = 95.0
    response_time_warning: float = 3.0
    response_time_critical: float = 5.0
    error_rate_warning: float = 5.0
    error_rate_critical: float = 10.0

@dataclass
class RecoveryPlan:
    """System recovery plan"""
    trigger_threshold: float
    actions: List[RecoveryAction]
    cooldown_minutes: int = 5
    max_retries: int = 3
    escalation_delay_minutes: int = 10

class SystemStabilizer:
    """
    System stabilization and recovery management
    Monitors system health and triggers recovery actions
    """
    
    def __init__(self):
        self.thresholds = StabilityThresholds()
        self.is_monitoring = False
        self.emergency_mode = False
        self.last_metrics: Optional[SystemMetrics] = None
        
        # Recovery tracking
        self.recovery_history: List[Dict[str, Any]] = []
        self.recovery_plans: Dict[SystemStatus, RecoveryPlan] = self._create_recovery_plans()
        self.last_recovery_time: Optional[datetime] = None
        
        # Callbacks for external systems
        self.status_change_callbacks: List[Callable] = []
        self.recovery_callbacks: List[Callable] = []
        
        logger.info("✅ System Stabilizer initialized")
    
    def _create_recovery_plans(self) -> Dict[SystemStatus, RecoveryPlan]:
        """Create default recovery plans"""
        return {
            SystemStatus.WARNING: RecoveryPlan(
                trigger_threshold=70.0,
                actions=[RecoveryAction.CLEAR_CACHE],
                cooldown_minutes=2,
                max_retries=2
            ),
            SystemStatus.DEGRADED: RecoveryPlan(
                trigger_threshold=80.0,
                actions=[RecoveryAction.REDUCE_LOAD, RecoveryAction.CLEAR_CACHE],
                cooldown_minutes=5,
                max_retries=3
            ),
            SystemStatus.CRITICAL: RecoveryPlan(
                trigger_threshold=90.0,
                actions=[RecoveryAction.EMERGENCY_MODE, RecoveryAction.RESTART_SERVICE],
                cooldown_minutes=10,
                max_retries=2,
                escalation_delay_minutes=5
            ),
            SystemStatus.EMERGENCY: RecoveryPlan(
                trigger_threshold=95.0,
                actions=[RecoveryAction.EMERGENCY_MODE, RecoveryAction.SHUTDOWN_GRACEFUL],
                cooldown_minutes=15,
                max_retries=1
            )
        }
    
    def get_current_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Mock response time and error rate (would be replaced with actual metrics)
            response_time_avg = 1.5  # Would come from monitoring system
            error_rate_percent = 2.0  # Would come from monitoring system
            
            # Active connections (estimate)
            connections = len(psutil.net_connections())
            
            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_percent=disk_percent,
                response_time_avg=response_time_avg,
                error_rate_percent=error_rate_percent,
                active_connections=connections
            )
            
            self.last_metrics = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Failed to get system metrics: {e}")
            # Return safe defaults
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0, 
                disk_percent=0.0,
                response_time_avg=0.0,
                error_rate_percent=0.0,
                active_connections=0
            )
    
    def assess_system_status(self, metrics: Optional[SystemMetrics] = None) -> SystemStatus:
        """Assess current system status"""
        if not metrics:
            metrics = self.get_current_metrics()
        
        # Emergency conditions
        if (metrics.cpu_percent > self.thresholds.cpu_critical or
            metrics.memory_percent > self.thresholds.memory_critical or
            metrics.error_rate_percent > self.thresholds.error_rate_critical):
            return SystemStatus.EMERGENCY
        
        # Critical conditions
        if (metrics.cpu_percent > self.thresholds.cpu_critical * 0.9 or
            metrics.memory_percent > self.thresholds.memory_critical * 0.9 or
            metrics.response_time_avg > self.thresholds.response_time_critical):
            return SystemStatus.CRITICAL
        
        # Degraded conditions
        if (metrics.cpu_percent > self.thresholds.cpu_warning or
            metrics.memory_percent > self.thresholds.memory_warning or
            metrics.disk_percent > self.thresholds.disk_warning):
            return SystemStatus.DEGRADED
        
        # Warning conditions
        if (metrics.cpu_percent > self.thresholds.cpu_warning * 0.8 or
            metrics.memory_percent > self.thresholds.memory_warning * 0.8 or
            metrics.response_time_avg > self.thresholds.response_time_warning):
            return SystemStatus.WARNING
        
        return SystemStatus.HEALTHY
    
    def enable_emergency_mode(self):
        """Enable emergency mode"""
        if not self.emergency_mode:
            self.emergency_mode = True
            logger.warning("🚨 EMERGENCY MODE ENABLED")
            
            # Notify callbacks
            for callback in self.status_change_callbacks:
                try:
                    callback(SystemStatus.EMERGENCY, {"emergency_mode": True})
                except Exception as e:
                    logger.error(f"❌ Status change callback failed: {e}")
    
    def disable_emergency_mode(self):
        """Disable emergency mode"""
        if self.emergency_mode:
            self.emergency_mode = False
            logger.info("✅ Emergency mode disabled")
            
            # Notify callbacks
            for callback in self.status_change_callbacks:
                try:
                    callback(SystemStatus.HEALTHY, {"emergency_mode": False})
                except Exception as e:
                    logger.error(f"❌ Status change callback failed: {e}")
    
    def execute_recovery_action(self, action: RecoveryAction) -> bool:
        """Execute a recovery action"""
        try:
            logger.info(f"🔧 Executing recovery action: {action.value}")
            
            if action == RecoveryAction.CLEAR_CACHE:
                return self._clear_cache()
            elif action == RecoveryAction.REDUCE_LOAD:
                return self._reduce_load()
            elif action == RecoveryAction.EMERGENCY_MODE:
                self.enable_emergency_mode()
                return True
            elif action == RecoveryAction.RESTART_SERVICE:
                return self._restart_service()
            elif action == RecoveryAction.SHUTDOWN_GRACEFUL:
                return self._shutdown_graceful()
            else:
                logger.warning(f"⚠️ Unknown recovery action: {action}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Recovery action {action.value} failed: {e}")
            return False
    
    def _clear_cache(self) -> bool:
        """Clear system caches"""
        try:
            # Clear memory caches (this would interface with actual cache systems)
            logger.info("🧹 Clearing system caches...")
            
            # Simulate cache clearing
            time.sleep(0.1)
            
            # Record action
            self._record_recovery_action(RecoveryAction.CLEAR_CACHE, True)
            return True
            
        except Exception as e:
            logger.error(f"❌ Cache clearing failed: {e}")
            self._record_recovery_action(RecoveryAction.CLEAR_CACHE, False, str(e))
            return False
    
    def _reduce_load(self) -> bool:
        """Reduce system load"""
        try:
            logger.info("⚡ Reducing system load...")
            
            # This would implement actual load reduction strategies:
            # - Throttle requests
            # - Reduce AI model complexity
            # - Defer non-critical tasks
            # - Scale down resource usage
            
            # Simulate load reduction
            time.sleep(0.1)
            
            self._record_recovery_action(RecoveryAction.REDUCE_LOAD, True)
            return True
            
        except Exception as e:
            logger.error(f"❌ Load reduction failed: {e}")
            self._record_recovery_action(RecoveryAction.REDUCE_LOAD, False, str(e))
            return False
    
    def _restart_service(self) -> bool:
        """Restart critical services"""
        try:
            logger.warning("🔄 Restarting services...")
            
            # This would implement actual service restart
            # For now, just simulate
            time.sleep(0.5)
            
            self._record_recovery_action(RecoveryAction.RESTART_SERVICE, True)
            return True
            
        except Exception as e:
            logger.error(f"❌ Service restart failed: {e}")
            self._record_recovery_action(RecoveryAction.RESTART_SERVICE, False, str(e))
            return False
    
    def _shutdown_graceful(self) -> bool:
        """Graceful system shutdown"""
        try:
            logger.critical("🛑 Initiating graceful shutdown...")
            
            # This would implement actual graceful shutdown
            # - Save state
            # - Close connections
            # - Stop services
            # - Exit cleanly
            
            self._record_recovery_action(RecoveryAction.SHUTDOWN_GRACEFUL, True)
            return True
            
        except Exception as e:
            logger.error(f"❌ Graceful shutdown failed: {e}")
            self._record_recovery_action(RecoveryAction.SHUTDOWN_GRACEFUL, False, str(e))
            return False
    
    def _record_recovery_action(self, action: RecoveryAction, success: bool, error: str = None):
        """Record recovery action in history"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "action": action.value,
            "success": success,
            "error": error,
            "metrics": self.last_metrics.__dict__ if self.last_metrics else None
        }
        
        self.recovery_history.append(record)
        
        # Keep only last 100 records
        if len(self.recovery_history) > 100:
            self.recovery_history = self.recovery_history[-100:]
        
        # Notify recovery callbacks
        for callback in self.recovery_callbacks:
            try:
                callback(action, success, error)
            except Exception as e:
                logger.error(f"❌ Recovery callback failed: {e}")
    
    def trigger_recovery(self, status: SystemStatus) -> bool:
        """Trigger recovery based on system status"""
        if status not in self.recovery_plans:
            logger.warning(f"⚠️ No recovery plan for status: {status}")
            return False
        
        plan = self.recovery_plans[status]
        
        # Check cooldown
        if self.last_recovery_time:
            time_since_last = datetime.now() - self.last_recovery_time
            if time_since_last < timedelta(minutes=plan.cooldown_minutes):
                logger.info(f"⏱️ Recovery cooldown active, skipping...")
                return False
        
        self.last_recovery_time = datetime.now()
        
        # Execute recovery actions
        success_count = 0
        for action in plan.actions:
            if self.execute_recovery_action(action):
                success_count += 1
        
        success = success_count == len(plan.actions)
        
        if success:
            logger.info(f"✅ Recovery plan for {status.value} completed successfully")
        else:
            logger.error(f"❌ Recovery plan for {status.value} partially failed ({success_count}/{len(plan.actions)})")
        
        return success
    
    def start_monitoring(self, interval_seconds: int = 30):
        """Start continuous system monitoring"""
        if self.is_monitoring:
            logger.warning("⚠️ Monitoring already active")
            return
        
        self.is_monitoring = True
        logger.info(f"🔍 Starting system monitoring (interval: {interval_seconds}s)")
        
        def monitor_loop():
            while self.is_monitoring:
                try:
                    # Get current metrics
                    metrics = self.get_current_metrics()
                    
                    # Assess system status
                    current_status = self.assess_system_status(metrics)
                    
                    # Log status if changed or critical
                    if current_status != SystemStatus.HEALTHY:
                        logger.warning(f"⚠️ System status: {current_status.value}")
                        logger.info(f"   CPU: {metrics.cpu_percent:.1f}%, Memory: {metrics.memory_percent:.1f}%")
                        logger.info(f"   Response time: {metrics.response_time_avg:.2f}s, Error rate: {metrics.error_rate_percent:.1f}%")
                    
                    # Trigger recovery if needed
                    if current_status in [SystemStatus.DEGRADED, SystemStatus.CRITICAL, SystemStatus.EMERGENCY]:
                        self.trigger_recovery(current_status)
                    
                    time.sleep(interval_seconds)
                    
                except Exception as e:
                    logger.error(f"❌ Monitoring loop error: {e}")
                    time.sleep(interval_seconds)
        
        # Start monitoring in background thread
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        if self.is_monitoring:
            self.is_monitoring = False
            logger.info("✅ System monitoring stopped")
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Get comprehensive system health report"""
        metrics = self.get_current_metrics()
        status = self.assess_system_status(metrics)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "status": status.value,
            "emergency_mode": self.emergency_mode,
            "metrics": {
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
                "disk_percent": metrics.disk_percent,
                "response_time_avg": metrics.response_time_avg,
                "error_rate_percent": metrics.error_rate_percent,
                "active_connections": metrics.active_connections
            },
            "thresholds": {
                "cpu_warning": self.thresholds.cpu_warning,
                "cpu_critical": self.thresholds.cpu_critical,
                "memory_warning": self.thresholds.memory_warning,
                "memory_critical": self.thresholds.memory_critical,
                "response_time_warning": self.thresholds.response_time_warning,
                "response_time_critical": self.thresholds.response_time_critical
            },
            "recovery_history": self.recovery_history[-10:],  # Last 10 actions
            "is_monitoring": self.is_monitoring,
            "last_recovery": self.last_recovery_time.isoformat() if self.last_recovery_time else None
        }
    
    def add_status_change_callback(self, callback: Callable):
        """Add callback for status changes"""
        self.status_change_callbacks.append(callback)
    
    def add_recovery_callback(self, callback: Callable):
        """Add callback for recovery actions"""
        self.recovery_callbacks.append(callback)
    
    def update_thresholds(self, new_thresholds: Dict[str, float]):
        """Update system thresholds"""
        for key, value in new_thresholds.items():
            if hasattr(self.thresholds, key):
                setattr(self.thresholds, key, value)
                logger.info(f"✅ Updated threshold {key} = {value}")

class EmergencyManager:
    """Emergency mode management"""
    
    def __init__(self):
        self.emergency_active = False
        self.emergency_start_time: Optional[datetime] = None
        self.emergency_features_disabled: List[str] = []
        
    def activate_emergency_mode(self, reason: str = "System instability detected"):
        """Activate emergency mode"""
        if not self.emergency_active:
            self.emergency_active = True
            self.emergency_start_time = datetime.now()
            
            # Disable advanced features
            self.emergency_features_disabled = [
                "advanced_ai_features",
                "complex_workflows", 
                "learning_system",
                "multi_agent_processing",
                "heavy_computations"
            ]
            
            logger.critical(f"🚨 EMERGENCY MODE ACTIVATED: {reason}")
            logger.info(f"   Disabled features: {', '.join(self.emergency_features_disabled)}")
    
    def deactivate_emergency_mode(self):
        """Deactivate emergency mode"""
        if self.emergency_active:
            duration = datetime.now() - self.emergency_start_time
            
            self.emergency_active = False
            self.emergency_start_time = None
            self.emergency_features_disabled = []
            
            logger.info(f"✅ Emergency mode deactivated (was active for {duration})")
    
    def is_feature_available(self, feature: str) -> bool:
        """Check if feature is available (not disabled by emergency mode)"""
        return not self.emergency_active or feature not in self.emergency_features_disabled
    
    def get_emergency_status(self) -> Dict[str, Any]:
        """Get current emergency status"""
        return {
            "active": self.emergency_active,
            "start_time": self.emergency_start_time.isoformat() if self.emergency_start_time else None,
            "duration_minutes": int((datetime.now() - self.emergency_start_time).total_seconds() / 60) if self.emergency_start_time else 0,
            "disabled_features": self.emergency_features_disabled
        }

class StabilityHelper:
    """Helper functions for system stability"""
    
    @staticmethod
    def check_memory_usage() -> Dict[str, float]:
        """Check current memory usage"""
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percent": memory.percent
        }
    
    @staticmethod
    def check_disk_usage(path: str = "/") -> Dict[str, float]:
        """Check disk usage for given path"""
        disk = psutil.disk_usage(path)
        return {
            "total_gb": disk.total / (1024**3),
            "used_gb": disk.used / (1024**3),
            "free_gb": disk.free / (1024**3),
            "percent": (disk.used / disk.total) * 100
        }
    
    @staticmethod
    def check_cpu_usage(interval: float = 1.0) -> Dict[str, float]:
        """Check CPU usage"""
        cpu_percent = psutil.cpu_percent(interval=interval)
        cpu_count = psutil.cpu_count()
        
        return {
            "percent": cpu_percent,
            "logical_cores": cpu_count,
            "physical_cores": psutil.cpu_count(logical=False)
        }
    
    @staticmethod
    def check_network_connections() -> Dict[str, int]:
        """Check network connections"""
        connections = psutil.net_connections()
        
        status_counts = {}
        for conn in connections:
            status = conn.status if hasattr(conn, 'status') else 'UNKNOWN'
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total": len(connections),
            "by_status": status_counts
        }
    
    @staticmethod
    def force_garbage_collection():
        """Force garbage collection to free memory"""
        import gc
        collected = gc.collect()
        logger.info(f"🧹 Garbage collection freed {collected} objects")
        return collected

# Global instances
system_stabilizer = SystemStabilizer()
emergency_manager = EmergencyManager()

# Convenience functions
def get_system_health() -> Dict[str, Any]:
    """Get current system health"""
    return system_stabilizer.get_system_health_report()

def start_stability_monitoring(interval: int = 30):
    """Start system stability monitoring"""
    system_stabilizer.start_monitoring(interval)

def stop_stability_monitoring():
    """Stop system stability monitoring"""
    system_stabilizer.stop_monitoring()

def enable_emergency_mode(reason: str = "Manual activation"):
    """Enable emergency mode"""
    emergency_manager.activate_emergency_mode(reason)
    system_stabilizer.enable_emergency_mode()

def disable_emergency_mode():
    """Disable emergency mode"""
    emergency_manager.deactivate_emergency_mode()
    system_stabilizer.disable_emergency_mode()

def is_emergency_mode_active() -> bool:
    """Check if emergency mode is active"""
    return emergency_manager.emergency_active

def is_feature_available(feature: str) -> bool:
    """Check if feature is available"""
    return emergency_manager.is_feature_available(feature)

def check_system_resources() -> Dict[str, Any]:
    """Quick system resource check"""
    return {
        "memory": StabilityHelper.check_memory_usage(),
        "disk": StabilityHelper.check_disk_usage(),
        "cpu": StabilityHelper.check_cpu_usage(),
        "network": StabilityHelper.check_network_connections(),
        "timestamp": datetime.now().isoformat()
    }

def emergency_recovery():
    """Emergency recovery procedure"""
    logger.warning("🚨 Starting emergency recovery...")
    
    # Enable emergency mode
    enable_emergency_mode("Emergency recovery triggered")
    
    # Clear caches
    system_stabilizer._clear_cache()
    
    # Reduce load
    system_stabilizer._reduce_load()
    
    # Force garbage collection
    StabilityHelper.force_garbage_collection()
    
    # Get updated status
    health = get_system_health()
    logger.info(f"🏥 Post-recovery status: {health['status']}")
    
    return health

# Export main classes and functions
__all__ = [
    'SystemStatus', 'RecoveryAction', 'SystemMetrics', 'StabilityThresholds',
    'SystemStabilizer', 'EmergencyManager', 'StabilityHelper',
    'system_stabilizer', 'emergency_manager',
    'get_system_health', 'start_stability_monitoring', 'stop_stability_monitoring',
    'enable_emergency_mode', 'disable_emergency_mode', 'is_emergency_mode_active',
    'is_feature_available', 'check_system_resources', 'emergency_recovery'
]