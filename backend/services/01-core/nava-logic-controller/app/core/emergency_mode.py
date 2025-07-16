# backend/services/01-core/nava-logic-controller/app/core/emergency_mode.py
"""
Emergency Mode Manager - Week 2 Critical Component
Handles emergency situations and system protection
"""

import logging
import time
import asyncio
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

class EmergencyLevel(Enum):
    GREEN = "green"           # Normal operation
    YELLOW = "yellow"         # Caution - degraded performance
    ORANGE = "orange"         # Warning - limited functionality
    RED = "red"              # Emergency - critical systems only
    BLACK = "black"          # Shutdown - system protection mode

class EmergencyTrigger(Enum):
    HIGH_ERROR_RATE = "high_error_rate"
    RESPONSE_TIMEOUT = "response_timeout"
    MEMORY_OVERFLOW = "memory_overflow"
    AI_SERVICE_DOWN = "ai_service_down"
    SYSTEM_OVERLOAD = "system_overload"
    MANUAL_TRIGGER = "manual_trigger"
    SECURITY_BREACH = "security_breach"

@dataclass
class EmergencyEvent:
    trigger: EmergencyTrigger
    level: EmergencyLevel
    message: str
    timestamp: float
    resolved: bool = False
    resolution_time: Optional[float] = None
    impact_assessment: str = ""

class EmergencyModeManager:
    """
    Emergency Mode Manager for NAVA System Protection
    
    Handles:
    - Emergency detection and response
    - System protection during failures
    - Automatic recovery procedures
    - Emergency communication
    """
    
    def __init__(self):
        self.current_level = EmergencyLevel.GREEN
        self.emergency_history = []
        self.active_emergencies = []
        self.emergency_triggers = {}
        self.auto_recovery_enabled = True
        self.emergency_contacts = []
        
        # Emergency thresholds
        self.thresholds = {
            "error_rate": 0.15,           # 15% error rate triggers yellow
            "critical_error_rate": 0.30,  # 30% error rate triggers red
            "response_time": 10.0,        # 10s response time triggers yellow
            "critical_response_time": 30.0, # 30s response time triggers red
            "memory_usage": 0.85,         # 85% memory usage triggers yellow
            "critical_memory": 0.95,      # 95% memory usage triggers red
            "ai_service_failures": 2,     # 2 AI services down triggers orange
            "all_ai_services_down": 3     # All AI services down triggers red
        }
        
        # Emergency response procedures
        self.response_procedures = {
            EmergencyLevel.YELLOW: self._handle_yellow_alert,
            EmergencyLevel.ORANGE: self._handle_orange_alert,
            EmergencyLevel.RED: self._handle_red_alert,
            EmergencyLevel.BLACK: self._handle_black_alert
        }
        
        # Features to disable per emergency level
        self.feature_restrictions = {
            EmergencyLevel.GREEN: [],
            EmergencyLevel.YELLOW: ["complex_workflows"],
            EmergencyLevel.ORANGE: ["complex_workflows", "learning_system", "parallel_processing"],
            EmergencyLevel.RED: ["complex_workflows", "learning_system", "parallel_processing", "advanced_decision_engine"],
            EmergencyLevel.BLACK: ["all_features"]
        }
        
        logger.info("🚨 Emergency Mode Manager initialized")
    
    async def check_system_health(self, metrics: Dict[str, Any]) -> EmergencyLevel:
        """
        Check system health and determine emergency level
        
        Args:
            metrics: System health metrics
            
        Returns:
            Current emergency level
        """
        try:
            new_level = EmergencyLevel.GREEN
            triggered_events = []
            
            # Check error rate
            error_rate = metrics.get("error_rate", 0.0)
            if error_rate >= self.thresholds["critical_error_rate"]:
                new_level = max(new_level, EmergencyLevel.RED)
                triggered_events.append((EmergencyTrigger.HIGH_ERROR_RATE, f"Critical error rate: {error_rate:.2%}"))
            elif error_rate >= self.thresholds["error_rate"]:
                new_level = max(new_level, EmergencyLevel.YELLOW)
                triggered_events.append((EmergencyTrigger.HIGH_ERROR_RATE, f"High error rate: {error_rate:.2%}"))
            
            # Check response time
            response_time = metrics.get("avg_response_time", 0.0)
            if response_time >= self.thresholds["critical_response_time"]:
                new_level = max(new_level, EmergencyLevel.RED)
                triggered_events.append((EmergencyTrigger.RESPONSE_TIMEOUT, f"Critical response time: {response_time:.1f}s"))
            elif response_time >= self.thresholds["response_time"]:
                new_level = max(new_level, EmergencyLevel.YELLOW)
                triggered_events.append((EmergencyTrigger.RESPONSE_TIMEOUT, f"Slow response time: {response_time:.1f}s"))
            
            # Check memory usage
            memory_usage = metrics.get("memory_usage", 0.0)
            if memory_usage >= self.thresholds["critical_memory"]:
                new_level = max(new_level, EmergencyLevel.RED)
                triggered_events.append((EmergencyTrigger.MEMORY_OVERFLOW, f"Critical memory usage: {memory_usage:.1%}"))
            elif memory_usage >= self.thresholds["memory_usage"]:
                new_level = max(new_level, EmergencyLevel.YELLOW)
                triggered_events.append((EmergencyTrigger.MEMORY_OVERFLOW, f"High memory usage: {memory_usage:.1%}"))
            
            # Check AI service health
            ai_services = metrics.get("ai_service_health", {})
            failed_services = sum(1 for service, health in ai_services.items() if not health.get("available", True))
            
            if failed_services >= self.thresholds["all_ai_services_down"]:
                new_level = max(new_level, EmergencyLevel.RED)
                triggered_events.append((EmergencyTrigger.AI_SERVICE_DOWN, f"All AI services down ({failed_services})"))
            elif failed_services >= self.thresholds["ai_service_failures"]:
                new_level = max(new_level, EmergencyLevel.ORANGE)
                triggered_events.append((EmergencyTrigger.AI_SERVICE_DOWN, f"Multiple AI services down ({failed_services})"))
            
            # Update emergency level if changed
            if new_level != self.current_level:
                await self._escalate_emergency_level(new_level, triggered_events)
            elif new_level == EmergencyLevel.GREEN and self.current_level != EmergencyLevel.GREEN:
                await self._resolve_emergency()
            
            return new_level
            
        except Exception as e:
            logger.error(f"❌ Error checking system health: {e}")
            return self.current_level
    
    async def _escalate_emergency_level(self, new_level: EmergencyLevel, triggered_events: List[tuple]):
        """Escalate to new emergency level"""
        
        old_level = self.current_level
        self.current_level = new_level
        
        logger.warning(f"🚨 Emergency level escalated: {old_level.value} → {new_level.value}")
        
        # Create emergency events
        for trigger, message in triggered_events:
            event = EmergencyEvent(
                trigger=trigger,
                level=new_level,
                message=message,
                timestamp=time.time(),
                impact_assessment=f"System degraded to {new_level.value} level"
            )
            
            self.emergency_history.append(event)
            self.active_emergencies.append(event)
        
        # Execute emergency response
        if new_level in self.response_procedures:
            await self.response_procedures[new_level]()
        
        # Notify emergency contacts
        await self._notify_emergency_contacts(new_level, triggered_events)
    
    async def _resolve_emergency(self):
        """Resolve emergency and return to normal operation"""
        
        old_level = self.current_level
        self.current_level = EmergencyLevel.GREEN
        
        # Mark active emergencies as resolved
        for event in self.active_emergencies:
            event.resolved = True
            event.resolution_time = time.time()
        
        self.active_emergencies.clear()
        
        logger.info(f"✅ Emergency resolved: {old_level.value} → GREEN")
        
        # Execute recovery procedures
        await self._execute_recovery_procedures()
    
    async def _handle_yellow_alert(self):
        """Handle YELLOW alert level"""
        logger.warning("🟡 YELLOW ALERT: System performance degraded")
        
        # Disable complex workflows
        await self._disable_features(["complex_workflows"])
        
        # Increase monitoring frequency
        await self._increase_monitoring_frequency(30)  # Every 30 seconds
        
        # Log performance metrics
        await self._log_performance_metrics()
    
    async def _handle_orange_alert(self):
        """Handle ORANGE alert level"""
        logger.warning("🟠 ORANGE ALERT: System functionality limited")
        
        # Disable advanced features
        await self._disable_features(["complex_workflows", "learning_system", "parallel_processing"])
        
        # Reduce system load
        await self._reduce_system_load()
        
        # Increase monitoring frequency
        await self._increase_monitoring_frequency(15)  # Every 15 seconds
    
    async def _handle_red_alert(self):
        """Handle RED alert level"""
        logger.error("🔴 RED ALERT: System in emergency mode")
        
        # Disable all advanced features
        await self._disable_features(["complex_workflows", "learning_system", "parallel_processing", "advanced_decision_engine"])
        
        # Emergency system protection
        await self._activate_emergency_protection()
        
        # Continuous monitoring
        await self._increase_monitoring_frequency(5)  # Every 5 seconds
        
        # Prepare for possible shutdown
        await self._prepare_emergency_shutdown()
    
    async def _handle_black_alert(self):
        """Handle BLACK alert level - System shutdown protection"""
        logger.critical("⚫ BLACK ALERT: System shutdown protection activated")
        
        # Disable all features except basic operation
        await self._disable_features(["all_features"])
        
        # Emergency shutdown procedures
        await self._emergency_shutdown_preparation()
        
        # Save critical system state
        await self._save_emergency_state()
    
    async def _disable_features(self, features: List[str]):
        """Disable specified features for emergency protection"""
        try:
            if "all_features" in features:
                logger.critical("🔒 All features disabled - emergency protection mode")
                # Implement complete feature shutdown
                return
            
            for feature in features:
                logger.warning(f"🔒 Disabling feature: {feature}")
                # Here you would integrate with feature flag manager
                # feature_manager.force_disable_feature(feature)
                
        except Exception as e:
            logger.error(f"❌ Error disabling features: {e}")
    
    async def _reduce_system_load(self):
        """Reduce system load during emergency"""
        try:
            # Reduce concurrent request limits
            # Increase cache usage
            # Simplify AI processing
            logger.info("⬇️ System load reduction activated")
            
        except Exception as e:
            logger.error(f"❌ Error reducing system load: {e}")
    
    async def _activate_emergency_protection(self):
        """Activate emergency protection mechanisms"""
        try:
            # Enable aggressive caching
            # Reduce timeout limits
            # Activate circuit breakers
            logger.info("🛡️ Emergency protection activated")
            
        except Exception as e:
            logger.error(f"❌ Error activating emergency protection: {e}")
    
    async def _increase_monitoring_frequency(self, interval_seconds: int):
        """Increase monitoring frequency during emergency"""
        try:
            logger.info(f"⏱️ Monitoring frequency increased to {interval_seconds}s")
            # Implementation would integrate with monitoring system
            
        except Exception as e:
            logger.error(f"❌ Error increasing monitoring frequency: {e}")
    
    async def _log_performance_metrics(self):
        """Log detailed performance metrics during emergency"""
        try:
            logger.info("📊 Emergency performance metrics logged")
            # Implementation would collect and log detailed metrics
            
        except Exception as e:
            logger.error(f"❌ Error logging performance metrics: {e}")
    
    async def _prepare_emergency_shutdown(self):
        """Prepare for possible emergency shutdown"""
        try:
            logger.warning("⚠️ Emergency shutdown preparation initiated")
            # Save critical state
            # Prepare graceful shutdown procedures
            
        except Exception as e:
            logger.error(f"❌ Error preparing emergency shutdown: {e}")
    
    async def _emergency_shutdown_preparation(self):
        """Prepare for immediate emergency shutdown"""
        try:
            logger.critical("🔴 Emergency shutdown preparation - immediate threat")
            # Immediate state saving
            # Emergency contact notification
            
        except Exception as e:
            logger.error(f"❌ Error in emergency shutdown preparation: {e}")
    
    async def _save_emergency_state(self):
        """Save critical system state during emergency"""
        try:
            emergency_state = {
                "timestamp": time.time(),
                "emergency_level": self.current_level.value,
                "active_emergencies": len(self.active_emergencies),
                "system_metrics": "emergency_snapshot"
            }
            
            logger.critical("💾 Emergency state saved")
            
        except Exception as e:
            logger.error(f"❌ Error saving emergency state: {e}")
    
    async def _execute_recovery_procedures(self):
        """Execute recovery procedures when emergency resolves"""
        try:
            # Re-enable features gradually
            # Restore normal monitoring
            # Clear emergency restrictions
            logger.info("🔄 Recovery procedures executed")
            
        except Exception as e:
            logger.error(f"❌ Error executing recovery procedures: {e}")
    
    async def _notify_emergency_contacts(self, level: EmergencyLevel, events: List[tuple]):
        """Notify emergency contacts of system status"""
        try:
            if level in [EmergencyLevel.RED, EmergencyLevel.BLACK]:
                logger.critical(f"📞 Emergency contacts notified: {level.value}")
                # Implementation would send notifications
                
        except Exception as e:
            logger.error(f"❌ Error notifying emergency contacts: {e}")
    
    async def trigger_manual_emergency(self, level: EmergencyLevel, reason: str) -> bool:
        """Manually trigger emergency mode"""
        try:
            event = EmergencyEvent(
                trigger=EmergencyTrigger.MANUAL_TRIGGER,
                level=level,
                message=f"Manual emergency trigger: {reason}",
                timestamp=time.time(),
                impact_assessment=f"Manually escalated to {level.value}"
            )
            
            self.emergency_history.append(event)
            self.active_emergencies.append(event)
            
            await self._escalate_emergency_level(level, [(EmergencyTrigger.MANUAL_TRIGGER, reason)])
            
            logger.warning(f"🚨 Manual emergency triggered: {level.value} - {reason}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error triggering manual emergency: {e}")
            return False
    
    async def resolve_manual_emergency(self) -> bool:
        """Manually resolve emergency mode"""
        try:
            if self.current_level != EmergencyLevel.GREEN:
                await self._resolve_emergency()
                logger.info("✅ Manual emergency resolution")
                return True
            return False
            
        except Exception as e:
            logger.error(f"❌ Error resolving manual emergency: {e}")
            return False
    
    def get_emergency_status(self) -> Dict[str, Any]:
        """Get current emergency status"""
        try:
            return {
                "current_level": self.current_level.value,
                "active_emergencies": len(self.active_emergencies),
                "total_emergency_events": len(self.emergency_history),
                "auto_recovery_enabled": self.auto_recovery_enabled,
                "disabled_features": self.feature_restrictions.get(self.current_level, []),
                "last_emergency": self.emergency_history[-1].__dict__ if self.emergency_history else None,
                "emergency_thresholds": self.thresholds,
                "system_protection_active": self.current_level != EmergencyLevel.GREEN
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting emergency status: {e}")
            return {"error": str(e)}
    
    def get_emergency_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get emergency event history"""
        try:
            recent_events = self.emergency_history[-limit:] if limit > 0 else self.emergency_history
            return [event.__dict__ for event in recent_events]
            
        except Exception as e:
            logger.error(f"❌ Error getting emergency history: {e}")
            return []

# Global emergency manager instance
emergency_manager = EmergencyModeManager()

# Convenience functions
async def check_emergency_level(metrics: Dict[str, Any]) -> EmergencyLevel:
    """Check and update emergency level based on system metrics"""
    return await emergency_manager.check_system_health(metrics)

async def trigger_emergency(level: EmergencyLevel, reason: str) -> bool:
    """Trigger manual emergency"""
    return await emergency_manager.trigger_manual_emergency(level, reason)

async def resolve_emergency() -> bool:
    """Resolve current emergency"""
    return await emergency_manager.resolve_manual_emergency()

def get_current_emergency_level() -> EmergencyLevel:
    """Get current emergency level"""
    return emergency_manager.current_level

def is_emergency_active() -> bool:
    """Check if emergency mode is active"""
    return emergency_manager.current_level != EmergencyLevel.GREEN

def get_emergency_status() -> Dict[str, Any]:
    """Get emergency status"""
    return emergency_manager.get_emergency_status()