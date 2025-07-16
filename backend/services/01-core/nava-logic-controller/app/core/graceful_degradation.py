# backend/services/01-core/nava-logic-controller/app/core/graceful_degradation.py
"""
Graceful Degradation System - ระบบลดประสิทธิภาพอย่างสง่างาม
Week 1: Critical Component for Handling System Failures

Handles automatic system degradation when services fail:
Full Intelligence → Smart Routing → Simple Routing → Emergency Mode
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class DegradationLevel(Enum):
    """System degradation levels from best to worst"""
    FULL_INTELLIGENCE = "full_intelligence"      # All features working
    SMART_ROUTING = "smart_routing"             # Pattern detection only
    SIMPLE_ROUTING = "simple_routing"           # Keyword matching only
    EMERGENCY_MODE = "emergency_mode"           # Single AI, basic routing
    OFFLINE_MODE = "offline_mode"               # Local processing only

class ServiceStatus(Enum):
    """Individual service status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    DOWN = "down"

@dataclass
class SystemHealth:
    """System health metrics"""
    overall_health: float          # 0.0 to 1.0
    ai_services_health: float      # AI services health
    core_services_health: float    # Core system health
    degradation_level: DegradationLevel
    active_services: List[str]
    failed_services: List[str]
    last_update: float

class GracefulDegradationManager:
    """Manages automatic system degradation based on service health"""
    
    def __init__(self):
        self.current_level = DegradationLevel.FULL_INTELLIGENCE
        self.service_status = {
            "gpt": ServiceStatus.HEALTHY,
            "claude": ServiceStatus.HEALTHY,
            "gemini": ServiceStatus.HEALTHY,
            "decision_engine": ServiceStatus.HEALTHY,
            "cache": ServiceStatus.HEALTHY,
            "database": ServiceStatus.HEALTHY
        }
        
        # Health thresholds for degradation
        self.degradation_thresholds = {
            DegradationLevel.FULL_INTELLIGENCE: 0.9,  # 90% health required
            DegradationLevel.SMART_ROUTING: 0.7,      # 70% health required
            DegradationLevel.SIMPLE_ROUTING: 0.5,     # 50% health required
            DegradationLevel.EMERGENCY_MODE: 0.3,     # 30% health required
            DegradationLevel.OFFLINE_MODE: 0.0        # Any health level
        }
        
        # Service weights for health calculation
        self.service_weights = {
            "gpt": 0.25,
            "claude": 0.25,
            "gemini": 0.25,
            "decision_engine": 0.15,
            "cache": 0.05,
            "database": 0.05
        }
        
        self.degradation_history = []
        self.last_health_check = time.time()
        self.auto_recovery_enabled = True
        
    def update_service_status(self, service_name: str, status: ServiceStatus, health_score: float = None):
        """Update individual service status"""
        old_status = self.service_status.get(service_name, ServiceStatus.DOWN)
        self.service_status[service_name] = status
        
        if old_status != status:
            logger.info(f"Service {service_name} status changed: {old_status.value} → {status.value}")
            
            # Trigger health assessment
            self._assess_system_health()
            
    def _assess_system_health(self) -> SystemHealth:
        """Assess overall system health and determine degradation level"""
        current_time = time.time()
        
        # Calculate weighted health score
        total_health = 0.0
        active_services = []
        failed_services = []
        
        for service, status in self.service_status.items():
            weight = self.service_weights.get(service, 0.1)
            
            if status == ServiceStatus.HEALTHY:
                service_health = 1.0
                active_services.append(service)
            elif status == ServiceStatus.DEGRADED:
                service_health = 0.6
                active_services.append(service)
            elif status == ServiceStatus.FAILING:
                service_health = 0.3
                failed_services.append(service)
            else:  # DOWN
                service_health = 0.0
                failed_services.append(service)
            
            total_health += service_health * weight
        
        # Calculate AI and core service health separately
        ai_services = ["gpt", "claude", "gemini"]
        ai_health = sum(
            (1.0 if self.service_status[svc] == ServiceStatus.HEALTHY else 
             0.6 if self.service_status[svc] == ServiceStatus.DEGRADED else
             0.3 if self.service_status[svc] == ServiceStatus.FAILING else 0.0)
            for svc in ai_services if svc in self.service_status
        ) / len(ai_services)
        
        core_services = ["decision_engine", "cache", "database"]
        core_health = sum(
            (1.0 if self.service_status[svc] == ServiceStatus.HEALTHY else 
             0.6 if self.service_status[svc] == ServiceStatus.DEGRADED else
             0.3 if self.service_status[svc] == ServiceStatus.FAILING else 0.0)
            for svc in core_services if svc in self.service_status
        ) / len(core_services)
        
        # Determine appropriate degradation level
        new_level = self._determine_degradation_level(total_health)
        
        # Auto-adjust degradation level if enabled
        if self.auto_recovery_enabled and new_level != self.current_level:
            self._change_degradation_level(new_level)
        
        health = SystemHealth(
            overall_health=total_health,
            ai_services_health=ai_health,
            core_services_health=core_health,
            degradation_level=self.current_level,
            active_services=active_services,
            failed_services=failed_services,
            last_update=current_time
        )
        
        self.last_health_check = current_time
        return health
    
    def _determine_degradation_level(self, health_score: float) -> DegradationLevel:
        """Determine appropriate degradation level based on health score"""
        if health_score >= self.degradation_thresholds[DegradationLevel.FULL_INTELLIGENCE]:
            return DegradationLevel.FULL_INTELLIGENCE
        elif health_score >= self.degradation_thresholds[DegradationLevel.SMART_ROUTING]:
            return DegradationLevel.SMART_ROUTING
        elif health_score >= self.degradation_thresholds[DegradationLevel.SIMPLE_ROUTING]:
            return DegradationLevel.SIMPLE_ROUTING
        elif health_score >= self.degradation_thresholds[DegradationLevel.EMERGENCY_MODE]:
            return DegradationLevel.EMERGENCY_MODE
        else:
            return DegradationLevel.OFFLINE_MODE
    
    def _change_degradation_level(self, new_level: DegradationLevel):
        """Change system degradation level"""
        old_level = self.current_level
        self.current_level = new_level
        
        # Record degradation history
        self.degradation_history.append({
            "timestamp": time.time(),
            "from_level": old_level.value,
            "to_level": new_level.value,
            "reason": "automatic_health_assessment"
        })
        
        # Keep only recent history (last 100 changes)
        if len(self.degradation_history) > 100:
            self.degradation_history = self.degradation_history[-100:]
        
        logger.warning(f"System degradation level changed: {old_level.value} → {new_level.value}")
        
    async def handle_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle request with appropriate degradation strategy
        
        Args:
            request_data: Original request data
            
        Returns:
            Processed request data with degradation strategy applied
        """
        try:
            if self.current_level == DegradationLevel.FULL_INTELLIGENCE:
                return await self._full_intelligence_mode(request_data)
            elif self.current_level == DegradationLevel.SMART_ROUTING:
                return await self._smart_routing_mode(request_data)
            elif self.current_level == DegradationLevel.SIMPLE_ROUTING:
                return await self._simple_routing_mode(request_data)
            elif self.current_level == DegradationLevel.EMERGENCY_MODE:
                return await self._emergency_mode(request_data)
            else:  # OFFLINE_MODE
                return await self._offline_mode(request_data)
                
        except Exception as e:
            logger.error(f"Error in degradation handler: {str(e)}")
            # Further degrade on error
            await self._degrade_further()
            return await self.handle_request(request_data)
    
    async def _full_intelligence_mode(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Full intelligence mode - all features available"""
        logger.debug("Processing request in FULL_INTELLIGENCE mode")
        
        return {
            **request_data,
            "processing_mode": "full_intelligence",
            "features_enabled": [
                "behavior_pattern_detection",
                "advanced_decision_making",
                "learning_system",
                "quality_validation",
                "multi_agent_workflows"
            ],
            "timeout_multiplier": 1.0,
            "quality_threshold": 0.8
        }
    
    async def _smart_routing_mode(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Smart routing mode - pattern detection only"""
        logger.debug("Processing request in SMART_ROUTING mode")
        
        return {
            **request_data,
            "processing_mode": "smart_routing",
            "features_enabled": [
                "behavior_pattern_detection",
                "basic_decision_making",
                "simple_caching"
            ],
            "timeout_multiplier": 0.8,
            "quality_threshold": 0.6,
            "fallback_strategy": "simple_routing"
        }
    
    async def _simple_routing_mode(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simple routing mode - keyword matching only"""
        logger.debug("Processing request in SIMPLE_ROUTING mode")
        
        # Simple keyword-based routing
        message = request_data.get("message", "").lower()
        
        if any(keyword in message for keyword in ["code", "python", "javascript", "programming"]):
            preferred_ai = "gpt"
        elif any(keyword in message for keyword in ["analyze", "analysis", "business", "strategy"]):
            preferred_ai = "claude"
        elif any(keyword in message for keyword in ["search", "research", "find", "image"]):
            preferred_ai = "gemini"
        else:
            preferred_ai = self._get_healthiest_ai()
        
        return {
            **request_data,
            "processing_mode": "simple_routing",
            "features_enabled": ["keyword_routing", "basic_caching"],
            "preferred_ai": preferred_ai,
            "timeout_multiplier": 0.6,
            "quality_threshold": 0.4,
            "fallback_strategy": "emergency_mode"
        }
    
    async def _emergency_mode(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Emergency mode - single AI with basic routing"""
        logger.debug("Processing request in EMERGENCY mode")
        
        # Use only the healthiest AI service
        available_ai = self._get_healthiest_ai()
        
        return {
            **request_data,
            "processing_mode": "emergency",
            "features_enabled": ["basic_routing"],
            "forced_ai": available_ai,
            "timeout_multiplier": 0.4,
            "quality_threshold": 0.2,
            "emergency_mode": True,
            "fallback_strategy": "offline_mode"
        }
    
    async def _offline_mode(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Offline mode - local processing only"""
        logger.debug("Processing request in OFFLINE mode")
        
        return {
            **request_data,
            "processing_mode": "offline",
            "features_enabled": ["offline_processing"],
            "response_template": "System is currently in maintenance mode. Limited functionality available.",
            "timeout_multiplier": 0.2,
            "quality_threshold": 0.1,
            "offline_mode": True
        }
    
    def _get_healthiest_ai(self) -> str:
        """Get the healthiest available AI service"""
        ai_services = ["gpt", "claude", "gemini"]
        
        # Sort by health status
        healthy_ais = [
            ai for ai in ai_services 
            if self.service_status.get(ai, ServiceStatus.DOWN) in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]
        ]
        
        if healthy_ais:
            return healthy_ais[0]  # Return first healthy AI
        
        # If no healthy AIs, return least unhealthy
        return min(ai_services, key=lambda ai: self.service_status.get(ai, ServiceStatus.DOWN).value)
    
    async def _degrade_further(self):
        """Degrade system further when errors occur"""
        current_levels = list(DegradationLevel)
        current_index = current_levels.index(self.current_level)
        
        if current_index < len(current_levels) - 1:
            new_level = current_levels[current_index + 1]
            logger.warning(f"Degrading further due to error: {self.current_level.value} → {new_level.value}")
            self._change_degradation_level(new_level)
        else:
            logger.error("Already at lowest degradation level - cannot degrade further")
    
    def force_degradation_level(self, level: DegradationLevel, reason: str = "manual"):
        """Manually force a specific degradation level"""
        old_level = self.current_level
        self.current_level = level
        
        self.degradation_history.append({
            "timestamp": time.time(),
            "from_level": old_level.value,
            "to_level": level.value,
            "reason": reason
        })
        
        logger.info(f"Manually forced degradation level: {old_level.value} → {level.value} (reason: {reason})")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and health metrics"""
        health = self._assess_system_health()
        
        return {
            "current_degradation_level": self.current_level.value,
            "overall_health": round(health.overall_health, 3),
            "ai_services_health": round(health.ai_services_health, 3),
            "core_services_health": round(health.core_services_health, 3),
            "active_services": health.active_services,
            "failed_services": health.failed_services,
            "service_status": {svc: status.value for svc, status in self.service_status.items()},
            "auto_recovery_enabled": self.auto_recovery_enabled,
            "last_health_check": self.last_health_check,
            "degradation_history_count": len(self.degradation_history),
            "available_levels": [level.value for level in DegradationLevel]
        }
    
    def enable_auto_recovery(self):
        """Enable automatic recovery to higher degradation levels"""
        self.auto_recovery_enabled = True
        logger.info("Auto-recovery enabled")
    
    def disable_auto_recovery(self):
        """Disable automatic recovery (manual control only)"""
        self.auto_recovery_enabled = False
        logger.info("Auto-recovery disabled - manual control only")

# Global instance
degradation_manager = GracefulDegradationManager()

# Convenience functions
async def handle_request_with_degradation(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle request with graceful degradation"""
    return await degradation_manager.handle_request(request_data)

def update_service_health(service_name: str, status: ServiceStatus, health_score: float = None):
    """Update service health status"""
    degradation_manager.update_service_status(service_name, status, health_score)

def get_degradation_status() -> Dict[str, Any]:
    """Get current degradation status"""
    return degradation_manager.get_system_status()

def force_degradation(level: DegradationLevel, reason: str = "manual"):
    """Force specific degradation level"""
    degradation_manager.force_degradation_level(level, reason)

def enable_auto_recovery():
    """Enable automatic recovery"""
    degradation_manager.enable_auto_recovery()

def disable_auto_recovery():
    """Disable automatic recovery"""
    degradation_manager.disable_auto_recovery()