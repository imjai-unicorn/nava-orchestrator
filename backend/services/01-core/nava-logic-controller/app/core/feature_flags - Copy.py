# backend/services/01-core/nava-logic-controller/app/core/feature_flags.py
"""
Progressive Feature Manager - Fixed AttributeError Version
🔴 WEEK 1 PRIORITY: Safely activate advanced features that are already coded
"""

import logging
import time
import asyncio
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FeatureState(Enum):
    DISABLED = "disabled"
    TESTING = "testing"
    PARTIAL = "partial"
    ENABLED = "enabled"
    DEPRECATED = "deprecated"

class StabilityLevel(Enum):
    UNSTABLE = "unstable"
    TESTING = "testing" 
    STABLE = "stable"
    PRODUCTION = "production"

@dataclass
class FeatureConfig:
    """Feature configuration"""
    name: str
    description: str
    state: FeatureState = FeatureState.DISABLED
    stability_level: StabilityLevel = StabilityLevel.TESTING
    rollout_percentage: float = 0.0
    dependency_features: List[str] = None
    health_threshold: float = 0.95
    error_threshold: float = 0.05
    success_threshold: int = 10
    min_uptime_hours: int = 24
    
    def __post_init__(self):
        if self.dependency_features is None:
            self.dependency_features = []

class ProgressiveFeatureManager:
    """
    Progressive Feature Manager for NAVA - Fixed AttributeError Version
    
    Purpose: Safely activate advanced features that are already coded
    - Graduate from Emergency Mode to Intelligent Mode
    - Progressive rollout with health monitoring
    - Automatic rollback on issues
    - Dependencies management
    """
    
    def __init__(self):
        # 🛡️ Initialize ALL attributes FIRST before any method calls
        self.features = {}
        self.feature_metrics = {}
        self.system_health_metrics = {}
        self._monitoring_task = None
        self._initialization_complete = False
        
        # Initialize with error protection
        try:
            self._safe_initialize()
        except Exception as e:
            logger.error(f"❌ Error in ProgressiveFeatureManager init: {e}")
            self._create_emergency_fallback()
    
    def _safe_initialize(self):
        """Safe initialization with error handling"""
        try:
            # Initialize system health metrics
            self.system_health_metrics = {
                "uptime_start": time.time(),
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "avg_response_time": 0.0,
                "error_rate": 0.0,
                "ai_service_health": {
                    "gpt": {"available": True, "success_rate": 1.0},
                    "claude": {"available": True, "success_rate": 1.0},
                    "gemini": {"available": True, "success_rate": 1.0}
                }
            }
            
            # Initialize NAVA features
            self._initialize_nava_features()
            
            # Start monitoring
            self._start_monitoring()
            
            self._initialization_complete = True
            logger.info("✅ ProgressiveFeatureManager initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error in _safe_initialize: {e}")
            self._create_emergency_fallback()
    
    def _create_emergency_fallback(self):
        """Create emergency fallback state"""
        self.features = {}
        self.feature_metrics = {}
        self.system_health_metrics = {
            "uptime_start": time.time(),
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0,
            "error_rate": 0.0,
            "ai_service_health": {
                "gpt": {"available": True, "success_rate": 0.8},
                "claude": {"available": True, "success_rate": 0.8},
                "gemini": {"available": True, "success_rate": 0.8}
            }
        }
        self._initialization_complete = False
        logger.warning("⚠️ ProgressiveFeatureManager using emergency fallback")
    
    def _initialize_nava_features(self):
        """Initialize NAVA feature flags based on coded capabilities"""
        
        # Reset feature_metrics to ensure it exists
        self.feature_metrics = {}
        
        # 🔴 WEEK 1: Basic Features (Already Coded)
        self.features["enhanced_decision_engine"] = FeatureConfig(
            name="enhanced_decision_engine",
            description="9 behavior patterns + advanced model selection",
            state=FeatureState.TESTING,
            stability_level=StabilityLevel.TESTING,
            rollout_percentage=10.0,
            health_threshold=0.90,
            min_uptime_hours=1,
            success_threshold=5
        )
        
        self.features["intelligent_caching"] = FeatureConfig(
            name="intelligent_caching",
            description="Semantic similarity caching for speed",
            state=FeatureState.TESTING,
            stability_level=StabilityLevel.TESTING,
            rollout_percentage=25.0,
            health_threshold=0.95,
            min_uptime_hours=2,
            success_threshold=3
        )
        
        self.features["circuit_breaker"] = FeatureConfig(
            name="circuit_breaker",
            description="AI timeout protection and fallback",
            state=FeatureState.ENABLED,
            stability_level=StabilityLevel.STABLE,
            rollout_percentage=100.0,
            health_threshold=0.80,
            min_uptime_hours=0,
            success_threshold=1
        )
        
        # 🔴 WEEK 2: Learning Features (Already Coded)
        self.features["learning_system"] = FeatureConfig(
            name="learning_system",
            description="User feedback processing and adaptation",
            state=FeatureState.DISABLED,
            stability_level=StabilityLevel.TESTING,
            rollout_percentage=0.0,
            dependency_features=["enhanced_decision_engine"],
            health_threshold=0.95,
            min_uptime_hours=24,
            success_threshold=10
        )
        
        self.features["multi_agent_workflows"] = FeatureConfig(
            name="multi_agent_workflows",
            description="Sequential and parallel AI processing",
            state=FeatureState.DISABLED,
            stability_level=StabilityLevel.TESTING,
            rollout_percentage=0.0,
            dependency_features=["enhanced_decision_engine", "circuit_breaker"],
            health_threshold=0.95,
            min_uptime_hours=48,
            success_threshold=15
        )
        
        # 🔴 WEEK 3: Advanced Features (Already Coded)
        self.features["complex_workflows"] = FeatureConfig(
            name="complex_workflows",
            description="Complex request handling and orchestration",
            state=FeatureState.DISABLED,
            stability_level=StabilityLevel.TESTING,
            rollout_percentage=0.0,
            dependency_features=["learning_system", "multi_agent_workflows"],
            health_threshold=0.98,
            min_uptime_hours=72,
            success_threshold=20
        )
        
        self.features["quality_validation"] = FeatureConfig(
            name="quality_validation",
            description="Multi-dimensional response quality scoring",
            state=FeatureState.TESTING,
            stability_level=StabilityLevel.TESTING,
            rollout_percentage=5.0,
            dependency_features=["enhanced_decision_engine"],
            health_threshold=0.93,
            min_uptime_hours=12,
            success_threshold=8
        )
        
        # Initialize metrics for each feature
        for feature_name in self.features.keys():
            self.feature_metrics[feature_name] = {
                "activation_time": None,
                "total_uses": 0,
                "successful_uses": 0,
                "failed_uses": 0,
                "avg_response_time": 0.0,
                "user_satisfaction": 0.0,
                "rollback_count": 0,
                "last_health_check": time.time()
            }
        
        logger.info(f"✅ Initialized {len(self.features)} NAVA features")
    
    def _start_monitoring(self):
        """Start feature monitoring task"""
        try:
            if self._monitoring_task is None or self._monitoring_task.done():
                self._monitoring_task = asyncio.create_task(self._monitor_features())
                logger.info("✅ Feature monitoring started")
        except Exception as e:
            # Log error but don't fail initialization
            logger.debug(f"💤 Feature monitoring deferred: {e}")
            self._monitoring_task = None
    
    async def _monitor_features(self):
        """Monitor feature health and auto-promote/rollback"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                if not self._initialization_complete:
                    continue
                
                await self._check_feature_health()
                await self._auto_promote_features()
                await self._auto_rollback_unhealthy_features()
                
            except asyncio.CancelledError:
                logger.info("🔄 Feature monitoring stopped")
                break
            except Exception as e:
                logger.error(f"❌ Feature monitoring error: {e}")
                await asyncio.sleep(30)  # Wait before retrying
    
    async def is_feature_enabled(self, feature_name: str, user_id: str = "anonymous", 
                                context: Dict[str, Any] = None) -> bool:
        """
        Check if feature is enabled for this request
        
        Args:
            feature_name: Name of the feature
            user_id: User identifier
            context: Request context
            
        Returns:
            True if feature should be enabled
        """
        try:
            if not self._initialization_complete:
                logger.debug(f"🔒 Feature {feature_name} disabled: initialization not complete")
                return False
            
            if feature_name not in self.features:
                logger.warning(f"⚠️ Unknown feature: {feature_name}")
                return False
            
            feature = self.features[feature_name]
            
            # Check feature state
            if feature.state == FeatureState.DISABLED:
                return False
            elif feature.state == FeatureState.ENABLED:
                return await self._check_feature_health_requirements(feature_name)
            elif feature.state in [FeatureState.TESTING, FeatureState.PARTIAL]:
                # Check rollout percentage
                if not await self._is_in_rollout(feature_name, user_id):
                    return False
                return await self._check_feature_health_requirements(feature_name)
            else:  # DEPRECATED
                return False
                
        except Exception as e:
            logger.error(f"❌ Error checking feature {feature_name}: {e}")
            return False
    
    async def _is_in_rollout(self, feature_name: str, user_id: str) -> bool:
        """Check if user is in feature rollout"""
        try:
            feature = self.features[feature_name]
            
            if feature.rollout_percentage >= 100.0:
                return True
            elif feature.rollout_percentage <= 0.0:
                return False
            
            # Use consistent hash of user_id + feature_name for stable rollout
            hash_input = f"{user_id}_{feature_name}"
            hash_value = hash(hash_input) % 100
            
            return hash_value < feature.rollout_percentage
            
        except Exception as e:
            logger.error(f"❌ Error checking rollout for {feature_name}: {e}")
            return False
    
    async def _check_feature_health_requirements(self, feature_name: str) -> bool:
        """Check if feature meets health requirements"""
        try:
            feature = self.features[feature_name]
            
            # Check system health
            system_health = await self._get_system_health_score()
            if system_health < feature.health_threshold:
                logger.debug(f"🔒 Feature {feature_name} disabled: system health {system_health:.2f} < {feature.health_threshold}")
                return False
            
            # Check dependencies
            for dep_feature in feature.dependency_features:
                if not await self.is_feature_enabled(dep_feature):
                    logger.debug(f"🔒 Feature {feature_name} disabled: dependency {dep_feature} not enabled")
                    return False
            
            # Check uptime requirement
            uptime_hours = (time.time() - self.system_health_metrics["uptime_start"]) / 3600
            if uptime_hours < feature.min_uptime_hours:
                logger.debug(f"🔒 Feature {feature_name} disabled: uptime {uptime_hours:.1f}h < {feature.min_uptime_hours}h")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking health requirements for {feature_name}: {e}")
            return False
    
    async def _get_system_health_score(self) -> float:
        """Calculate overall system health score"""
        try:
            metrics = self.system_health_metrics
            
            if metrics["total_requests"] == 0:
                return 1.0  # Perfect health with no requests
            
            # Calculate success rate
            success_rate = metrics["successful_requests"] / metrics["total_requests"]
            
            # Calculate AI service health
            ai_health_scores = []
            for service, health in metrics["ai_service_health"].items():
                if health["available"]:
                    ai_health_scores.append(health["success_rate"])
                else:
                    ai_health_scores.append(0.0)
            
            avg_ai_health = sum(ai_health_scores) / len(ai_health_scores) if ai_health_scores else 0.0
            
            # Calculate response time health (inverse relationship)
            response_time_health = max(0.0, 1.0 - (metrics["avg_response_time"] / 10.0))
            
            # Weighted health score
            health_score = (
                success_rate * 0.4 +
                avg_ai_health * 0.4 +
                response_time_health * 0.2
            )
            
            return min(1.0, max(0.0, health_score))
            
        except Exception as e:
            logger.error(f"❌ Error calculating system health: {e}")
            return 0.8  # Default moderate health
    
    async def record_feature_usage(self, feature_name: str, success: bool, 
                                 response_time: float = 0.0, user_feedback: float = None):
        """Record feature usage for monitoring"""
        try:
            # Ensure feature_metrics exists
            if not hasattr(self, 'feature_metrics') or self.feature_metrics is None:
                self.feature_metrics = {}
            
            if feature_name not in self.feature_metrics:
                self.feature_metrics[feature_name] = {
                    "activation_time": None,
                    "total_uses": 0,
                    "successful_uses": 0,
                    "failed_uses": 0,
                    "avg_response_time": 0.0,
                    "user_satisfaction": 0.0,
                    "rollback_count": 0,
                    "last_health_check": time.time()
                }
            
            metrics = self.feature_metrics[feature_name]
            metrics["total_uses"] += 1
            metrics["last_health_check"] = time.time()
            
            if success:
                metrics["successful_uses"] += 1
            else:
                metrics["failed_uses"] += 1
            
            # Update average response time
            if response_time > 0:
                if metrics["avg_response_time"] == 0:
                    metrics["avg_response_time"] = response_time
                else:
                    metrics["avg_response_time"] = (metrics["avg_response_time"] * 0.8 + response_time * 0.2)
            
            # Update user satisfaction
            if user_feedback is not None:
                if metrics["user_satisfaction"] == 0:
                    metrics["user_satisfaction"] = user_feedback
                else:
                    metrics["user_satisfaction"] = (metrics["user_satisfaction"] * 0.9 + user_feedback * 0.1)
            
            logger.debug(f"📊 Feature {feature_name} usage recorded: success={success}, time={response_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Error recording feature usage for {feature_name}: {e}")
    
    async def record_system_health(self, total_requests: int, successful_requests: int,
                                 avg_response_time: float, ai_service_health: Dict[str, Any]):
        """Update system health metrics"""
        try:
            self.system_health_metrics.update({
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": total_requests - successful_requests,
                "avg_response_time": avg_response_time,
                "error_rate": (total_requests - successful_requests) / max(1, total_requests),
                "ai_service_health": ai_service_health
            })
        except Exception as e:
            logger.error(f"❌ Error recording system health: {e}")
    
    async def _check_feature_health(self):
        """Check health of enabled features"""
        try:
            for feature_name, feature in self.features.items():
                if feature.state not in [FeatureState.TESTING, FeatureState.ENABLED]:
                    continue
                
                if feature_name not in self.feature_metrics:
                    continue
                
                metrics = self.feature_metrics[feature_name]
                
                if metrics["total_uses"] == 0:
                    continue
                
                # Calculate feature success rate
                success_rate = metrics["successful_uses"] / metrics["total_uses"]
                
                # Check if feature is unhealthy
                if success_rate < (1.0 - feature.error_threshold):
                    logger.warning(f"⚠️ Feature {feature_name} unhealthy: success rate {success_rate:.2f}")
                    await self._consider_rollback(feature_name)
                    
        except Exception as e:
            logger.error(f"❌ Error checking feature health: {e}")
    
    async def _auto_promote_features(self):
        """Auto-promote features based on health and success"""
        try:
            for feature_name, feature in self.features.items():
                if feature.state != FeatureState.TESTING:
                    continue
                
                if feature_name not in self.feature_metrics:
                    continue
                
                metrics = self.feature_metrics[feature_name]
                
                # Check if ready for promotion
                if (metrics["successful_uses"] >= feature.success_threshold and
                    metrics["total_uses"] > 0):
                    
                    success_rate = metrics["successful_uses"] / metrics["total_uses"]
                    
                    if success_rate >= (1.0 - feature.error_threshold):
                        await self._promote_feature(feature_name)
                        
        except Exception as e:
            logger.error(f"❌ Error auto-promoting features: {e}")
    
    async def _promote_feature(self, feature_name: str):
        """Promote feature to next level"""
        try:
            feature = self.features[feature_name]
            
            if feature.state == FeatureState.TESTING:
                if feature.rollout_percentage < 50.0:
                    feature.rollout_percentage = min(50.0, feature.rollout_percentage * 2)
                    logger.info(f"📈 Feature {feature_name} rollout increased to {feature.rollout_percentage}%")
                else:
                    feature.state = FeatureState.PARTIAL
                    feature.rollout_percentage = 75.0
                    logger.info(f"🎉 Feature {feature_name} promoted to PARTIAL (75% rollout)")
            
            elif feature.state == FeatureState.PARTIAL:
                feature.state = FeatureState.ENABLED
                feature.rollout_percentage = 100.0
                feature.stability_level = StabilityLevel.STABLE
                logger.info(f"🎉 Feature {feature_name} promoted to ENABLED (100% rollout)")
                
        except Exception as e:
            logger.error(f"❌ Error promoting feature {feature_name}: {e}")
    
    async def _auto_rollback_unhealthy_features(self):
        """Auto-rollback unhealthy features"""
        try:
            for feature_name, feature in self.features.items():
                if feature.state == FeatureState.DISABLED:
                    continue
                
                if feature_name not in self.feature_metrics:
                    continue
                
                metrics = self.feature_metrics[feature_name]
                
                if metrics["total_uses"] < 5:
                    continue
                
                success_rate = metrics["successful_uses"] / metrics["total_uses"]
                
                if success_rate < 0.7:
                    await self._rollback_feature(feature_name, f"Low success rate: {success_rate:.2f}")
                    
        except Exception as e:
            logger.error(f"❌ Error auto-rolling back features: {e}")
    
    async def _consider_rollback(self, feature_name: str):
        """Consider rolling back a feature"""
        try:
            if feature_name not in self.feature_metrics:
                return
            
            metrics = self.feature_metrics[feature_name]
            
            if metrics["total_uses"] >= 10:
                success_rate = metrics["successful_uses"] / metrics["total_uses"]
                if success_rate < 0.8:
                    await self._rollback_feature(feature_name, f"Poor performance: {success_rate:.2f}")
                    
        except Exception as e:
            logger.error(f"❌ Error considering rollback for {feature_name}: {e}")
    
    async def _rollback_feature(self, feature_name: str, reason: str):
        """Rollback feature due to issues"""
        try:
            feature = self.features[feature_name]
            
            if feature_name not in self.feature_metrics:
                return
            
            metrics = self.feature_metrics[feature_name]
            
            if feature.state == FeatureState.ENABLED:
                feature.state = FeatureState.PARTIAL
                feature.rollout_percentage = 25.0
            elif feature.state == FeatureState.PARTIAL:
                feature.state = FeatureState.TESTING
                feature.rollout_percentage = 5.0
            else:
                feature.state = FeatureState.DISABLED
                feature.rollout_percentage = 0.0
            
            metrics["rollback_count"] += 1
            
            logger.warning(f"🔙 Feature {feature_name} rolled back: {reason}")
            
        except Exception as e:
            logger.error(f"❌ Error rolling back feature {feature_name}: {e}")
    
    def force_enable_feature(self, feature_name: str):
        """Manually enable a feature (for testing)"""
        try:
            if feature_name in self.features:
                feature = self.features[feature_name]
                feature.state = FeatureState.ENABLED
                feature.rollout_percentage = 100.0
                logger.info(f"🔧 Feature {feature_name} manually enabled")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Error force enabling feature {feature_name}: {e}")
            return False
    
    def force_disable_feature(self, feature_name: str):
        """Manually disable a feature"""
        try:
            if feature_name in self.features:
                feature = self.features[feature_name]
                feature.state = FeatureState.DISABLED
                feature.rollout_percentage = 0.0
                logger.info(f"🔧 Feature {feature_name} manually disabled")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Error force disabling feature {feature_name}: {e}")
            return False
    
    def get_feature_status(self) -> Dict[str, Any]:
        """Get status of all features"""
        try:
            status = {}
            
            for feature_name, feature in self.features.items():
                if feature_name not in self.feature_metrics:
                    continue
                
                metrics = self.feature_metrics[feature_name]
                
                status[feature_name] = {
                    "enabled": feature.state == FeatureState.ENABLED,
                    "state": feature.state.value,
                    "stability_level": feature.stability_level.value,
                    "rollout_percentage": feature.rollout_percentage,
                    "total_uses": metrics["total_uses"],
                    "success_rate": (metrics["successful_uses"] / max(1, metrics["total_uses"])),
                    "avg_response_time": metrics["avg_response_time"],
                    "user_satisfaction": metrics["user_satisfaction"],
                    "rollback_count": metrics["rollback_count"],
                    "dependencies": feature.dependency_features,
                    "health_threshold": feature.health_threshold
                }
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Error getting feature status: {e}")
            return {"error": str(e)}
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get system health summary"""
        try:
            return {
                "uptime_hours": (time.time() - self.system_health_metrics["uptime_start"]) / 3600,
                "system_metrics": self.system_health_metrics,
                "enabled_features": [
                    name for name, feature in self.features.items() 
                    if feature.state == FeatureState.ENABLED
                ],
                "testing_features": [
                    name for name, feature in self.features.items() 
                    if feature.state == FeatureState.TESTING
                ],
                "initialization_complete": self._initialization_complete
            }
        except Exception as e:
            logger.error(f"❌ Error getting system health summary: {e}")
            return {"error": str(e)}

# Global feature manager instance
feature_manager = ProgressiveFeatureManager()


# Convenience functions for easy integration
async def is_feature_enabled(feature_name: str, user_id: str = "anonymous", context: Dict[str, Any] = None):
    """Convenience function to check if feature is enabled"""
    try:
        return await feature_manager.is_feature_enabled(feature_name, user_id, context)
    except Exception as e:
        logger.error(f"❌ Error in is_feature_enabled: {e}")
        return False

async def record_feature_usage(feature_name: str, success: bool, response_time: float = 0.0, user_feedback: float = None):
    """Convenience function to record feature usage"""
    try:
        await feature_manager.record_feature_usage(feature_name, success, response_time, user_feedback)
    except Exception as e:
        logger.error(f"❌ Error in record_feature_usage: {e}")

def get_feature_status():
    """Convenience function to get feature status"""
    try:
        return feature_manager.get_feature_status()
    except Exception as e:
        logger.error(f"❌ Error in get_feature_status: {e}")
        return {"error": str(e)}

def force_enable_feature(feature_name: str):
    """Convenience function to manually enable feature"""
    try:
        return feature_manager.force_enable_feature(feature_name)
    except Exception as e:
        logger.error(f"❌ Error in force_enable_feature: {e}")
        return False

# 🔧 เพิ่มฟังก์ชันที่หายไป: update_system_health
def update_system_health(ai_response_time: Optional[float] = None,
                        integration_test_pass_rate: Optional[float] = None,
                        error_rate: Optional[float] = None,
                        availability: Optional[float] = None) -> float:
    """
    Update system health metrics - FIXED VERSION
    
    Args:
        ai_response_time: Average AI response time in seconds
        integration_test_pass_rate: Percentage of integration tests passing
        error_rate: Error rate (0.0 to 1.0)
        availability: System availability percentage
        
    Returns:
        Overall health score (0.0 to 1.0)
    """
    try:
        # Update system health metrics using the global feature manager
        import asyncio
        # Calculate health score first
        health_score = _calculate_sync_health_score(
            ai_response_time, integration_test_pass_rate, error_rate, availability
        )

        # Update feature manager health metrics directly (sync)
        if hasattr(feature_manager, 'system_health_metrics'):
            feature_manager.system_health_metrics.update({
                "total_requests": feature_manager.system_health_metrics.get("total_requests", 0) + 1,
                "successful_requests": feature_manager.system_health_metrics.get("successful_requests", 0) + (0 if error_rate and error_rate > 0.1 else 1),
                "avg_response_time": ai_response_time or feature_manager.system_health_metrics.get("avg_response_time", 0.0),
                "error_rate": error_rate or 0.0
            })
        
        logger.info(f"🏥 System health updated: {health_score:.2f}")
        return health_score
                
    except Exception as e:
        logger.error(f"❌ Error updating system health: {e}")
        # Return reasonable default health score
        return _calculate_sync_health_score(
            ai_response_time, integration_test_pass_rate, error_rate, availability
        )

async def _update_system_health_async(ai_response_time: Optional[float] = None,
                                    integration_test_pass_rate: Optional[float] = None,
                                    error_rate: Optional[float] = None,
                                    availability: Optional[float] = None) -> float:
    """Async version of health update"""
    
    # Update the feature manager's health metrics
    await feature_manager.record_system_health(
        total_requests=feature_manager.system_health_metrics.get("total_requests", 0) + 1,
        successful_requests=feature_manager.system_health_metrics.get("successful_requests", 0) + (0 if error_rate and error_rate > 0.1 else 1),
        avg_response_time=ai_response_time or feature_manager.system_health_metrics.get("avg_response_time", 0.0),
        ai_service_health={
            "gpt": {"available": True, "success_rate": 1.0 - (error_rate or 0.0)},
            "claude": {"available": True, "success_rate": 1.0 - (error_rate or 0.0)},
            "gemini": {"available": True, "success_rate": 1.0 - (error_rate or 0.0)}
        }
    )
    
    # Calculate health score
    return _calculate_sync_health_score(
        ai_response_time, integration_test_pass_rate, error_rate, availability
    )

def _calculate_sync_health_score(ai_response_time: Optional[float] = None,
                               integration_test_pass_rate: Optional[float] = None,
                               error_rate: Optional[float] = None,
                               availability: Optional[float] = None) -> float:
    """Calculate health score synchronously"""
    
    # Default values if not provided
    response_time = ai_response_time or 2.0
    test_pass_rate = integration_test_pass_rate or 95.0
    err_rate = error_rate or 0.05
    avail = availability or 99.0
    
    # Normalize metrics to 0-1 scale
    response_score = max(0.0, 1.0 - (response_time / 10.0))  # 10s = 0 score
    test_score = test_pass_rate / 100.0
    error_score = max(0.0, 1.0 - err_rate)
    availability_score = avail / 100.0
    
    # Weighted health score
    health_score = (
        response_score * 0.3 +
        test_score * 0.3 +
        error_score * 0.2 +
        availability_score * 0.2
    )
    
    return min(1.0, max(0.0, health_score))

# 🔧 เพิ่มฟังก์ชันสำหรับ convenience
def get_feature_manager():
    """Get the global feature manager instance"""
    return feature_manager

def get_current_health_score() -> float:
    """Get current system health score"""
    try:
        return feature_manager.system_health_metrics.get("overall_health", 0.8)
    except Exception as e:
        logger.error(f"❌ Error getting health score: {e}")
        return 0.8