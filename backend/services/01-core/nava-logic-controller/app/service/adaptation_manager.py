"""
Adaptation Manager - Week 4 System Adaptation
Intelligent system adaptation based on performance data and user feedback
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics
import json

class AdaptationType(Enum):
    """Types of system adaptations"""
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    MODEL_SELECTION = "model_selection"
    FEATURE_ADJUSTMENT = "feature_adjustment"
    RESOURCE_ALLOCATION = "resource_allocation"
    ERROR_HANDLING = "error_handling"
    USER_EXPERIENCE = "user_experience"

@dataclass
class AdaptationRule:
    """Rule for system adaptation"""
    rule_id: str
    name: str
    adaptation_type: AdaptationType
    condition: Dict[str, Any]
    action: Dict[str, Any]
    priority: int = 5  # 1-10, higher = more important
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    success_rate: float = 1.0

@dataclass
class AdaptationEvent:
    """Record of an adaptation that was applied"""
    event_id: str
    timestamp: datetime
    rule_id: str
    adaptation_type: AdaptationType
    trigger_reason: str
    action_taken: Dict[str, Any]
    success: bool
    impact_measurement: Optional[Dict[str, Any]] = None
    rollback_info: Optional[Dict[str, Any]] = None

class AdaptationManager:
    """
    Intelligent system adaptation manager for NAVA
    Monitors system performance and automatically adjusts configurations
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Adaptation rules and history
        self.adaptation_rules = {}
        self.adaptation_history = []
        self.active_adaptations = {}
        
        # Performance tracking
        self.performance_baseline = {}
        self.adaptation_impacts = {}
        
        # Configuration
        self.adaptation_config = {
            'max_adaptations_per_hour': 10,
            'min_time_between_adaptations': 300,  # 5 minutes
            'rollback_threshold': 0.8,  # Success rate below which to rollback
            'learning_window_hours': 24,
            'confidence_threshold': 0.7
        }
        
        # Monitoring
        self._monitoring_task = None
        self._is_monitoring = False
        
        # Initialize default rules
        self._initialize_default_rules()

    def _initialize_default_rules(self):
        """Initialize default adaptation rules"""
        
        # Performance optimization rules
        self.add_adaptation_rule(AdaptationRule(
            rule_id="high_response_time",
            name="High Response Time Optimization",
            adaptation_type=AdaptationType.PERFORMANCE_OPTIMIZATION,
            condition={
                'metric': 'avg_response_time',
                'threshold': 3000,  # 3 seconds
                'duration_minutes': 5,
                'comparison': 'greater_than'
            },
            action={
                'type': 'enable_caching',
                'parameters': {'cache_ttl': 300, 'cache_size_mb': 100}
            },
            priority=8
        ))
        
        # Model selection optimization
        self.add_adaptation_rule(AdaptationRule(
            rule_id="model_failure_rate",
            name="Model Failure Rate Adaptation",
            adaptation_type=AdaptationType.MODEL_SELECTION,
            condition={
                'metric': 'model_success_rate',
                'threshold': 0.9,
                'service': 'gpt-client',
                'duration_minutes': 10,
                'comparison': 'less_than'
            },
            action={
                'type': 'adjust_model_priority',
                'parameters': {'decrease_priority': 'gpt', 'increase_priority': 'claude'}
            },
            priority=9
        ))
        
        # Resource allocation
        self.add_adaptation_rule(AdaptationRule(
            rule_id="high_load_optimization",
            name="High Load Resource Optimization",
            adaptation_type=AdaptationType.RESOURCE_ALLOCATION,
            condition={
                'metric': 'requests_per_minute',
                'threshold': 100,
                'duration_minutes': 5,
                'comparison': 'greater_than'
            },
            action={
                'type': 'scale_resources',
                'parameters': {'action': 'increase_timeout', 'value': 1.5}
            },
            priority=7
        ))
        
        # Error handling adaptation
        self.add_adaptation_rule(AdaptationRule(
            rule_id="high_error_rate",
            name="High Error Rate Response",
            adaptation_type=AdaptationType.ERROR_HANDLING,
            condition={
                'metric': 'error_rate',
                'threshold': 0.1,  # 10%
                'duration_minutes': 5,
                'comparison': 'greater_than'
            },
            action={
                'type': 'enable_fallback_mode',
                'parameters': {'mode': 'conservative', 'duration_minutes': 30}
            },
            priority=10
        ))

    def add_adaptation_rule(self, rule: AdaptationRule):
        """Add or update an adaptation rule"""
        self.adaptation_rules[rule.rule_id] = rule
        self.logger.info(f"Added adaptation rule: {rule.name}")

    def remove_adaptation_rule(self, rule_id: str):
        """Remove an adaptation rule"""
        if rule_id in self.adaptation_rules:
            del self.adaptation_rules[rule_id]
            self.logger.info(f"Removed adaptation rule: {rule_id}")

    def enable_rule(self, rule_id: str, enabled: bool = True):
        """Enable or disable an adaptation rule"""
        if rule_id in self.adaptation_rules:
            self.adaptation_rules[rule_id].enabled = enabled
            status = "enabled" if enabled else "disabled"
            self.logger.info(f"Rule {rule_id} {status}")

    async def start_monitoring(self):
        """Start adaptive monitoring"""
        if self._is_monitoring:
            return
            
        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._adaptation_loop())
        self.logger.info("Adaptation monitoring started")

    async def stop_monitoring(self):
        """Stop adaptive monitoring"""
        self._is_monitoring = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Adaptation monitoring stopped")

    async def _adaptation_loop(self):
        """Main adaptation monitoring loop"""
        while self._is_monitoring:
            try:
                # Check for adaptation opportunities
                await self._evaluate_adaptations()
                
                # Clean up old adaptations
                self._cleanup_old_adaptations()
                
                # Measure adaptation impacts
                await self._measure_adaptation_impacts()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in adaptation loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

    async def _evaluate_adaptations(self):
        """Evaluate whether adaptations should be triggered"""
        # Import here to avoid circular imports
        try:
            from .performance_tracker import performance_tracker
            performance_summary = performance_tracker.get_performance_summary(30)
        except ImportError:
            # Fallback performance data
            performance_summary = {
                'avg_response_time': 2000,
                'success_rate': 0.95,
                'requests_per_minute': 50,
                'system_health_score': 0.85
            }
        
        for rule in self.adaptation_rules.values():
            if not rule.enabled:
                continue
                
            # Check if rule conditions are met
            if await self._check_rule_condition(rule, performance_summary):
                await self._trigger_adaptation(rule, performance_summary)

    async def _check_rule_condition(self, rule: AdaptationRule, performance_data: Dict[str, Any]) -> bool:
        """Check if a rule's conditions are met"""
        condition = rule.condition
        
        # Rate limiting - don't trigger too frequently
        if rule.last_triggered:
            time_since_last = (datetime.now() - rule.last_triggered).total_seconds()
            if time_since_last < self.adaptation_config['min_time_between_adaptations']:
                return False
        
        # Check hourly rate limit
        recent_triggers = sum(1 for event in self.adaptation_history 
                            if event.timestamp > datetime.now() - timedelta(hours=1))
        if recent_triggers >= self.adaptation_config['max_adaptations_per_hour']:
            return False
        
        # Evaluate the specific condition
        metric_value = self._extract_metric_value(condition, performance_data)
        if metric_value is None:
            return False
        
        threshold = condition['threshold']
        comparison = condition.get('comparison', 'greater_than')
        
        if comparison == 'greater_than':
            return metric_value > threshold
        elif comparison == 'less_than':
            return metric_value < threshold
        elif comparison == 'equals':
            return metric_value == threshold
        
        return False

    def _extract_metric_value(self, condition: Dict[str, Any], performance_data: Dict[str, Any]) -> Optional[float]:
        """Extract metric value from performance data based on condition"""
        metric = condition['metric']
        service = condition.get('service')
        
        if metric == 'avg_response_time':
            if service and 'service_breakdown' in performance_data:
                return performance_data['service_breakdown'].get(service, {}).get('avg_response_time')
            return performance_data.get('avg_response_time')
        
        elif metric == 'error_rate':
            success_rate = performance_data.get('success_rate', 1.0)
            return 1.0 - success_rate
        
        elif metric == 'model_success_rate':
            if service and 'service_breakdown' in performance_data:
                return performance_data['service_breakdown'].get(service, {}).get('success_rate')
            return performance_data.get('success_rate')
        
        elif metric == 'requests_per_minute':
            return performance_data.get('requests_per_minute')
        
        elif metric == 'system_health_score':
            return performance_data.get('system_health_score')
        
        return None

    async def _trigger_adaptation(self, rule: AdaptationRule, performance_data: Dict[str, Any]):
        """Trigger an adaptation based on a rule"""
        try:
            # Create adaptation event
            event = AdaptationEvent(
                event_id=f"{rule.rule_id}_{int(datetime.now().timestamp())}",
                timestamp=datetime.now(),
                rule_id=rule.rule_id,
                adaptation_type=rule.adaptation_type,
                trigger_reason=f"Rule {rule.name} triggered",
                action_taken=rule.action,
                success=False  # Will be updated after execution
            )
            
            # Execute the adaptation
            success = await self._execute_adaptation(rule.action, event)
            event.success = success
            
            # Update rule statistics
            rule.last_triggered = datetime.now()
            rule.trigger_count += 1
            
            # Store the event
            self.adaptation_history.append(event)
            if success:
                self.active_adaptations[event.event_id] = event
            
            # Update rule success rate
            recent_events = [e for e in self.adaptation_history 
                           if e.rule_id == rule.rule_id and 
                           e.timestamp > datetime.now() - timedelta(hours=24)]
            if recent_events:
                rule.success_rate = sum(1 for e in recent_events if e.success) / len(recent_events)
            
            self.logger.info(f"Adaptation triggered: {rule.name}, Success: {success}")
            
        except Exception as e:
            self.logger.error(f"Error triggering adaptation {rule.rule_id}: {e}")

    async def _execute_adaptation(self, action: Dict[str, Any], event: AdaptationEvent) -> bool:
        """Execute an adaptation action"""
        action_type = action.get('type')
        parameters = action.get('parameters', {})
        
        try:
            if action_type == 'enable_caching':
                return await self._enable_caching(parameters, event)
            
            elif action_type == 'adjust_model_priority':
                return await self._adjust_model_priority(parameters, event)
            
            elif action_type == 'scale_resources':
                return await self._scale_resources(parameters, event)
            
            elif action_type == 'enable_fallback_mode':
                return await self._enable_fallback_mode(parameters, event)
            
            elif action_type == 'adjust_timeouts':
                return await self._adjust_timeouts(parameters, event)
            
            elif action_type == 'modify_feature_flags':
                return await self._modify_feature_flags(parameters, event)
            
            else:
                self.logger.warning(f"Unknown adaptation action type: {action_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error executing adaptation {action_type}: {e}")
            return False

    async def _enable_caching(self, parameters: Dict[str, Any], event: AdaptationEvent) -> bool:
        """Enable or optimize caching"""
        try:
            # This would integrate with the actual cache manager
            cache_ttl = parameters.get('cache_ttl', 300)
            cache_size_mb = parameters.get('cache_size_mb', 100)
            
            # Log the adaptation for now
            self.logger.info(f"Would enable caching: TTL={cache_ttl}s, Size={cache_size_mb}MB")
            
            # Store rollback information
            event.rollback_info = {
                'action': 'restore_cache_settings',
                'previous_settings': {'ttl': 600, 'size_mb': 50}
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to enable caching: {e}")
            return False

    async def _adjust_model_priority(self, parameters: Dict[str, Any], event: AdaptationEvent) -> bool:
        """Adjust AI model selection priorities"""
        try:
            decrease_model = parameters.get('decrease_priority')
            increase_model = parameters.get('increase_priority')
            adjustment = parameters.get('adjustment', 0.1)
            
            # Log the adaptation for now
            self.logger.info(f"Would adjust model priorities: {decrease_model}↓ {increase_model}↑")
            
            # Store rollback information
            event.rollback_info = {
                'action': 'restore_model_priorities',
                'previous_priorities': {'gpt': 0.8, 'claude': 0.9, 'gemini': 0.7}
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to adjust model priorities: {e}")
            return False

    async def _scale_resources(self, parameters: Dict[str, Any], event: AdaptationEvent) -> bool:
        """Scale system resources"""
        try:
            action = parameters.get('action')
            value = parameters.get('value', 1.0)
            
            if action == 'increase_timeout':
                # Log the adaptation for now
                self.logger.info(f"Would increase timeouts by factor {value}")
                
                # Store rollback information
                event.rollback_info = {
                    'action': 'restore_timeouts',
                    'previous_timeouts': {'gpt': 15, 'claude': 20, 'gemini': 18}
                }
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to scale resources: {e}")
            return False

    async def _enable_fallback_mode(self, parameters: Dict[str, Any], event: AdaptationEvent) -> bool:
        """Enable fallback mode for error handling"""
        try:
            mode = parameters.get('mode', 'conservative')
            duration_minutes = parameters.get('duration_minutes', 30)
            
            # Log the adaptation for now
            self.logger.info(f"Would enable fallback mode: {mode} for {duration_minutes} minutes")
            
            # Store rollback information
            event.rollback_info = {
                'action': 'disable_fallback_mode',
                'enabled_until': datetime.now() + timedelta(minutes=duration_minutes)
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to enable fallback mode: {e}")
            return False

    async def _adjust_timeouts(self, parameters: Dict[str, Any], event: AdaptationEvent) -> bool:
        """Adjust service timeouts"""
        try:
            service = parameters.get('service', 'all')
            timeout_ms = parameters.get('timeout_ms')
            multiplier = parameters.get('multiplier', 1.0)
            
            # Log the adaptation for now
            self.logger.info(f"Would adjust timeout for {service}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to adjust timeouts: {e}")
            return False

    async def _modify_feature_flags(self, parameters: Dict[str, Any], event: AdaptationEvent) -> bool:
        """Modify feature flags"""
        try:
            feature = parameters.get('feature')
            enabled = parameters.get('enabled', True)
            
            # Log the adaptation for now
            self.logger.info(f"Would modify feature flag {feature}: {enabled}")
            
            # Store rollback information
            event.rollback_info = {
                'action': 'restore_feature_flag',
                'feature': feature,
                'previous_state': not enabled
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to modify feature flags: {e}")
            return False

    async def _measure_adaptation_impacts(self):
        """Measure the impact of recent adaptations"""
        # Simplified impact measurement for now
        for event_id, event in list(self.active_adaptations.items()):
            # Only measure adaptations that have been active for at least 10 minutes
            if (datetime.now() - event.timestamp).total_seconds() < 600:
                continue
            
            # Simple impact calculation
            impact = {
                'adaptation_id': event.event_id,
                'measurement_time': datetime.now().isoformat(),
                'success_score': 0.8,  # Simulated success score
                'recommendation': 'continue'
            }
            
            event.impact_measurement = impact
            
            # Archive long-running adaptations
            if (datetime.now() - event.timestamp).total_seconds() > 3600:  # 1 hour
                self.adaptation_impacts[event_id] = impact
                self.active_adaptations.pop(event_id, None)

    def _cleanup_old_adaptations(self):
        """Clean up old adaptation records"""
        cutoff_time = datetime.now() - timedelta(days=7)
        
        # Remove old history entries
        self.adaptation_history = [
            event for event in self.adaptation_history 
            if event.timestamp > cutoff_time
        ]
        
        # Remove old impact measurements
        old_impacts = [
            event_id for event_id, impact in self.adaptation_impacts.items()
            if datetime.fromisoformat(impact['measurement_time']) < cutoff_time
        ]
        for event_id in old_impacts:
            self.adaptation_impacts.pop(event_id, None)

    def get_adaptation_status(self) -> Dict[str, Any]:
        """Get current adaptation system status"""
        active_count = len(self.active_adaptations)
        total_rules = len(self.adaptation_rules)
        enabled_rules = sum(1 for rule in self.adaptation_rules.values() if rule.enabled)
        
        recent_events = [
            event for event in self.adaptation_history 
            if event.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        success_rate = (sum(1 for event in recent_events if event.success) / len(recent_events)) if recent_events else 1.0
        
        return {
            'monitoring_active': self._is_monitoring,
            'total_rules': total_rules,
            'enabled_rules': enabled_rules,
            'active_adaptations': active_count,
            'adaptations_last_24h': len(recent_events),
            'success_rate_24h': success_rate,
            'adaptation_impacts': len(self.adaptation_impacts),
            'system_adaptive_health': self._calculate_adaptive_health()
        }

    def _calculate_adaptive_health(self) -> float:
        """Calculate overall adaptive system health"""
        # Based on rule success rates and recent performance
        if not self.adaptation_rules:
            return 1.0
        
        rule_health_scores = []
        for rule in self.adaptation_rules.values():
            if rule.enabled:
                rule_health_scores.append(rule.success_rate)
        
        return statistics.mean(rule_health_scores) if rule_health_scores else 1.0

    def get_adaptation_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for system improvements"""
        recommendations = []
        
        # Simplified recommendations for now
        recommendations.append({
            'type': 'monitoring',
            'priority': 'medium',
            'suggestion': 'System adaptation monitoring is active',
            'estimated_impact': 'Continuous optimization'
        })
        
        return recommendations

    def export_adaptation_data(self) -> Dict[str, Any]:
        """Export adaptation data for analysis"""
        return {
            'rules': {
                rule_id: {
                    'name': rule.name,
                    'type': rule.adaptation_type.value,
                    'enabled': rule.enabled,
                    'priority': rule.priority,
                    'trigger_count': rule.trigger_count,
                    'success_rate': rule.success_rate,
                    'last_triggered': rule.last_triggered.isoformat() if rule.last_triggered else None
                }
                for rule_id, rule in self.adaptation_rules.items()
            },
            'recent_events': [
                {
                    'event_id': event.event_id,
                    'timestamp': event.timestamp.isoformat(),
                    'rule_id': event.rule_id,
                    'type': event.adaptation_type.value,
                    'success': event.success,
                    'trigger_reason': event.trigger_reason
                }
                for event in self.adaptation_history[-50:]  # Last 50 events
            ],
            'active_adaptations': len(self.active_adaptations),
            'system_status': self.get_adaptation_status(),
            'recommendations': self.get_adaptation_recommendations()
        }

# Global adaptation manager instance
adaptation_manager = AdaptationManager()
