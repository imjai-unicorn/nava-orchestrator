# backend/services/01-core/nava-logic-controller/app/core/model_switcher.py
"""
Model Switcher - AI Model Selection and Switching Logic
Handles intelligent routing between GPT, Claude, and Gemini based on criteria
"""

import logging
import time
from typing import Dict, Any, Optional, List
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AIModel(Enum):
    """Available AI models"""
    GPT = "gpt"
    CLAUDE = "claude"
    GEMINI = "gemini"
    LOCAL = "local"

class ModelSwitcher:
    """
    Intelligent AI model selection and switching
    Optimizes model selection based on performance, availability, and requirements
    """
    
    def __init__(self):
        self.models = {
            AIModel.GPT: {
                "name": "GPT",
                "endpoint": "http://localhost:8002",
                "strengths": ["conversation", "creative", "general"],
                "performance_score": 0.85,
                "availability": True,
                "cost_per_request": 0.10,
                "avg_response_time": 2.5,
                "timeout_threshold": 15.0,
                "failure_count": 0,
                "last_failure": None
            },
            AIModel.CLAUDE: {
                "name": "Claude",
                "endpoint": "http://localhost:8003", 
                "strengths": ["analysis", "reasoning", "safety"],
                "performance_score": 0.90,
                "availability": True,
                "cost_per_request": 0.12,
                "avg_response_time": 2.8,
                "timeout_threshold": 20.0,
                "failure_count": 0,
                "last_failure": None
            },
            AIModel.GEMINI: {
                "name": "Gemini",
                "endpoint": "http://localhost:8004",
                "strengths": ["research", "factual", "multimodal"],
                "performance_score": 0.80,
                "availability": True,
                "cost_per_request": 0.08,
                "avg_response_time": 3.2,
                "timeout_threshold": 18.0,
                "failure_count": 0,
                "last_failure": None
            },
            AIModel.LOCAL: {
                "name": "Local AI",
                "endpoint": "http://localhost:8018",
                "strengths": ["privacy", "speed", "cost"],
                "performance_score": 0.70,
                "availability": False,  # ✅ แก้เป็น False (Not yet available in Phase 1)
                "cost_per_request": 0.01,
                "avg_response_time": 0.5,
                "timeout_threshold": 5.0,
                "failure_count": 0,
                "last_failure": None            
            }
        }
        
        self.selection_criteria = {
            "performance": 0.30,  # 30% weight on performance
            "availability": 0.25,  # 25% weight on availability
            "speed": 0.20,        # 20% weight on response time
            "cost": 0.15,         # 15% weight on cost
            "task_match": 0.10    # 10% weight on task alignment
        }
        
        self.fallback_chain = [AIModel.GPT, AIModel.CLAUDE, AIModel.GEMINI, AIModel.LOCAL]
        self.circuit_breaker_threshold = 5  # failures before circuit break
        self.circuit_breaker_timeout = 300  # 5 minutes before retry
        
        logger.info("✅ Model Switcher initialized")
    
    def select_best_model(self, 
                         task_type: str = "general",
                         user_preference: Optional[str] = None,
                         priority: str = "balanced",
                         context: Optional[Dict[str, Any]] = None) -> AIModel:
        """
        Select the best AI model for the given task and criteria
        
        Args:
            task_type: Type of task (conversation, analysis, creative, etc.)
            user_preference: User's preferred model (if any)
            priority: Priority mode (speed, quality, cost, balanced)
            context: Additional context for selection
        
        Returns:
            Selected AI model enum
        """
        try:
            # Check user preference first
            if user_preference:
                preferred_model = self._get_model_by_name(user_preference)
                if preferred_model and self._is_model_available(preferred_model):
                    logger.info(f"✅ Using user preference: {preferred_model.value}")
                    return preferred_model
            
            # Calculate scores for each available model
            model_scores = {}
            for model, config in self.models.items():
                if not config["availability"] or not self._is_model_available(model):
                    continue
                
                score = self._calculate_model_score(model, task_type, priority, context)
                model_scores[model] = score
            
            if not model_scores:
                logger.warning("⚠️ No models available, using fallback")
                return self._get_fallback_model()
            
            # Select model with highest score
            best_model = max(model_scores, key=model_scores.get)
            best_score = model_scores[best_model]
            
            logger.info(f"✅ Selected {best_model.value} (score: {best_score:.3f}) for {task_type}")
            return best_model
            
        except Exception as e:
            logger.error(f"❌ Model selection error: {e}")
            return self._get_fallback_model()
    
    def _calculate_model_score(self, 
                              model: AIModel, 
                              task_type: str, 
                              priority: str,
                              context: Optional[Dict[str, Any]]) -> float:
        """Calculate weighted score for a model"""
        config = self.models[model]
        score = 0.0
        
        # Performance score
        performance_score = config["performance_score"]
        score += performance_score * self.selection_criteria["performance"]
        
        # Availability score (binary: 1.0 if available and healthy)
        availability_score = 1.0 if self._is_model_healthy(model) else 0.0
        score += availability_score * self.selection_criteria["availability"]
        
        # Speed score (inverse of response time, normalized)
        max_response_time = 10.0  # seconds
        speed_score = max(0.0, (max_response_time - config["avg_response_time"]) / max_response_time)
        score += speed_score * self.selection_criteria["speed"]
        
        # Cost score (inverse of cost, normalized)
        max_cost = 0.20
        cost_score = max(0.0, (max_cost - config["cost_per_request"]) / max_cost)
        score += cost_score * self.selection_criteria["cost"]
        
        # Task alignment score
        task_match_score = self._calculate_task_match(model, task_type)
        score += task_match_score * self.selection_criteria["task_match"]
        
        # Priority adjustments
        score = self._apply_priority_adjustments(score, model, priority)
        
        return score
    
    def _calculate_task_match(self, model: AIModel, task_type: str) -> float:
        """Calculate how well a model matches the task type"""
        config = self.models[model]
        strengths = config["strengths"]
        
        # Task type mapping
        task_keywords = {
            "conversation": ["conversation", "general"],
            "creative": ["creative", "conversation"],
            "analysis": ["analysis", "reasoning"],
            "research": ["research", "factual"],
            "code": ["analysis", "reasoning"],
            "safety": ["safety", "reasoning"],
            "general": ["general", "conversation"]
        }
        
        task_matches = task_keywords.get(task_type.lower(), ["general"])
        
        # Calculate match score
        match_count = sum(1 for strength in strengths if strength in task_matches)
        max_possible_matches = len(task_matches)
        
        return match_count / max(max_possible_matches, 1)
    
    def _apply_priority_adjustments(self, base_score: float, model: AIModel, priority: str) -> float:
        """Apply priority-based adjustments to the score"""
        config = self.models[model]
        
        if priority == "speed":
            # Boost models with faster response times
            if config["avg_response_time"] < 2.0:
                base_score += 0.2
        elif priority == "quality":
            # Boost models with higher performance scores
            if config["performance_score"] > 0.85:
                base_score += 0.2
        elif priority == "cost":
            # Boost cheaper models
            if config["cost_per_request"] < 0.10:
                base_score += 0.2
        # "balanced" priority uses base score without adjustment
        
        return base_score
    
    def _is_model_available(self, model: AIModel) -> bool:
        """Check if model is available and not circuit broken"""
        config = self.models[model]
        
        # Check basic availability
        if not config["availability"]:
            return False
        
        # Check circuit breaker
        if config["failure_count"] >= self.circuit_breaker_threshold:
            if config["last_failure"]:
                time_since_failure = (datetime.now() - config["last_failure"]).total_seconds()
                if time_since_failure < self.circuit_breaker_timeout:
                    return False
                else:
                    # Reset circuit breaker
                    config["failure_count"] = 0
                    config["last_failure"] = None
                    logger.info(f"✅ Circuit breaker reset for {model.value}")
        
        return True
    
    def _is_model_healthy(self, model: AIModel) -> bool:
        """Check if model is healthy (available and low failure rate)"""
        config = self.models[model]
        return (config["availability"] and 
                config["failure_count"] < self.circuit_breaker_threshold)
    
    def _get_model_by_name(self, name: str) -> Optional[AIModel]:
        """Get model enum by name string"""
        name_lower = name.lower()
        for model in AIModel:
            if model.value.lower() == name_lower:
                return model
        return None
    
    def _get_fallback_model(self) -> AIModel:
        """Get the first available fallback model"""
        for model in self.fallback_chain:
            if self._is_model_available(model):
                logger.info(f"✅ Using fallback model: {model.value}")
                return model
        
        # If no models available, return GPT as last resort
        logger.warning("⚠️ No fallback models available, using GPT")
        return AIModel.GPT
    
    def report_model_success(self, model: AIModel, response_time: float):
        """Report successful model usage"""
        config = self.models[model]
        
        # Update average response time (simple moving average)
        config["avg_response_time"] = (config["avg_response_time"] * 0.8 + response_time * 0.2)
        
        # Reset failure count on success
        if config["failure_count"] > 0:
            config["failure_count"] = max(0, config["failure_count"] - 1)
        
        logger.debug(f"✅ Model {model.value} success: {response_time:.2f}s")
    
    def report_model_failure(self, model: AIModel, error_type: str = "unknown"):
        """Report model failure"""
        config = self.models[model]
        config["failure_count"] += 1
        config["last_failure"] = datetime.now()
        
        if config["failure_count"] >= self.circuit_breaker_threshold:
            logger.warning(f"⚠️ Circuit breaker triggered for {model.value}")
        
        logger.error(f"❌ Model {model.value} failure: {error_type}")
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models"""
        status = {}
        for model, config in self.models.items():
            status[model.value] = {
                "available": config["availability"],
                "healthy": self._is_model_healthy(model),
                "performance_score": config["performance_score"],
                "avg_response_time": config["avg_response_time"],
                "failure_count": config["failure_count"],
                "circuit_breaker_active": (config["failure_count"] >= self.circuit_breaker_threshold),
                "endpoint": config["endpoint"]
            }
        return status
    
    def set_model_availability(self, model: AIModel, available: bool):
        """Manually set model availability"""
        self.models[model]["availability"] = available
        logger.info(f"✅ Model {model.value} availability set to {available}")
    
    def get_recommended_fallback_chain(self, task_type: str = "general") -> List[AIModel]:
        """Get recommended fallback chain for a specific task"""
        # Score all models for the task
        model_scores = []
        for model in self.models:
            if self.models[model]["availability"]:
                score = self._calculate_task_match(model, task_type)
                model_scores.append((model, score))
        
        # Sort by score descending
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [model for model, score in model_scores]
    
    def update_model_config(self, model: AIModel, config_updates: Dict[str, Any]):
        """Update model configuration"""
        if model in self.models:
            self.models[model].update(config_updates)
            logger.info(f"✅ Updated config for {model.value}")
        else:
            logger.warning(f"⚠️ Unknown model: {model}")

# Global instance
model_switcher = ModelSwitcher()

def select_ai_model(task_type: str = "general", 
                   user_preference: Optional[str] = None,
                   priority: str = "balanced",
                   context: Optional[Dict[str, Any]] = None) -> AIModel:
    """
    Convenient function for AI model selection
    Used throughout the core logic controller
    """
    return model_switcher.select_best_model(task_type, user_preference, priority, context)

def get_model_endpoint(model: AIModel) -> str:
    """Get endpoint URL for a specific model"""
    return model_switcher.models[model]["endpoint"]

def report_success(model: AIModel, response_time: float):
    """Report successful model usage"""
    model_switcher.report_model_success(model, response_time)

def report_failure(model: AIModel, error_type: str = "unknown"):
    """Report model failure"""
    model_switcher.report_model_failure(model, error_type)

def get_all_model_status() -> Dict[str, Any]:
    """Get status of all AI models"""
    return model_switcher.get_model_status()