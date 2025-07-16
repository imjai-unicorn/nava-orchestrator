# backend/services/01-core/nava-logic-controller/app/service/learning_engine.py
"""
Learning Engine - Enable Advanced Intelligence
Exit emergency mode and enable adaptive learning
"""

import logging
import time
import json
from typing import Dict, Any, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

class LearningEngine:
    """Simple learning engine for advanced intelligence"""
    
    def __init__(self):
        self.feedback_data = defaultdict(list)
        self.model_performance = {
            "gpt": {"score": 0.8, "count": 0, "avg_response_time": 2.5},
            "claude": {"score": 0.85, "count": 0, "avg_response_time": 3.0},
            "gemini": {"score": 0.82, "count": 0, "avg_response_time": 2.8}
        }
        self.pattern_weights = {
            "conversation": 0.3,
            "teaching": 0.2,
            "brainstorm": 0.15,
            "code_development": 0.15,
            "deep_analysis": 0.1,
            "creative_writing": 0.05,
            "strategic_planning": 0.05
        }
        self.learning_active = True
        self.adaptation_threshold = 5  # Minimum feedback needed for adaptation
        
        logger.info("🧠 Learning Engine initialized - Intelligence features active")
    
    def process_feedback(self, model_used: str, pattern: str, feedback_score: float, 
                        response_time: float = 0.0, context: Dict = None) -> Dict[str, Any]:
        """Process user feedback for learning"""
        
        try:
            if not self.learning_active:
                return {"status": "learning_disabled"}
            
            # Record feedback
            feedback_entry = {
                "model": model_used,
                "pattern": pattern,
                "score": feedback_score,
                "response_time": response_time,
                "timestamp": time.time(),
                "context": context or {}
            }
            
            self.feedback_data[model_used].append(feedback_entry)
            
            # Update model performance
            self._update_model_performance(model_used, feedback_score, response_time)
            
            # Adapt pattern weights if enough data
            if len(self.feedback_data[model_used]) >= self.adaptation_threshold:
                self._adapt_pattern_weights(pattern, feedback_score)
            
            logger.info(f"💡 Feedback processed: {model_used} pattern:{pattern} score:{feedback_score}")
            
            return {
                "status": "feedback_processed",
                "model": model_used,
                "pattern": pattern,
                "learning_applied": True,
                "updated_performance": self.model_performance[model_used]["score"]
            }
            
        except Exception as e:
            logger.error(f"Learning engine feedback error: {e}")
            return {"status": "error", "error": str(e)}
    
    def _update_model_performance(self, model: str, score: float, response_time: float):
        """Update model performance metrics"""
        
        if model in self.model_performance:
            current = self.model_performance[model]
            
            # Update average score (weighted)
            weight = 0.1  # Learning rate
            current["score"] = (1 - weight) * current["score"] + weight * score
            
            # Update count
            current["count"] += 1
            
            # Update average response time
            if response_time > 0:
                current["avg_response_time"] = (
                    (current["avg_response_time"] * (current["count"] - 1) + response_time) / 
                    current["count"]
                )
    
    def _adapt_pattern_weights(self, pattern: str, score: float):
        """Adapt pattern weights based on feedback"""
        
        if pattern in self.pattern_weights:
            # Increase weight for successful patterns
            if score > 0.7:
                adjustment = 0.01  # Small adjustment
                self.pattern_weights[pattern] = min(1.0, self.pattern_weights[pattern] + adjustment)
                
                # Normalize weights
                total_weight = sum(self.pattern_weights.values())
                if total_weight > 1.0:
                    for p in self.pattern_weights:
                        self.pattern_weights[p] /= total_weight
    
    def get_model_recommendation(self, pattern: str, context: Dict = None) -> Dict[str, Any]:
        """Get AI model recommendation based on learning"""
        
        try:
            # Pattern-based initial selection
            pattern_preferences = {
                "conversation": ["gpt", "claude", "gemini"],
                "teaching": ["claude", "gpt", "gemini"],
                "brainstorm": ["gpt", "gemini", "claude"],
                "code_development": ["gpt", "claude", "gemini"],
                "deep_analysis": ["claude", "gemini", "gpt"],
                "creative_writing": ["claude", "gemini", "gpt"],
                "strategic_planning": ["gemini", "claude", "gpt"]
            }
            
            preferred_models = pattern_preferences.get(pattern, ["gpt", "claude", "gemini"])
            
            # Apply learning-based ranking
            ranked_models = []
            for model in preferred_models:
                performance = self.model_performance[model]
                
                # Calculate recommendation score
                score = (
                    performance["score"] * 0.7 +  # Quality weight
                    (1 / max(performance["avg_response_time"], 1)) * 0.3  # Speed weight
                )
                
                ranked_models.append((model, score))
            
            # Sort by score
            ranked_models.sort(key=lambda x: x[1], reverse=True)
            
            recommended_model = ranked_models[0][0]
            confidence = ranked_models[0][1]
            
            return {
                "recommended_model": recommended_model,
                "confidence": min(confidence, 1.0),
                "pattern": pattern,
                "reasoning": f"Best for {pattern} based on performance history",
                "alternatives": [model for model, _ in ranked_models[1:3]],
                "learning_applied": True
            }
            
        except Exception as e:
            logger.error(f"Model recommendation error: {e}")
            return {
                "recommended_model": "gpt",
                "confidence": 0.7,
                "pattern": pattern,
                "reasoning": "Fallback recommendation",
                "learning_applied": False,
                "error": str(e)
            }
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning system statistics"""
        
        total_feedback = sum(len(feedback) for feedback in self.feedback_data.values())
        
        return {
            "learning_active": self.learning_active,
            "total_feedback_count": total_feedback,
            "model_performance": dict(self.model_performance),
            "pattern_weights": dict(self.pattern_weights),
            "adaptation_threshold": self.adaptation_threshold,
            "models_with_feedback": list(self.feedback_data.keys()),
            "learning_effectiveness": self._calculate_learning_effectiveness()
        }
    
    def _calculate_learning_effectiveness(self) -> float:
        """Calculate how effective the learning has been"""
        
        if not self.feedback_data:
            return 0.0
        
        # Simple effectiveness based on average scores
        all_scores = []
        for model_feedback in self.feedback_data.values():
            for feedback in model_feedback:
                all_scores.append(feedback["score"])
        
        if not all_scores:
            return 0.0
        
        return sum(all_scores) / len(all_scores)
    
    def reset_learning(self):
        """Reset learning system to initial state"""
        
        self.feedback_data.clear()
        
        # Reset to initial performance values
        self.model_performance = {
            "gpt": {"score": 0.8, "count": 0, "avg_response_time": 2.5},
            "claude": {"score": 0.85, "count": 0, "avg_response_time": 3.0},
            "gemini": {"score": 0.82, "count": 0, "avg_response_time": 2.8}
        }
        
        # Reset pattern weights
        self.pattern_weights = {
            "conversation": 0.3,
            "teaching": 0.2,
            "brainstorm": 0.15,
            "code_development": 0.15,
            "deep_analysis": 0.1,
            "creative_writing": 0.05,
            "strategic_planning": 0.05
        }
        
        logger.info("🔄 Learning system reset to initial state")
    
    def enable_learning(self):
        """Enable learning system"""
        self.learning_active = True
        logger.info("🧠 Learning system enabled")
    
    def disable_learning(self):
        """Disable learning system"""
        self.learning_active = False
        logger.info("⏸️ Learning system disabled")

# Global instance
learning_engine = LearningEngine()

# Convenience functions
def process_user_feedback(model: str, pattern: str, score: float, **kwargs) -> Dict[str, Any]:
    """Process user feedback"""
    return learning_engine.process_feedback(model, pattern, score, **kwargs)

def get_model_recommendation(pattern: str, context: Dict = None) -> Dict[str, Any]:
    """Get AI model recommendation"""
    return learning_engine.get_model_recommendation(pattern, context)

def get_learning_statistics() -> Dict[str, Any]:
    """Get learning statistics"""
    return learning_engine.get_learning_stats()

def is_learning_active() -> bool:
    """Check if learning is active"""
    return learning_engine.learning_active