# backend/services/05-enhanced-intelligence/decision-engine/app/decision_engine.py
"""
Enhanced Decision Engine Core
Advanced AI Model Selection with Transparency
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging
import time
import asyncio
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# Create router
enhanced_decision_router = APIRouter()

# Models
class DecisionRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    user_id: str = Field(default="anonymous")
    context: Optional[Dict[str, Any]] = Field(None)
    priority: str = Field(default="normal")  # low, normal, high, urgent
    quality_requirements: Optional[Dict[str, str]] = Field(None)
    model_preferences: Optional[List[str]] = Field(None)
    decision_type: str = Field(default="auto")  # auto, manual, hybrid

class DecisionResponse(BaseModel):
    decision_id: str
    recommended_model: str
    confidence: float
    reasoning: Dict[str, Any]
    alternatives: List[Dict[str, Any]]
    criteria_scores: Dict[str, float]
    risk_assessment: Dict[str, Any]
    processing_time_seconds: float
    timestamp: str

class EnhancedDecisionEngine:
    """Enhanced AI Decision Making Engine"""
    
    def __init__(self):
        self.models = {
            "gpt": {
                "capabilities": ["conversation", "creative", "analytical", "code"],
                "speed": "fast",
                "cost": "medium",
                "quality": "high",
                "max_tokens": 4096,
                "specialties": ["general_purpose", "creative_writing", "problem_solving"]
            },
            "claude": {
                "capabilities": ["analytical", "reasoning", "detailed", "research"],
                "speed": "medium", 
                "cost": "medium",
                "quality": "very_high",
                "max_tokens": 100000,
                "specialties": ["deep_analysis", "reasoning", "long_context"]
            },
            "gemini": {
                "capabilities": ["multimodal", "search", "factual", "research"],
                "speed": "fast",
                "cost": "low",
                "quality": "high", 
                "max_tokens": 1000000,
                "specialties": ["fact_checking", "research", "multimodal"]
            }
        }
        
        self.decision_criteria = {
            "capability_match": 0.30,
            "performance_history": 0.25,
            "cost_efficiency": 0.20,
            "quality_requirements": 0.15,
            "availability": 0.10
        }
        
        self.behavior_patterns = {
            "conversation": {"models": ["gpt", "claude"], "priority": "gpt"},
            "creative": {"models": ["gpt", "claude"], "priority": "gpt"},
            "analytical": {"models": ["claude", "gemini"], "priority": "claude"},
            "research": {"models": ["gemini", "claude"], "priority": "gemini"},
            "coding": {"models": ["gpt", "claude"], "priority": "gpt"},
            "factual": {"models": ["gemini", "claude"], "priority": "gemini"}
        }
    
    async def analyze_request(self, request: DecisionRequest) -> Dict[str, Any]:
        """Analyze request to determine best AI model"""
        
        analysis_start = time.time()
        
        try:
            # Step 1: Pattern Recognition
            detected_pattern = self._detect_pattern(request.message)
            
            # Step 2: Capability Matching
            capability_scores = self._score_capabilities(request.message, detected_pattern)
            
            # Step 3: Context Analysis
            context_factors = self._analyze_context(request.context or {})
            
            # Step 4: Quality Requirements
            quality_scores = self._evaluate_quality_requirements(
                request.quality_requirements or {}
            )
            
            # Step 5: Risk Assessment
            risk_factors = self._assess_risks(request.message, request.priority)
            
            # Step 6: Final Decision
            final_decision = self._make_final_decision(
                capability_scores,
                context_factors,
                quality_scores,
                risk_factors,
                request.model_preferences
            )
            
            analysis_time = time.time() - analysis_start
            
            return {
                "pattern_detected": detected_pattern,
                "capability_scores": capability_scores,
                "context_factors": context_factors,
                "quality_scores": quality_scores,
                "risk_factors": risk_factors,
                "final_decision": final_decision,
                "analysis_time": analysis_time
            }
            
        except Exception as e:
            logger.error(f"❌ Decision analysis error: {e}")
            return self._emergency_decision(request.message)
    
    def _detect_pattern(self, message: str) -> str:
        """Detect behavior pattern from message"""
        
        message_lower = message.lower()
        
        # Creative indicators
        if any(word in message_lower for word in ["story", "creative", "write", "poem", "fiction"]):
            return "creative"
        
        # Analytical indicators  
        if any(word in message_lower for word in ["analyze", "compare", "evaluate", "assessment"]):
            return "analytical"
        
        # Research indicators
        if any(word in message_lower for word in ["research", "find", "search", "facts", "data"]):
            return "research"
        
        # Coding indicators
        if any(word in message_lower for word in ["code", "program", "function", "debug", "api"]):
            return "coding"
        
        # Factual indicators
        if any(word in message_lower for word in ["what is", "how to", "explain", "define"]):
            return "factual"
        
        # Default to conversation
        return "conversation"
    
    def _score_capabilities(self, message: str, pattern: str) -> Dict[str, float]:
        """Score each model's capability for this request"""
        
        scores = {}
        pattern_config = self.behavior_patterns.get(pattern, {})
        suitable_models = pattern_config.get("models", ["gpt", "claude", "gemini"])
        priority_model = pattern_config.get("priority", "gpt")
        
        for model_name, model_info in self.models.items():
            base_score = 0.5
            
            # Pattern matching bonus
            if model_name in suitable_models:
                base_score += 0.3
                if model_name == priority_model:
                    base_score += 0.2
            
            # Capability matching
            if pattern in model_info["capabilities"]:
                base_score += 0.2
            
            # Message length consideration
            message_length = len(message)
            if message_length > 1000 and model_info["max_tokens"] > 10000:
                base_score += 0.1
            
            scores[model_name] = min(1.0, base_score)
        
        return scores
    
    def _analyze_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze context factors"""
        
        factors = {
            "urgency": context.get("priority", "normal"),
            "complexity": "medium",
            "user_preference": context.get("preferred_model"),
            "conversation_history": len(context.get("conversation_history", [])),
            "quality_needs": context.get("quality_requirements", {})
        }
        
        return factors
    
    def _evaluate_quality_requirements(self, requirements: Dict[str, str]) -> Dict[str, float]:
        """Evaluate quality requirements for each model"""
        
        scores = {}
        
        for model_name, model_info in self.models.items():
            score = 0.5
            
            if "accuracy" in requirements:
                if model_info["quality"] == "very_high":
                    score += 0.3
                elif model_info["quality"] == "high":
                    score += 0.2
            
            if "speed" in requirements:
                if model_info["speed"] == "fast":
                    score += 0.2
                elif model_info["speed"] == "medium":
                    score += 0.1
            
            if "cost" in requirements:
                if model_info["cost"] == "low":
                    score += 0.2
                elif model_info["cost"] == "medium":
                    score += 0.1
            
            scores[model_name] = min(1.0, score)
        
        return scores
    
    def _assess_risks(self, message: str, priority: str) -> Dict[str, Any]:
        """Assess risks for decision making"""
        
        risk_level = "low"
        risk_factors = []
        
        # High priority = higher risk if we choose wrong model
        if priority in ["high", "urgent"]:
            risk_level = "medium"
            risk_factors.append("high_priority_request")
        
        # Complex message = higher risk
        if len(message) > 2000:
            risk_level = "medium"
            risk_factors.append("complex_request")
        
        # Sensitive content detection (basic)
        sensitive_words = ["password", "secret", "confidential", "private"]
        if any(word in message.lower() for word in sensitive_words):
            risk_level = "high"
            risk_factors.append("sensitive_content")
        
        return {
            "level": risk_level,
            "factors": risk_factors,
            "mitigation_needed": risk_level != "low"
        }
    
    def _make_final_decision(
        self,
        capability_scores: Dict[str, float],
        context_factors: Dict[str, Any],
        quality_scores: Dict[str, float],
        risk_factors: Dict[str, Any],
        model_preferences: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Make final model selection decision"""
        
        final_scores = {}
        
        for model_name in self.models.keys():
            # Weighted scoring
            score = (
                capability_scores.get(model_name, 0.5) * self.decision_criteria["capability_match"] +
                0.8 * self.decision_criteria["performance_history"] +  # Assume good history
                quality_scores.get(model_name, 0.5) * self.decision_criteria["quality_requirements"] +
                (0.9 if self.models[model_name]["cost"] == "low" else 0.7) * self.decision_criteria["cost_efficiency"] +
                0.9 * self.decision_criteria["availability"]  # Assume available
            )
            
            # User preference bonus
            if model_preferences and model_name in model_preferences:
                score += 0.1
            
            # Risk adjustment
            if risk_factors["level"] == "high" and model_name == "claude":
                score += 0.1  # Claude is more conservative
            
            final_scores[model_name] = min(1.0, score)
        
        # Select best model
        best_model = max(final_scores, key=final_scores.get)
        confidence = final_scores[best_model]
        
        # Generate alternatives
        alternatives = []
        sorted_models = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[1:3]
        
        for model, score in sorted_models:
            alternatives.append({
                "model": model,
                "score": score,
                "reason": f"Alternative with {score:.2f} confidence"
            })
        
        return {
            "selected_model": best_model,
            "confidence": confidence,
            "final_scores": final_scores,
            "alternatives": alternatives,
            "reasoning": {
                "primary_factor": "capability_match",
                "decision_criteria": self.decision_criteria,
                "risk_considered": risk_factors["level"]
            }
        }
    
    def _emergency_decision(self, message: str) -> Dict[str, Any]:
        """Emergency fallback decision"""
        
        return {
            "pattern_detected": "emergency",
            "capability_scores": {"gpt": 0.8, "claude": 0.7, "gemini": 0.6},
            "final_decision": {
                "selected_model": "gpt",
                "confidence": 0.7,
                "reasoning": {"method": "emergency_fallback"}
            },
            "analysis_time": 0.001
        }

# Initialize decision engine
decision_engine = EnhancedDecisionEngine()

@enhanced_decision_router.post("/analyze", response_model=DecisionResponse)
async def analyze_decision(request: DecisionRequest):
    """Analyze and make AI model selection decision"""
    
    start_time = time.time()
    
    try:
        # Generate decision ID
        decision_id = f"dec_{int(time.time())}_{hash(request.message) % 1000}"
        
        # Perform analysis
        analysis = await decision_engine.analyze_request(request)
        
        # Extract decision
        final_decision = analysis["final_decision"]
        
        processing_time = time.time() - start_time
        
        response = DecisionResponse(
            decision_id=decision_id,
            recommended_model=final_decision["selected_model"],
            confidence=final_decision["confidence"],
            reasoning=final_decision.get("reasoning", {}),
            alternatives=final_decision.get("alternatives", []),
            criteria_scores=analysis.get("capability_scores", {}),
            risk_assessment=analysis.get("risk_factors", {}),
            processing_time_seconds=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(
            f"🎯 Decision: {final_decision['selected_model']} "
            f"(confidence: {final_decision['confidence']:.2f}, "
            f"time: {processing_time:.3f}s)"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Decision analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Decision analysis failed: {str(e)}")

@enhanced_decision_router.get("/models")
async def get_available_models():
    """Get available AI models and their capabilities"""
    
    return {
        "models": decision_engine.models,
        "behavior_patterns": decision_engine.behavior_patterns,
        "decision_criteria": decision_engine.decision_criteria,
        "timestamp": datetime.now().isoformat()
    }

@enhanced_decision_router.get("/patterns")
async def get_behavior_patterns():
    """Get supported behavior patterns"""
    
    return {
        "patterns": decision_engine.behavior_patterns,
        "total_patterns": len(decision_engine.behavior_patterns),
        "timestamp": datetime.now().isoformat()
    }

@enhanced_decision_router.post("/quick-select")
async def quick_model_select(message: str, priority: str = "normal"):
    """Quick model selection without full analysis"""
    
    try:
        pattern = decision_engine._detect_pattern(message)
        pattern_config = decision_engine.behavior_patterns.get(pattern, {})
        
        recommended_model = pattern_config.get("priority", "gpt")
        alternatives = pattern_config.get("models", ["gpt"])
        
        return {
            "recommended_model": recommended_model,
            "pattern_detected": pattern,
            "alternatives": alternatives,
            "confidence": 0.8,
            "method": "quick_select",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Quick select error: {e}")
        return {
            "recommended_model": "gpt",
            "pattern_detected": "fallback",
            "confidence": 0.6,
            "error": str(e)
        }