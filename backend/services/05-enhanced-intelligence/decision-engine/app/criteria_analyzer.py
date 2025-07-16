# backend/services/05-enhanced-intelligence/decision-engine/app/criteria_analyzer.py
"""
Criteria Analyzer
Advanced criteria evaluation for decision making
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

# Create router
criteria_router = APIRouter()

# Models
class CriteriaRequest(BaseModel):
    criteria: Dict[str, Any] = Field(..., description="Criteria to analyze")
    weights: Optional[Dict[str, float]] = Field(None, description="Criteria weights")
    options: List[str] = Field(..., description="Options to evaluate")
    context: Optional[Dict[str, Any]] = Field(None)

class CriteriaResponse(BaseModel):
    criteria_id: str
    analysis_result: Dict[str, Any]
    scores: Dict[str, float]
    rankings: List[Dict[str, Any]]
    recommendations: List[str]
    processing_time_seconds: float
    timestamp: str

class CriteriaAnalyzer:
    """Advanced Criteria Analysis Engine"""
    
    def __init__(self):
        self.default_weights = {
            "performance": 0.25,
            "cost": 0.20,
            "quality": 0.25,
            "reliability": 0.15,
            "usability": 0.15
        }
        
        self.criteria_types = {
            "performance": {"type": "numeric", "higher_better": True},
            "cost": {"type": "numeric", "higher_better": False},
            "quality": {"type": "scale", "scale": [1, 10]},
            "reliability": {"type": "percentage", "range": [0, 100]},
            "usability": {"type": "categorical", "values": ["low", "medium", "high"]}
        }
    
    async def analyze_criteria(self, request: CriteriaRequest) -> Dict[str, Any]:
        """Analyze criteria and rank options"""
        
        try:
            # Normalize weights
            weights = request.weights or self.default_weights
            total_weight = sum(weights.values())
            normalized_weights = {k: v/total_weight for k, v in weights.items()}
            
            # Score each option
            option_scores = {}
            detailed_scores = {}
            
            for option in request.options:
                scores = self._score_option(option, request.criteria, normalized_weights)
                option_scores[option] = scores["total_score"]
                detailed_scores[option] = scores["detailed_scores"]
            
            # Rank options
            rankings = self._rank_options(option_scores, detailed_scores)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(rankings, request.criteria)
            
            return {
                "option_scores": option_scores,
                "detailed_scores": detailed_scores,
                "rankings": rankings,
                "recommendations": recommendations,
                "weights_used": normalized_weights,
                "criteria_analyzed": list(request.criteria.keys())
            }
            
        except Exception as e:
            logger.error(f"❌ Criteria analysis error: {e}")
            return self._emergency_analysis(request.options)
    
    def _score_option(self, option: str, criteria: Dict[str, Any], weights: Dict[str, float]) -> Dict[str, Any]:
        """Score a single option against criteria"""
        
        detailed_scores = {}
        total_score = 0.0
        
        for criterion_name, criterion_value in criteria.items():
            weight = weights.get(criterion_name, 0.1)
            
            # Normalize criterion score to 0-1 scale
            normalized_score = self._normalize_criterion_score(
                criterion_name, 
                criterion_value, 
                option
            )
            
            detailed_scores[criterion_name] = {
                "raw_value": criterion_value,
                "normalized_score": normalized_score,
                "weight": weight,
                "weighted_score": normalized_score * weight
            }
            
            total_score += normalized_score * weight
        
        return {
            "total_score": total_score,
            "detailed_scores": detailed_scores
        }
    
    def _normalize_criterion_score(self, criterion_name: str, criterion_value: Any, option: str) -> float:
        """Normalize criterion score to 0-1 scale"""
        
        # Get criterion type info
        criterion_info = self.criteria_types.get(criterion_name, {"type": "numeric", "higher_better": True})
        
        if criterion_info["type"] == "numeric":
            # For numeric values, assume they're already in reasonable range
            if isinstance(criterion_value, (int, float)):
                normalized = min(1.0, max(0.0, criterion_value / 100.0))
                if not criterion_info.get("higher_better", True):
                    normalized = 1.0 - normalized
                return normalized
        
        elif criterion_info["type"] == "scale":
            # For scale values (e.g., 1-10 rating)
            scale = criterion_info.get("scale", [1, 10])
            if isinstance(criterion_value, (int, float)):
                return (criterion_value - scale[0]) / (scale[1] - scale[0])
        
        elif criterion_info["type"] == "percentage":
            # For percentage values
            if isinstance(criterion_value, (int, float)):
                return criterion_value / 100.0
        
        elif criterion_info["type"] == "categorical":
            # For categorical values
            values = criterion_info.get("values", ["low", "medium", "high"])
            if criterion_value in values:
                return values.index(criterion_value) / (len(values) - 1)
        
        # Default fallback
        return 0.5
    
    def _rank_options(self, option_scores: Dict[str, float], detailed_scores: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank options by score"""
        
        rankings = []
        
        # Sort by score (highest first)
        sorted_options = sorted(option_scores.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (option, score) in enumerate(sorted_options, 1):
            rankings.append({
                "rank": rank,
                "option": option,
                "total_score": score,
                "score_percentage": score * 100,
                "detailed_breakdown": detailed_scores.get(option, {}),
                "strengths": self._identify_strengths(detailed_scores.get(option, {})),
                "weaknesses": self._identify_weaknesses(detailed_scores.get(option, {}))
            })
        
        return rankings
    
    def _identify_strengths(self, detailed_scores: Dict[str, Any]) -> List[str]:
        """Identify strengths of an option"""
        
        strengths = []
        
        for criterion, scores in detailed_scores.items():
            if scores.get("normalized_score", 0) > 0.7:
                strengths.append(f"Strong {criterion} ({scores['normalized_score']:.2f})")
        
        return strengths
    
    def _identify_weaknesses(self, detailed_scores: Dict[str, Any]) -> List[str]:
        """Identify weaknesses of an option"""
        
        weaknesses = []
        
        for criterion, scores in detailed_scores.items():
            if scores.get("normalized_score", 0) < 0.4:
                weaknesses.append(f"Weak {criterion} ({scores['normalized_score']:.2f})")
        
        return weaknesses
    
    def _generate_recommendations(self, rankings: List[Dict[str, Any]], criteria: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis"""
        
        recommendations = []
        
        if not rankings:
            return ["No options to evaluate"]
        
        top_option = rankings[0]
        
        # Primary recommendation
        recommendations.append(
            f"Recommend '{top_option['option']}' with {top_option['score_percentage']:.1f}% score"
        )
        
        # Strength-based recommendation
        if top_option.get("strengths"):
            recommendations.append(
                f"Top choice excels in: {', '.join(top_option['strengths'][:2])}"
            )
        
        # Alternative recommendation
        if len(rankings) > 1:
            second_option = rankings[1]
            score_diff = top_option['score_percentage'] - second_option['score_percentage']
            
            if score_diff < 10:  # Close competition
                recommendations.append(
                    f"Consider '{second_option['option']}' as close alternative ({second_option['score_percentage']:.1f}%)"
                )
        
        # Improvement recommendations
        if top_option.get("weaknesses"):
            recommendations.append(
                f"Monitor weaknesses: {', '.join(top_option['weaknesses'][:2])}"
            )
        
        return recommendations
    
    def _emergency_analysis(self, options: List[str]) -> Dict[str, Any]:
        """Emergency fallback analysis"""
        
        return {
            "option_scores": {option: 0.5 for option in options},
            "rankings": [{"rank": i+1, "option": option, "total_score": 0.5} for i, option in enumerate(options)],
            "recommendations": ["Emergency analysis - manual review recommended"],
            "weights_used": self.default_weights,
            "error": "Emergency fallback used"
        }

# Initialize analyzer
criteria_analyzer = CriteriaAnalyzer()

@criteria_router.post("/analyze", response_model=CriteriaResponse)
async def analyze_criteria(request: CriteriaRequest):
    """Analyze criteria and rank options"""
    
    start_time = time.time()
    
    try:
        # Generate criteria ID
        criteria_id = f"crit_{int(time.time())}_{hash(str(request.criteria)) % 1000}"
        
        # Perform analysis
        analysis = await criteria_analyzer.analyze_criteria(request)
        
        processing_time = time.time() - start_time
        
        response = CriteriaResponse(
            criteria_id=criteria_id,
            analysis_result=analysis,
            scores=analysis.get("option_scores", {}),
            rankings=analysis.get("rankings", []),
            recommendations=analysis.get("recommendations", []),
            processing_time_seconds=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"🎯 Criteria analysis completed: {len(request.options)} options analyzed")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Criteria analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Criteria analysis failed: {str(e)}")

@criteria_router.get("/types")
async def get_criteria_types():
    """Get supported criteria types"""
    
    return {
        "criteria_types": criteria_analyzer.criteria_types,
        "default_weights": criteria_analyzer.default_weights,
        "timestamp": datetime.now().isoformat()
    }

@criteria_router.post("/compare")
async def compare_options(option1: str, option2: str, criteria: Dict[str, Any]):
    """Compare two options directly"""
    
    try:
        request = CriteriaRequest(
            criteria=criteria,
            options=[option1, option2]
        )
        
        analysis = await criteria_analyzer.analyze_criteria(request)
        
        rankings = analysis.get("rankings", [])
        if len(rankings) >= 2:
            winner = rankings[0]
            runner_up = rankings[1]
            
            return {
                "winner": winner["option"],
                "winner_score": winner["total_score"],
                "runner_up": runner_up["option"],
                "runner_up_score": runner_up["total_score"],
                "score_difference": winner["total_score"] - runner_up["total_score"],
                "detailed_comparison": {
                    winner["option"]: winner["detailed_breakdown"],
                    runner_up["option"]: runner_up["detailed_breakdown"]
                },
                "recommendation": f"{winner['option']} performs better by {((winner['total_score'] - runner_up['total_score']) * 100):.1f} percentage points"
            }
        else:
            return {"error": "Insufficient data for comparison"}
            
    except Exception as e:
        logger.error(f"❌ Option comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))