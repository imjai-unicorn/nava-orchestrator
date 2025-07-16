# backend/services/05-enhanced-intelligence/decision-engine/app/risk_assessor.py
"""
Risk Assessor
Advanced risk assessment for decision making
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

# Create router
risk_router = APIRouter()

# Models
class RiskRequest(BaseModel):
    decision: str = Field(..., description="Decision to assess risk for")
    context: Optional[Dict[str, Any]] = Field(None)
    risk_categories: Optional[List[str]] = Field(None)
    impact_areas: Optional[List[str]] = Field(None)
    timeline: str = Field(default="immediate", description="immediate, short, medium, long")

class RiskResponse(BaseModel):
    risk_id: str
    decision: str
    overall_risk_level: str
    risk_score: float
    risk_categories: List[Dict[str, Any]]
    mitigation_strategies: List[str]
    monitoring_points: List[str]
    escalation_triggers: List[str]
    processing_time_seconds: float
    timestamp: str

class RiskAssessor:
    """Advanced Risk Assessment Engine"""
    
    def __init__(self):
        self.risk_categories = {
            "technical": {
                "factors": ["complexity", "dependencies", "technology_maturity", "scalability"],
                "indicators": ["system_failure", "performance_degradation", "integration_issues"],
                "weight": 0.25
            },
            "operational": {
                "factors": ["resources", "timeline", "process_maturity", "team_capability"],
                "indicators": ["delays", "resource_shortage", "process_failure"],
                "weight": 0.20
            },
            "financial": {
                "factors": ["budget", "cost_overrun", "roi_uncertainty", "market_conditions"],
                "indicators": ["budget_exceeded", "poor_roi", "market_volatility"],
                "weight": 0.20
            },
            "security": {
                "factors": ["data_exposure", "access_controls", "compliance", "vulnerabilities"],
                "indicators": ["data_breach", "unauthorized_access", "compliance_violation"],
                "weight": 0.15
            },
            "business": {
                "factors": ["market_acceptance", "competitive_response", "strategic_alignment"],
                "indicators": ["poor_adoption", "competitive_threat", "strategic_misalignment"],
                "weight": 0.20
            }
        }
        
        self.risk_levels = {
            "low": {"threshold": 0.3, "color": "green", "action": "proceed"},
            "medium": {"threshold": 0.6, "color": "yellow", "action": "monitor"},
            "high": {"threshold": 0.8, "color": "orange", "action": "mitigate"},
            "critical": {"threshold": 1.0, "color": "red", "action": "immediate_action"}
        }
        
        self.timeline_multipliers = {
            "immediate": 1.2,  # Higher risk for immediate decisions
            "short": 1.0,
            "medium": 0.9,
            "long": 0.8
        }
    
    async def assess_risk(self, request: RiskRequest) -> Dict[str, Any]:
        """Perform comprehensive risk assessment"""
        
        try:
            # Analyze each risk category
            category_assessments = {}
            for category, config in self.risk_categories.items():
                if not request.risk_categories or category in request.risk_categories:
                    assessment = self._assess_category(
                        category,
                        config,
                        request.decision,
                        request.context or {}
                    )
                    category_assessments[category] = assessment
            
            # Calculate overall risk
            overall_risk = self._calculate_overall_risk(
                category_assessments,
                request.timeline
            )
            
            # Generate mitigation strategies
            mitigation = self._generate_mitigation_strategies(
                category_assessments,
                overall_risk
            )
            
            # Identify monitoring points
            monitoring = self._identify_monitoring_points(
                category_assessments,
                request.decision
            )
            
            # Set escalation triggers
            escalation = self._set_escalation_triggers(
                overall_risk,
                category_assessments
            )
            
            return {
                "category_assessments": category_assessments,
                "overall_risk": overall_risk,
                "mitigation": mitigation,
                "monitoring": monitoring,
                "escalation": escalation
            }
            
        except Exception as e:
            logger.error(f"❌ Risk assessment error: {e}")
            return self._emergency_assessment(request.decision)
    
    def _assess_category(self, category: str, config: Dict[str, Any], decision: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk for a specific category"""
        
        factors = config["factors"]
        indicators = config["indicators"]
        
        # Base risk calculation
        risk_score = 0.3  # Base risk
        identified_risks = []
        risk_factors = []
        
        # Decision-based risk analysis
        decision_lower = decision.lower()
        
        if category == "technical":
            if any(word in decision_lower for word in ["new", "experimental", "complex"]):
                risk_score += 0.2
                identified_risks.append("technology_complexity")
            
            if any(word in decision_lower for word in ["integrate", "dependency", "external"]):
                risk_score += 0.15
                identified_risks.append("integration_dependency")
        
        elif category == "operational":
            priority = context.get("priority", "normal")
            if priority in ["high", "urgent"]:
                risk_score += 0.2
                identified_risks.append("time_pressure")
            
            team_size = context.get("team_size", "medium")
            if team_size == "small":
                risk_score += 0.15
                identified_risks.append("resource_limitation")
        
        elif category == "financial":
            budget_status = context.get("budget", "adequate")
            if budget_status in ["tight", "limited"]:
                risk_score += 0.3
                identified_risks.append("budget_constraint")
        
        elif category == "security":
            if any(word in decision_lower for word in ["data", "user", "external", "api"]):
                risk_score += 0.2
                identified_risks.append("data_exposure_risk")
        
        elif category == "business":
            if any(word in decision_lower for word in ["change", "new", "different"]):
                risk_score += 0.15
                identified_risks.append("change_resistance")
        
        # Context-based adjustments
        complexity = context.get("complexity", "medium")
        if complexity == "high":
            risk_score += 0.1
        elif complexity == "low":
            risk_score -= 0.1
        
        # Normalize risk score
        risk_score = max(0.0, min(1.0, risk_score))
        
        # Determine risk level
        risk_level = self._determine_risk_level(risk_score)
        
        return {
            "category": category,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "identified_risks": identified_risks,
            "risk_factors": factors,
            "potential_indicators": indicators,
            "weight": config["weight"],
            "weighted_score": risk_score * config["weight"]
        }
    
    def _calculate_overall_risk(self, assessments: Dict[str, Any], timeline: str) -> Dict[str, Any]:
        """Calculate overall risk score and level"""
        
        if not assessments:
            return {"score": 0.5, "level": "medium"}
        
        # Weighted average
        total_weighted_score = sum(assessment["weighted_score"] for assessment in assessments.values())
        total_weight = sum(assessment["weight"] for assessment in assessments.values())
        
        overall_score = total_weighted_score / max(total_weight, 0.1)
        
        # Timeline adjustment
        timeline_multiplier = self.timeline_multipliers.get(timeline, 1.0)
        adjusted_score = min(1.0, overall_score * timeline_multiplier)
        
        # Determine overall level
        overall_level = self._determine_risk_level(adjusted_score)
        
        return {
            "score": adjusted_score,
            "level": overall_level,
            "timeline_adjusted": timeline_multiplier != 1.0,
            "category_count": len(assessments),
            "highest_category": max(assessments.items(), key=lambda x: x[1]["risk_score"])[0]
        }
    
    def _determine_risk_level(self, score: float) -> str:
        """Determine risk level from score"""
        
        for level, config in self.risk_levels.items():
            if score <= config["threshold"]:
                return level
        
        return "critical"
    
    def _generate_mitigation_strategies(self, assessments: Dict[str, Any], overall_risk: Dict[str, Any]) -> List[str]:
        """Generate risk mitigation strategies"""
        
        strategies = []
        
        # Overall strategies based on risk level
        risk_level = overall_risk.get("level", "medium")
        
        if risk_level in ["high", "critical"]:
            strategies.append("Implement comprehensive risk monitoring dashboard")
            strategies.append("Establish daily risk review meetings")
            strategies.append("Prepare detailed rollback procedures")
        
        if risk_level in ["medium", "high", "critical"]:
            strategies.append("Create risk escalation matrix")
            strategies.append("Implement checkpoint reviews")
        
        # Category-specific strategies
        for category, assessment in assessments.items():
            if assessment["risk_score"] > 0.6:
                if category == "technical":
                    strategies.append("Conduct technical proof-of-concept")
                    strategies.append("Implement comprehensive testing strategy")
                
                elif category == "operational":
                    strategies.append("Increase resource allocation")
                    strategies.append("Extend timeline with buffer periods")
                
                elif category == "financial":
                    strategies.append("Establish contingency budget")
                    strategies.append("Implement cost tracking and controls")
                
                elif category == "security":
                    strategies.append("Conduct security audit and penetration testing")
                    strategies.append("Implement enhanced access controls")
                
                elif category == "business":
                    strategies.append("Develop stakeholder communication plan")
                    strategies.append("Create user training and adoption strategy")
        
        return list(set(strategies))  # Remove duplicates
    
    def _identify_monitoring_points(self, assessments: Dict[str, Any], decision: str) -> List[str]:
        """Identify key monitoring points"""
        
        monitoring_points = []
        
        # Standard monitoring points
        monitoring_points.extend([
            "Progress against timeline",
            "Budget consumption rate",
            "Quality metrics tracking"
        ])
        
        # Category-specific monitoring
        for category, assessment in assessments.items():
            if assessment["risk_score"] > 0.5:
                if category == "technical":
                    monitoring_points.extend([
                        "System performance metrics",
                        "Error rate tracking",
                        "Integration test results"
                    ])
                
                elif category == "operational":
                    monitoring_points.extend([
                        "Resource utilization",
                        "Team velocity",
                        "Milestone completion"
                    ])
                
                elif category == "security":
                    monitoring_points.extend([
                        "Security incident tracking",
                        "Access log monitoring",
                        "Compliance audit results"
                    ])
        
        return list(set(monitoring_points))
    
    def _set_escalation_triggers(self, overall_risk: Dict[str, Any], assessments: Dict[str, Any]) -> List[str]:
        """Set escalation triggers"""
        
        triggers = []
        risk_level = overall_risk.get("level", "medium")
        
        # Standard triggers
        triggers.extend([
            "Timeline delay > 20%",
            "Budget overrun > 15%",
            "Quality metrics below threshold"
        ])
        
        # Risk level specific triggers
        if risk_level in ["high", "critical"]:
            triggers.extend([
                "Any critical issue identified",
                "Risk score increase > 0.2",
                "Multiple risk categories elevated"
            ])
        
        # Category specific triggers
        high_risk_categories = [cat for cat, assess in assessments.items() if assess["risk_score"] > 0.7]
        
        for category in high_risk_categories:
            if category == "technical":
                triggers.append("System failure or critical bug")
            elif category == "security":
                triggers.append("Security incident or breach")
            elif category == "financial":
                triggers.append("Budget variance > 25%")
        
        return list(set(triggers))
    
    def _emergency_assessment(self, decision: str) -> Dict[str, Any]:
        """Emergency fallback assessment"""
        
        return {
            "category_assessments": {
                "general": {
                    "risk_score": 0.5,
                    "risk_level": "medium",
                    "identified_risks": ["insufficient_analysis"]
                }
            },
            "overall_risk": {"score": 0.5, "level": "medium"},
            "mitigation": ["Manual risk review required"],
            "monitoring": ["General progress tracking"],
            "escalation": ["Escalate if issues arise"]
        }

# Initialize risk assessor
risk_assessor = RiskAssessor()

@risk_router.post("/assess", response_model=RiskResponse)
async def assess_risk(request: RiskRequest):
    """Perform comprehensive risk assessment"""
    
    start_time = time.time()
    
    try:
        # Generate risk ID
        risk_id = f"risk_{int(time.time())}_{hash(request.decision) % 1000}"
        
        # Perform assessment
        assessment = await risk_assessor.assess_risk(request)
        
        # Extract results
        overall_risk = assessment.get("overall_risk", {})
        category_assessments = assessment.get("category_assessments", {})
        
        processing_time = time.time() - start_time
        
        response = RiskResponse(
            risk_id=risk_id,
            decision=request.decision,
            overall_risk_level=overall_risk.get("level", "medium"),
            risk_score=overall_risk.get("score", 0.5),
            risk_categories=[
                {
                    "category": cat,
                    "risk_score": data["risk_score"],
                    "risk_level": data["risk_level"],
                    "identified_risks": data["identified_risks"]
                }
                for cat, data in category_assessments.items()
            ],
            mitigation_strategies=assessment.get("mitigation", []),
            monitoring_points=assessment.get("monitoring", []),
            escalation_triggers=assessment.get("escalation", []),
            processing_time_seconds=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(
            f"🛡️ Risk Assessment: {overall_risk.get('level', 'unknown')} "
            f"({overall_risk.get('score', 0):.2f}) for '{request.decision}'"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Risk assessment error: {e}")
        raise HTTPException(status_code=500, detail=f"Risk assessment failed: {str(e)}")

@risk_router.get("/categories")
async def get_risk_categories():
    """Get available risk categories"""
    
    return {
        "categories": risk_assessor.risk_categories,
        "risk_levels": risk_assessor.risk_levels,
        "timeline_options": list(risk_assessor.timeline_multipliers.keys()),
        "timestamp": datetime.now().isoformat()
    }

@risk_router.post("/quick")
async def quick_risk_check(decision: str, priority: str = "normal"):
    """Quick risk check for simple decisions"""
    
    try:
        request = RiskRequest(
            decision=decision,
            context={"priority": priority},
            timeline="immediate"
        )
        
        assessment = await risk_assessor.assess_risk(request)
        overall_risk = assessment.get("overall_risk", {})
        
        return {
            "decision": decision,
            "risk_level": overall_risk.get("level", "medium"),
            "risk_score": overall_risk.get("score", 0.5),
            "recommendation": risk_assessor.risk_levels.get(
                overall_risk.get("level", "medium"), {}
            ).get("action", "review"),
            "top_mitigation": assessment.get("mitigation", ["Review required"])[0] if assessment.get("mitigation") else "Review required",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Quick risk check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@risk_router.post("/matrix")
async def risk_matrix_analysis(decisions: List[str], context: Optional[Dict[str, Any]] = None):
    """Analyze multiple decisions in risk matrix"""
    
    try:
        results = []
        
        for decision in decisions[:10]:  # Limit to 10 decisions
            request = RiskRequest(
                decision=decision,
                context=context or {},
                timeline="short"
            )
            
            assessment = await risk_assessor.assess_risk(request)
            overall_risk = assessment.get("overall_risk", {})
            
            results.append({
                "decision": decision,
                "risk_score": overall_risk.get("score", 0.5),
                "risk_level": overall_risk.get("level", "medium"),
                "action_required": risk_assessor.risk_levels.get(
                    overall_risk.get("level", "medium"), {}
                ).get("action", "review")
            })
        
        # Sort by risk score (highest first)
        results.sort(key=lambda x: x["risk_score"], reverse=True)
        
        return {
            "matrix_analysis": results,
            "summary": {
                "total_decisions": len(results),
                "high_risk_count": sum(1 for r in results if r["risk_level"] in ["high", "critical"]),
                "medium_risk_count": sum(1 for r in results if r["risk_level"] == "medium"),
                "low_risk_count": sum(1 for r in results if r["risk_level"] == "low"),
                "highest_risk": results[0] if results else None,
                "recommendations": [
                    f"Prioritize mitigation for {sum(1 for r in results if r['risk_level'] in ['high', 'critical'])} high-risk decisions",
                    f"Monitor {sum(1 for r in results if r['risk_level'] == 'medium')} medium-risk decisions",
                    f"Proceed with {sum(1 for r in results if r['risk_level'] == 'low')} low-risk decisions"
                ]
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Risk matrix analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))