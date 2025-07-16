# backend/services/05-enhanced-intelligence/slf-framework/app/reasoning_validator.py
"""
Reasoning Validator
Advanced reasoning validation and logical consistency checking
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

# Create router
reasoning_router = APIRouter()

# Models
class ReasoningRequest(BaseModel):
    reasoning_content: str = Field(..., min_length=1, max_length=20000)
    validation_type: str = Field(default="comprehensive", description="basic, logical, comprehensive, formal")
    reasoning_domain: str = Field(default="general", description="general, scientific, business, philosophical, technical")
    strictness_level: str = Field(default="moderate", description="lenient, moderate, strict, rigorous")
    context: Optional[Dict[str, Any]] = Field(None)

class ReasoningResponse(BaseModel):
    validation_id: str
    original_reasoning: str
    validation_results: Dict[str, Any]
    logical_consistency_score: float
    reasoning_quality_score: float
    identified_issues: List[Dict[str, Any]]
    improvement_suggestions: List[str]
    validation_summary: str
    processing_time_seconds: float
    timestamp: str

class ReasoningValidator:
    """Advanced Reasoning Validation Engine"""
    
    def __init__(self):
        self.validation_criteria = {
            "logical_consistency": {
                "weight": 0.30,
                "checks": ["contradiction_detection", "premise_conclusion_alignment", "logical_flow"],
                "description": "Internal logical consistency and coherence"
            },
            "argument_structure": {
                "weight": 0.25,
                "checks": ["premise_identification", "conclusion_validity", "supporting_evidence"],
                "description": "Quality of argument structure and support"
            },
            "evidence_quality": {
                "weight": 0.20,
                "checks": ["evidence_relevance", "evidence_sufficiency", "source_credibility"],
                "description": "Quality and adequacy of supporting evidence"
            },
            "reasoning_clarity": {
                "weight": 0.15,
                "checks": ["clarity_of_expression", "terminology_consistency", "step_transparency"],
                "description": "Clarity and transparency of reasoning process"
            },
            "completeness": {
                "weight": 0.10,
                "checks": ["coverage_adequacy", "missing_considerations", "thoroughness"],
                "description": "Completeness of reasoning coverage"
            }
        }
        
        self.logical_fallacies = {
            "ad_hominem": {
                "description": "Attacking the person rather than the argument",
                "indicators": ["personal attack", "character assassination", "irrelevant personal criticism"],
                "severity": "medium"
            },
            "straw_man": {
                "description": "Misrepresenting opponent's argument",
                "indicators": ["oversimplification", "distortion", "mischaracterization"],
                "severity": "medium"
            },
            "false_dichotomy": {
                "description": "Presenting only two options when more exist",
                "indicators": ["either/or", "only two choices", "binary thinking"],
                "severity": "medium"
            },
            "circular_reasoning": {
                "description": "Using conclusion as premise",
                "indicators": ["repetitive logic", "begging the question", "circular argument"],
                "severity": "high"
            },
            "hasty_generalization": {
                "description": "Drawing broad conclusions from limited evidence",
                "indicators": ["insufficient sample", "overgeneralization", "broad claims"],
                "severity": "medium"
            },
            "appeal_to_authority": {
                "description": "Relying on authority rather than evidence",
                "indicators": ["expert says", "authority claims", "appeal to position"],
                "severity": "low"
            },
            "false_cause": {
                "description": "Assuming causation from correlation",
                "indicators": ["correlation implies causation", "post hoc", "causal assumption"],
                "severity": "high"
            }
        }
        
        self.reasoning_patterns = {
            "deductive": {
                "structure": ["general_premise", "specific_premise", "logical_conclusion"],
                "validation": "premise_truth_and_logical_validity",
                "strength": "certainty_if_premises_true"
            },
            "inductive": {
                "structure": ["specific_observations", "pattern_identification", "general_conclusion"],
                "validation": "sample_adequacy_and_pattern_strength",
                "strength": "probability_based_on_evidence"
            },
            "abductive": {
                "structure": ["observation", "hypothesis_formation", "best_explanation"],
                "validation": "explanation_adequacy_and_alternative_consideration",
                "strength": "best_available_explanation"
            },
            "analogical": {
                "structure": ["source_case", "target_case", "similarity_mapping", "inference"],
                "validation": "similarity_relevance_and_mapping_accuracy",
                "strength": "degree_of_relevant_similarity"
            }
        }
        
        self.domain_specific_criteria = {
            "scientific": {
                "additional_checks": ["hypothesis_testing", "empirical_support", "reproducibility"],
                "evidence_standards": "empirical_data",
                "reasoning_rigor": "high"
            },
            "business": {
                "additional_checks": ["market_evidence", "financial_viability", "stakeholder_impact"],
                "evidence_standards": "business_data",
                "reasoning_rigor": "moderate"
            },
            "philosophical": {
                "additional_checks": ["conceptual_clarity", "logical_rigor", "assumption_examination"],
                "evidence_standards": "logical_argument",
                "reasoning_rigor": "very_high"
            },
            "technical": {
                "additional_checks": ["technical_accuracy", "implementation_feasibility", "specification_clarity"],
                "evidence_standards": "technical_data",
                "reasoning_rigor": "high"
            }
        }
    
    async def validate_reasoning(self, request: ReasoningRequest) -> Dict[str, Any]:
        """Validate reasoning using comprehensive framework"""
        
        try:
            # Analyze reasoning structure
            structure_analysis = self._analyze_reasoning_structure(
                request.reasoning_content,
                request.reasoning_domain
            )
            
            # Check logical consistency
            consistency_results = self._check_logical_consistency(
                request.reasoning_content,
                structure_analysis
            )
            
            # Validate argument quality
            argument_quality = self._validate_argument_quality(
                request.reasoning_content,
                request.strictness_level
            )
            
            # Check for logical fallacies
            fallacy_detection = self._detect_logical_fallacies(
                request.reasoning_content
            )
            
            # Domain-specific validation
            domain_validation = self._apply_domain_validation(
                request.reasoning_content,
                request.reasoning_domain,
                structure_analysis
            )
            
            # Calculate overall scores
            scores = self._calculate_validation_scores(
                consistency_results,
                argument_quality,
                fallacy_detection,
                domain_validation
            )
            
            # Generate improvement suggestions
            improvements = self._generate_improvement_suggestions(
                consistency_results,
                argument_quality,
                fallacy_detection,
                domain_validation
            )
            
            # Compile validation results
            validation_results = {
                "structure_analysis": structure_analysis,
                "consistency_results": consistency_results,
                "argument_quality": argument_quality,
                "fallacy_detection": fallacy_detection,
                "domain_validation": domain_validation,
                "scores": scores,
                "improvements": improvements
            }
            
            return validation_results
            
        except Exception as e:
            logger.error(f"❌ Reasoning validation error: {e}")
            return self._emergency_validation(request.reasoning_content)
    
    def _analyze_reasoning_structure(self, reasoning_content: str, domain: str) -> Dict[str, Any]:
        """Analyze the structure and pattern of reasoning"""
        
        analysis = {
            "reasoning_pattern": "mixed",
            "argument_components": [],
            "logical_flow": "basic",
            "complexity_level": "medium",
            "structure_quality": "acceptable"
        }
        
        content_lower = reasoning_content.lower()
        
        # Identify reasoning pattern
        if any(indicator in content_lower for indicator in ["all", "every", "therefore", "thus"]):
            analysis["reasoning_pattern"] = "deductive"
        elif any(indicator in content_lower for indicator in ["evidence suggests", "pattern shows", "likely"]):
            analysis["reasoning_pattern"] = "inductive"
        elif any(indicator in content_lower for indicator in ["best explanation", "hypothesis", "most likely"]):
            analysis["reasoning_pattern"] = "abductive"
        elif any(indicator in content_lower for indicator in ["similar to", "like", "analogous"]):
            analysis["reasoning_pattern"] = "analogical"
        
        # Identify argument components
        if "because" in content_lower or "since" in content_lower:
            analysis["argument_components"].append("causal_reasoning")
        if "evidence" in content_lower or "data" in content_lower:
            analysis["argument_components"].append("evidence_based")
        if "conclude" in content_lower or "therefore" in content_lower:
            analysis["argument_components"].append("explicit_conclusion")
        if "assume" in content_lower or "given" in content_lower:
            analysis["argument_components"].append("premise_identification")
        
        # Assess logical flow
        flow_indicators = ["first", "second", "then", "next", "finally", "therefore", "thus", "consequently"]
        flow_count = sum(1 for indicator in flow_indicators if indicator in content_lower)
        
        if flow_count >= 3:
            analysis["logical_flow"] = "excellent"
        elif flow_count >= 2:
            analysis["logical_flow"] = "good"
        elif flow_count >= 1:
            analysis["logical_flow"] = "basic"
        else:
            analysis["logical_flow"] = "poor"
        
        # Assess complexity
        sentences = [s.strip() for s in reasoning_content.split('.') if s.strip()]
        words = reasoning_content.split()
        
        if len(sentences) > 10 and len(words) > 200:
            analysis["complexity_level"] = "high"
        elif len(sentences) > 5 and len(words) > 100:
            analysis["complexity_level"] = "medium"
        else:
            analysis["complexity_level"] = "low"
        
        # Overall structure quality
        quality_factors = [
            len(analysis["argument_components"]) >= 2,
            analysis["logical_flow"] in ["good", "excellent"],
            "conclusion" in content_lower or "therefore" in content_lower
        ]
        
        quality_score = sum(quality_factors)
        if quality_score >= 3:
            analysis["structure_quality"] = "excellent"
        elif quality_score >= 2:
            analysis["structure_quality"] = "good"
        elif quality_score >= 1:
            analysis["structure_quality"] = "acceptable"
        else:
            analysis["structure_quality"] = "poor"
        
        return analysis
    
    def _check_logical_consistency(self, reasoning_content: str, structure_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Check for logical consistency and coherence"""
        
        consistency = {
            "overall_consistency": "consistent",
            "contradictions_found": [],
            "logical_gaps": [],
            "consistency_score": 0.8
        }
        
        content_lower = reasoning_content.lower()
        sentences = [s.strip() for s in reasoning_content.split('.') if s.strip()]
        
        # Check for explicit contradictions
        contradiction_patterns = [
            ("always", "never"),
            ("all", "none"),
            ("impossible", "possible"),
            ("certain", "uncertain"),
            ("true", "false")
        ]
        
        for positive, negative in contradiction_patterns:
            if positive in content_lower and negative in content_lower:
                consistency["contradictions_found"].append({
                    "type": "explicit_contradiction",
                    "terms": [positive, negative],
                    "severity": "high"
                })
        
        # Check for logical gaps
        has_premises = any(word in content_lower for word in ["because", "since", "given", "assume"])
        has_conclusion = any(word in content_lower for word in ["therefore", "thus", "conclude", "result"])
        
        if has_conclusion and not has_premises:
            consistency["logical_gaps"].append({
                "type": "missing_premises",
                "description": "Conclusion present without clear supporting premises"
            })
        
        if has_premises and not has_conclusion:
            consistency["logical_gaps"].append({
                "type": "missing_conclusion",
                "description": "Premises present without clear conclusion"
            })
        
        # Check for circular reasoning
        key_terms = [word for word in content_lower.split() if len(word) > 5]
        if len(key_terms) > 0:
            # Simple check for repeated key concepts in premise and conclusion
            conclusion_indicators = ["therefore", "thus", "conclude"]
            for indicator in conclusion_indicators:
                if indicator in content_lower:
                    parts = content_lower.split(indicator)
                    if len(parts) == 2:
                        premise_part = parts[0]
                        conclusion_part = parts[1]
                        
                        # Check if conclusion uses same key terms as premises
                        premise_terms = set(premise_part.split())
                        conclusion_terms = set(conclusion_part.split())
                        overlap = len(premise_terms.intersection(conclusion_terms))
                        
                        if overlap > len(conclusion_terms) * 0.7:  # High overlap suggests circularity
                            consistency["logical_gaps"].append({
                                "type": "potential_circular_reasoning",
                                "description": "Conclusion may be using premises as support"
                            })
        
        # Calculate consistency score
        base_score = 0.8
        base_score -= len(consistency["contradictions_found"]) * 0.2
        base_score -= len(consistency["logical_gaps"]) * 0.1
        
        consistency["consistency_score"] = max(0.0, min(1.0, base_score))
        
        if consistency["consistency_score"] < 0.5:
            consistency["overall_consistency"] = "inconsistent"
        elif consistency["consistency_score"] < 0.7:
            consistency["overall_consistency"] = "somewhat_consistent"
        
        return consistency
    
    def _validate_argument_quality(self, reasoning_content: str, strictness_level: str) -> Dict[str, Any]:
        """Validate the quality of arguments presented"""
        
        quality = {
            "argument_strength": "moderate",
            "evidence_quality": "adequate",
            "support_adequacy": "sufficient",
            "quality_score": 0.7,
            "quality_issues": []
        }
        
        content_lower = reasoning_content.lower()
        
        # Evidence quality assessment
        evidence_indicators = ["evidence", "data", "research", "study", "statistics", "facts"]
        evidence_count = sum(1 for indicator in evidence_indicators if indicator in content_lower)
        
        if evidence_count >= 3:
            quality["evidence_quality"] = "strong"
        elif evidence_count >= 1:
            quality["evidence_quality"] = "adequate"
        else:
            quality["evidence_quality"] = "weak"
            quality["quality_issues"].append("insufficient_evidence")
        
        # Support adequacy
        support_indicators = ["supports", "demonstrates", "proves", "shows", "indicates"]
        support_count = sum(1 for indicator in support_indicators if indicator in content_lower)
        
        if support_count >= 2:
            quality["support_adequacy"] = "strong"
        elif support_count >= 1:
            quality["support_adequacy"] = "sufficient"
        else:
            quality["support_adequacy"] = "insufficient"
            quality["quality_issues"].append("weak_support_connections")
        
        # Argument strength based on structure
        has_clear_thesis = any(phrase in content_lower for phrase in ["argue that", "claim that", "position is"])
        has_multiple_points = content_lower.count("first") + content_lower.count("second") + content_lower.count("third") >= 2
        has_counterargument = any(phrase in content_lower for phrase in ["however", "although", "critics argue"])
        
        strength_factors = sum([has_clear_thesis, has_multiple_points, has_counterargument])
        
        if strength_factors >= 3:
            quality["argument_strength"] = "strong"
        elif strength_factors >= 2:
            quality["argument_strength"] = "moderate"
        else:
            quality["argument_strength"] = "weak"
            quality["quality_issues"].append("weak_argument_structure")
        
        # Calculate quality score
        quality_weights = {
            "strong": 1.0,
            "moderate": 0.7,
            "adequate": 0.7,
            "sufficient": 0.7,
            "weak": 0.4,
            "insufficient": 0.4
        }
        
        evidence_score = quality_weights.get(quality["evidence_quality"], 0.5)
        support_score = quality_weights.get(quality["support_adequacy"], 0.5)
        argument_score = quality_weights.get(quality["argument_strength"], 0.5)
        
        quality["quality_score"] = (evidence_score + support_score + argument_score) / 3
        
        # Adjust for strictness level
        strictness_adjustments = {
            "lenient": 1.1,
            "moderate": 1.0,
            "strict": 0.9,
            "rigorous": 0.8
        }
        
        adjustment = strictness_adjustments.get(strictness_level, 1.0)
        quality["quality_score"] *= adjustment
        quality["quality_score"] = min(1.0, quality["quality_score"])
        
        return quality
    
    def _detect_logical_fallacies(self, reasoning_content: str) -> Dict[str, Any]:
        """Detect logical fallacies in reasoning"""
        
        detection = {
            "fallacies_found": [],
            "fallacy_count": 0,
            "severity_assessment": "none",
            "fallacy_impact_score": 0.0
        }
        
        content_lower = reasoning_content.lower()
        
        # Check for each fallacy type
        for fallacy_name, fallacy_info in self.logical_fallacies.items():
            indicators = fallacy_info["indicators"]
            severity = fallacy_info["severity"]
            
            for indicator in indicators:
                if indicator in content_lower:
                    detection["fallacies_found"].append({
                        "fallacy_type": fallacy_name,
                        "description": fallacy_info["description"],
                        "indicator_found": indicator,
                        "severity": severity,
                        "confidence": 0.6  # Basic detection confidence
                    })
                    break  # Only report each fallacy type once
        
        detection["fallacy_count"] = len(detection["fallacies_found"])
        
        # Assess overall severity
        if detection["fallacy_count"] == 0:
            detection["severity_assessment"] = "none"
        else:
            severities = [f["severity"] for f in detection["fallacies_found"]]
            if "high" in severities:
                detection["severity_assessment"] = "high"
            elif "medium" in severities:
                detection["severity_assessment"] = "medium"
            else:
                detection["severity_assessment"] = "low"
        
        # Calculate fallacy impact score
        severity_weights = {"low": 0.2, "medium": 0.5, "high": 0.8}
        total_impact = sum(severity_weights.get(f["severity"], 0.3) for f in detection["fallacies_found"])
        detection["fallacy_impact_score"] = min(1.0, total_impact / max(1, len(detection["fallacies_found"])))
        
        return detection
    
    def _apply_domain_validation(self, reasoning_content: str, domain: str, 
                                structure_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Apply domain-specific validation criteria"""
        
        validation = {
            "domain_compliance": "satisfactory",
            "domain_specific_issues": [],
            "domain_strengths": [],
            "domain_score": 0.7
        }
        
        if domain == "general":
            return validation
        
        domain_criteria = self.domain_specific_criteria.get(domain, {})
        additional_checks = domain_criteria.get("additional_checks", [])
        
        content_lower = reasoning_content.lower()
        
        # Domain-specific validation
        if domain == "scientific":
            # Check for empirical support
            empirical_terms = ["data", "experiment", "observation", "measurement", "empirical"]
            if any(term in content_lower for term in empirical_terms):
                validation["domain_strengths"].append("empirical_support")
            else:
                validation["domain_specific_issues"].append("lacks_empirical_support")
            
            # Check for hypothesis testing
            if "hypothesis" in content_lower or "prediction" in content_lower:
                validation["domain_strengths"].append("hypothesis_driven")
        
        elif domain == "business":
            # Check for market evidence
            business_terms = ["market", "customer", "revenue", "cost", "profit", "roi"]
            if any(term in content_lower for term in business_terms):
                validation["domain_strengths"].append("business_relevance")
            else:
                validation["domain_specific_issues"].append("lacks_business_context")
            
            # Check for stakeholder consideration
            if "stakeholder" in content_lower or "impact" in content_lower:
                validation["domain_strengths"].append("stakeholder_awareness")
        
        elif domain == "philosophical":
            # Check for conceptual clarity
            philosophical_terms = ["concept", "definition", "meaning", "assumption", "principle"]
            if any(term in content_lower for term in philosophical_terms):
                validation["domain_strengths"].append("conceptual_clarity")
            else:
                validation["domain_specific_issues"].append("lacks_conceptual_clarity")
        
        elif domain == "technical":
            # Check for technical accuracy
            technical_terms = ["specification", "implementation", "requirement", "system", "process"]
            if any(term in content_lower for term in technical_terms):
                validation["domain_strengths"].append("technical_focus")
            else:
                validation["domain_specific_issues"].append("lacks_technical_specificity")
        
        # Calculate domain score
        strengths_count = len(validation["domain_strengths"])
        issues_count = len(validation["domain_specific_issues"])
        
        domain_score = 0.7 + (strengths_count * 0.1) - (issues_count * 0.15)
        validation["domain_score"] = max(0.0, min(1.0, domain_score))
        
        # Overall compliance assessment
        if validation["domain_score"] >= 0.8:
            validation["domain_compliance"] = "excellent"
        elif validation["domain_score"] >= 0.6:
            validation["domain_compliance"] = "satisfactory"
        else:
            validation["domain_compliance"] = "needs_improvement"
        
        return validation
    
    def _calculate_validation_scores(self, consistency_results: Dict[str, Any],
                                   argument_quality: Dict[str, Any], fallacy_detection: Dict[str, Any],
                                   domain_validation: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall validation scores"""
        
        # Extract individual scores
        consistency_score = consistency_results.get("consistency_score", 0.5)
        quality_score = argument_quality.get("quality_score", 0.5)
        fallacy_penalty = fallacy_detection.get("fallacy_impact_score", 0.0)
        domain_score = domain_validation.get("domain_score", 0.7)
        
        # Calculate weighted scores
        weights = self.validation_criteria
        
        logical_consistency = consistency_score * weights["logical_consistency"]["weight"]
        argument_structure = quality_score * weights["argument_structure"]["weight"]
        evidence_quality = quality_score * weights["evidence_quality"]["weight"]
        reasoning_clarity = (consistency_score + quality_score) / 2 * weights["reasoning_clarity"]["weight"]
        completeness = domain_score * weights["completeness"]["weight"]
        
        # Overall reasoning quality score
        reasoning_quality = logical_consistency + argument_structure + evidence_quality + reasoning_clarity + completeness
        
        # Apply fallacy penalty
        reasoning_quality *= (1.0 - fallacy_penalty * 0.3)  # Max 30% penalty for fallacies
        
        scores = {
            "logical_consistency_score": consistency_score,
            "reasoning_quality_score": max(0.0, min(1.0, reasoning_quality)),
            "individual_scores": {
                "logical_consistency": logical_consistency,
                "argument_structure": argument_structure,
                "evidence_quality": evidence_quality,
                "reasoning_clarity": reasoning_clarity,
                "completeness": completeness
            },
            "fallacy_penalty_applied": fallacy_penalty * 0.3
        }
        
        return scores
    
    def _generate_improvement_suggestions(self, consistency_results: Dict[str, Any],
                                        argument_quality: Dict[str, Any], fallacy_detection: Dict[str, Any],
                                        domain_validation: Dict[str, Any]) -> List[str]:
        """Generate specific improvement suggestions"""
        
        suggestions = []
        
        # Consistency improvements
        if consistency_results.get("consistency_score", 1.0) < 0.7:
            suggestions.append("Improve logical consistency by checking for contradictions")
        
        if consistency_results.get("logical_gaps"):
            suggestions.append("Address logical gaps by providing missing premises or conclusions")
        
        # Quality improvements
        quality_issues = argument_quality.get("quality_issues", [])
        for issue in quality_issues:
            if issue == "insufficient_evidence":
                suggestions.append("Strengthen arguments with more evidence and supporting data")
            elif issue == "weak_support_connections":
                suggestions.append("Clarify how evidence supports your conclusions")
            elif issue == "weak_argument_structure":
                suggestions.append("Improve argument structure with clear thesis and supporting points")
        
        # Fallacy improvements
        fallacies = fallacy_detection.get("fallacies_found", [])
        for fallacy in fallacies:
            suggestions.append(f"Address {fallacy['fallacy_type'].replace('_', ' ')}: {fallacy['description']}")
        
        # Domain improvements
        domain_issues = domain_validation.get("domain_specific_issues", [])
        for issue in domain_issues:
            suggestions.append(f"Improve domain relevance: {issue.replace('_', ' ')}")
        
        return suggestions[:5]  # Limit to top 5 suggestions
    
    def _emergency_validation(self, reasoning_content: str) -> Dict[str, Any]:
        """Emergency fallback validation"""
        
        return {
            "structure_analysis": {"reasoning_pattern": "unknown", "structure_quality": "unknown"},
            "consistency_results": {"overall_consistency": "unknown", "consistency_score": 0.5},
            "argument_quality": {"argument_strength": "unknown", "quality_score": 0.5},
            "fallacy_detection": {"fallacies_found": [], "fallacy_count": 0},
            "domain_validation": {"domain_compliance": "unknown", "domain_score": 0.5},
            "scores": {"logical_consistency_score": 0.5, "reasoning_quality_score": 0.5},
            "improvements": ["Manual reasoning review recommended due to processing error"]
        }

# Initialize reasoning validator
reasoning_validator = ReasoningValidator()

@reasoning_router.post("/validate", response_model=ReasoningResponse)
async def validate_reasoning(request: ReasoningRequest):
    """Validate reasoning comprehensively"""
    
    start_time = time.time()
    
    try:
        # Generate validation ID
        validation_id = f"val_{int(time.time())}_{hash(request.reasoning_content) % 1000}"
        
        # Perform validation
        validation_result = await reasoning_validator.validate_reasoning(request)
        
        # Extract scores
        scores = validation_result.get("scores", {})
        
        # Compile issues
        issues = []
        
        # Add consistency issues
        contradictions = validation_result.get("consistency_results", {}).get("contradictions_found", [])
        for contradiction in contradictions:
            issues.append({
                "type": "logical_consistency",
                "issue": "contradiction",
                "description": f"Contradiction between {contradiction.get('terms', ['terms'])}",
                "severity": contradiction.get("severity", "medium")
            })
        
        # Add fallacy issues
        fallacies = validation_result.get("fallacy_detection", {}).get("fallacies_found", [])
        for fallacy in fallacies:
            issues.append({
                "type": "logical_fallacy",
                "issue": fallacy.get("fallacy_type", "unknown"),
                "description": fallacy.get("description", "Logical fallacy detected"),
                "severity": fallacy.get("severity", "medium")
            })
        
        # Generate validation summary
        logical_score = scores.get("logical_consistency_score", 0.5)
        quality_score = scores.get("reasoning_quality_score", 0.5)
        
        if quality_score >= 0.8:
            summary = "Strong reasoning with good logical structure"
        elif quality_score >= 0.6:
            summary = "Adequate reasoning with room for improvement"
        else:
            summary = "Reasoning needs significant improvement"
        
        processing_time = time.time() - start_time
        
        response = ReasoningResponse(
            validation_id=validation_id,
            original_reasoning=request.reasoning_content,
            validation_results=validation_result,
            logical_consistency_score=logical_score,
            reasoning_quality_score=quality_score,
            identified_issues=issues,
            improvement_suggestions=validation_result.get("improvements", []),
            validation_summary=summary,
            processing_time_seconds=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(
            f"🔍 Reasoning validated: {quality_score:.2f} quality score, "
            f"{len(issues)} issues found"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Reasoning validation error: {e}")
        raise HTTPException(status_code=500, detail=f"Reasoning validation failed: {str(e)}")

@reasoning_router.get("/criteria")
async def get_validation_criteria():
    """Get validation criteria and fallacy information"""
    
    return {
        "validation_criteria": reasoning_validator.validation_criteria,
        "logical_fallacies": reasoning_validator.logical_fallacies,
        "reasoning_patterns": reasoning_validator.reasoning_patterns,
        "domain_criteria": reasoning_validator.domain_specific_criteria,
        "timestamp": datetime.now().isoformat()
    }

@reasoning_router.post("/quick")
async def quick_reasoning_check(reasoning_text: str, domain: str = "general"):
    """Quick reasoning validation"""
    
    try:
        request = ReasoningRequest(
            reasoning_content=reasoning_text,
            validation_type="basic",
            reasoning_domain=domain,
            strictness_level="moderate"
        )
        
        validation_result = await reasoning_validator.validate_reasoning(request)
        scores = validation_result.get("scores", {})
        
        # Quick assessment
        quality_score = scores.get("reasoning_quality_score", 0.5)
        fallacy_count = validation_result.get("fallacy_detection", {}).get("fallacy_count", 0)
        
        if quality_score >= 0.8 and fallacy_count == 0:
            assessment = "Strong reasoning"
        elif quality_score >= 0.6 and fallacy_count <= 1:
            assessment = "Good reasoning"
        elif quality_score >= 0.4:
            assessment = "Adequate reasoning"
        else:
            assessment = "Weak reasoning"
        
        return {
            "quality_score": quality_score,
            "consistency_score": scores.get("logical_consistency_score", 0.5),
            "fallacies_detected": fallacy_count,
            "assessment": assessment,
            "top_issue": validation_result.get("improvements", ["No issues found"])[0],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Quick reasoning check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))