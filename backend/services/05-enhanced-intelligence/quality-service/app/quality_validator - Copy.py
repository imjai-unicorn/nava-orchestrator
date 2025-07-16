# backend/services/05-enhanced-intelligence/quality-service/app/quality_validator.py
"""
Quality Validator Core
Multi-dimensional response quality validation
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging
import time
import re
from datetime import datetime

logger = logging.getLogger(__name__)

# Create router
quality_router = APIRouter()

# Models
class QualityRequest(BaseModel):
    response_text: str = Field(..., min_length=1, max_length=50000)
    original_query: Optional[str] = Field(None)
    context: Optional[Dict[str, Any]] = Field(None)
    quality_requirements: Optional[Dict[str, str]] = Field(None)
    model_used: Optional[str] = Field(None)
    response_time: Optional[float] = Field(None)

class QualityResponse(BaseModel):
    quality_id: str
    overall_score: float
    quality_level: str
    dimension_scores: Dict[str, float]
    quality_metrics: Dict[str, Any]
    compliance_status: Dict[str, bool]
    improvement_areas: List[str]
    passed_thresholds: Dict[str, bool]
    processing_time_seconds: float
    timestamp: str

class QualityValidator:
    """Advanced Quality Validation Engine"""
    
    def __init__(self):
        self.quality_dimensions = {
            "accuracy": {
                "weight": 0.25,
                "description": "Factual correctness and relevance",
                "checks": ["factual_consistency", "relevance_to_query", "information_accuracy"]
            },
            "completeness": {
                "weight": 0.20,
                "description": "Comprehensive coverage of topic",
                "checks": ["topic_coverage", "detail_level", "missing_information"]
            },
            "clarity": {
                "weight": 0.20,
                "description": "Clear and understandable communication",
                "checks": ["readability", "structure", "language_clarity"]
            },
            "relevance": {
                "weight": 0.15,
                "description": "Direct relevance to user query",
                "checks": ["query_alignment", "context_appropriateness", "user_intent_match"]
            },
            "safety": {
                "weight": 0.10,
                "description": "Safety and appropriateness",
                "checks": ["harmful_content", "bias_detection", "inappropriate_content"]
            },
            "compliance": {
                "weight": 0.10,
                "description": "Enterprise compliance requirements",
                "checks": ["policy_adherence", "regulatory_compliance", "enterprise_standards"]
            }
        }
        
        self.enterprise_thresholds = {
            "minimum_quality": 0.75,
            "safety_threshold": 0.95,
            "compliance_threshold": 0.90,
            "accuracy_threshold": 0.80,
            "clarity_threshold": 0.70
        }
        
        self.quality_levels = {
            "excellent": {"min_score": 0.90, "description": "Exceptional quality"},
            "good": {"min_score": 0.75, "description": "High quality, meets standards"},
            "acceptable": {"min_score": 0.60, "description": "Adequate quality"},
            "poor": {"min_score": 0.40, "description": "Below standards, needs improvement"},
            "unacceptable": {"min_score": 0.0, "description": "Critical quality issues"}
        }
    
    async def validate_quality(self, request: QualityRequest) -> Dict[str, Any]:
        """Perform comprehensive quality validation"""
        
        try:
            # Analyze each quality dimension
            dimension_scores = {}
            detailed_metrics = {}
            
            for dimension, config in self.quality_dimensions.items():
                score, metrics = self._analyze_dimension(
                    dimension,
                    config,
                    request.response_text,
                    request.original_query,
                    request.context or {}
                )
                dimension_scores[dimension] = score
                detailed_metrics[dimension] = metrics
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(dimension_scores)
            
            # Determine quality level
            quality_level = self._determine_quality_level(overall_score)
            
            # Check compliance
            compliance_status = self._check_compliance(
                dimension_scores,
                request.quality_requirements or {}
            )
            
            # Identify improvement areas
            improvement_areas = self._identify_improvement_areas(
                dimension_scores,
                detailed_metrics
            )
            
            # Check thresholds
            threshold_results = self._check_thresholds(dimension_scores, overall_score)
            
            return {
                "dimension_scores": dimension_scores,
                "detailed_metrics": detailed_metrics,
                "overall_score": overall_score,
                "quality_level": quality_level,
                "compliance_status": compliance_status,
                "improvement_areas": improvement_areas,
                "threshold_results": threshold_results
            }
            
        except Exception as e:
            logger.error(f"❌ Quality validation error: {e}")
            return self._emergency_validation(request.response_text)
    
    def _analyze_dimension(self, dimension: str, config: Dict[str, Any], response_text: str, 
                          original_query: Optional[str], context: Dict[str, Any]) -> tuple:
        """Analyze a specific quality dimension"""
        
        score = 0.5  # Base score
        metrics = {}
        
        if dimension == "accuracy":
            score, metrics = self._analyze_accuracy(response_text, original_query, context)
        
        elif dimension == "completeness":
            score, metrics = self._analyze_completeness(response_text, original_query, context)
        
        elif dimension == "clarity":
            score, metrics = self._analyze_clarity(response_text, context)
        
        elif dimension == "relevance":
            score, metrics = self._analyze_relevance(response_text, original_query, context)
        
        elif dimension == "safety":
            score, metrics = self._analyze_safety(response_text, context)
        
        elif dimension == "compliance":
            score, metrics = self._analyze_compliance(response_text, context)
        
        return score, metrics
    
    def _analyze_accuracy(self, response_text: str, original_query: Optional[str], context: Dict[str, Any]) -> tuple:
        """Analyze accuracy dimension"""
        
        score = 0.7  # Base accuracy score
        metrics = {}
        
        # Length-based accuracy indicators
        response_length = len(response_text)
        if response_length < 50:
            score -= 0.2
            metrics["length_concern"] = "Response too short"
        elif response_length > 5000:
            score -= 0.1
            metrics["length_concern"] = "Response very long"
        
        # Query alignment (if original query provided)
        if original_query:
            query_words = set(original_query.lower().split())
            response_words = set(response_text.lower().split())
            overlap = len(query_words.intersection(response_words)) / max(len(query_words), 1)
            
            if overlap > 0.5:
                score += 0.2
                metrics["query_alignment"] = "Good"
            elif overlap < 0.2:
                score -= 0.2
                metrics["query_alignment"] = "Poor"
            else:
                metrics["query_alignment"] = "Moderate"
        
        # Factual consistency indicators
        has_numbers = bool(re.search(r'\d+', response_text))
        has_specific_info = bool(re.search(r'\b(specific|exact|precisely|according to)\b', response_text.lower()))
        
        if has_numbers and has_specific_info:
            score += 0.1
            metrics["specificity"] = "High"
        
        # Uncertainty indicators (good for accuracy)
        uncertainty_phrases = ["might", "could", "possibly", "perhaps", "likely"]
        uncertainty_count = sum(1 for phrase in uncertainty_phrases if phrase in response_text.lower())
        
        if uncertainty_count > 0:
            score += 0.05  # Appropriate uncertainty is good
            metrics["uncertainty_handling"] = "Appropriate"
        
        metrics["estimated_accuracy"] = score
        return max(0.0, min(1.0, score)), metrics
    
    def _analyze_completeness(self, response_text: str, original_query: Optional[str], context: Dict[str, Any]) -> tuple:
        """Analyze completeness dimension"""
        
        score = 0.6  # Base completeness score
        metrics = {}
        
        # Response structure analysis
        sentences = response_text.split('.')
        paragraphs = response_text.split('\n\n')
        
        metrics["sentence_count"] = len(sentences)
        metrics["paragraph_count"] = len(paragraphs)
        
        # Length-based completeness
        response_length = len(response_text)
        if response_length > 500:
            score += 0.2
        elif response_length > 200:
            score += 0.1
        elif response_length < 100:
            score -= 0.3
        
        # Content structure indicators
        has_examples = bool(re.search(r'\b(example|for instance|such as|like)\b', response_text.lower()))
        has_explanations = bool(re.search(r'\b(because|since|due to|explanation)\b', response_text.lower()))
        has_details = bool(re.search(r'\b(detail|specific|particular|comprehensive)\b', response_text.lower()))
        
        structure_score = sum([has_examples, has_explanations, has_details]) * 0.1
        score += structure_score
        
        metrics["has_examples"] = has_examples
        metrics["has_explanations"] = has_explanations
        metrics["has_details"] = has_details
        
        # Query coverage (if query provided)
        if original_query:
            query_length = len(original_query)
            coverage_ratio = response_length / max(query_length, 1)
            
            if coverage_ratio > 5:  # Good coverage
                score += 0.15
                metrics["coverage"] = "Comprehensive"
            elif coverage_ratio < 2:  # Poor coverage
                score -= 0.15
                metrics["coverage"] = "Limited"
            else:
                metrics["coverage"] = "Moderate"
        
        metrics["completeness_score"] = score
        return max(0.0, min(1.0, score)), metrics
    
    def _analyze_clarity(self, response_text: str, context: Dict[str, Any]) -> tuple:
        """Analyze clarity dimension"""
        
        score = 0.7  # Base clarity score
        metrics = {}
        
        # Readability indicators
        words = response_text.split()
        sentences = [s for s in response_text.split('.') if s.strip()]
        
        if sentences:
            avg_sentence_length = len(words) / len(sentences)
            metrics["avg_sentence_length"] = avg_sentence_length
            
            # Optimal sentence length (10-20 words)
            if 10 <= avg_sentence_length <= 20:
                score += 0.15
            elif avg_sentence_length > 30:
                score -= 0.2
                metrics["readability_concern"] = "Sentences too long"
            elif avg_sentence_length < 5:
                score -= 0.1
                metrics["readability_concern"] = "Sentences too short"
        
        # Structure indicators
        has_bullet_points = '•' in response_text or '- ' in response_text
        has_numbers = bool(re.search(r'\d+\.', response_text))
        has_headers = bool(re.search(r'^[A-Z][^.]*:$', response_text, re.MULTILINE))
        
        structure_elements = sum([has_bullet_points, has_numbers, has_headers])
        if structure_elements > 0:
            score += structure_elements * 0.05
            metrics["structure_elements"] = structure_elements
        
        # Language clarity
        complex_words = len([w for w in words if len(w) > 10])
        complexity_ratio = complex_words / max(len(words), 1)
        
        if complexity_ratio < 0.1:
            score += 0.1
            metrics["language_complexity"] = "Simple"
        elif complexity_ratio > 0.3:
            score -= 0.15
            metrics["language_complexity"] = "Complex"
        
        # Transition words (good for clarity)
        transitions = ["however", "therefore", "furthermore", "additionally", "consequently"]
        transition_count = sum(1 for trans in transitions if trans in response_text.lower())
        
        if transition_count > 0:
            score += min(0.1, transition_count * 0.03)
            metrics["transitions"] = transition_count
        
        metrics["clarity_score"] = score
        return max(0.0, min(1.0, score)), metrics
    
    def _analyze_relevance(self, response_text: str, original_query: Optional[str], context: Dict[str, Any]) -> tuple:
        """Analyze relevance dimension"""
        
        score = 0.6  # Base relevance score
        metrics = {}
        
        if not original_query:
            metrics["note"] = "No original query provided for relevance analysis"
            return score, metrics
        
        # Direct query word matching
        query_words = set(original_query.lower().split())
        response_words = set(response_text.lower().split())
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        query_words -= stop_words
        response_words -= stop_words
        
        if query_words:
            word_overlap = len(query_words.intersection(response_words)) / len(query_words)
            score += word_overlap * 0.3
            metrics["word_overlap"] = word_overlap
        
        # Query intent matching
        query_lower = original_query.lower()
        response_lower = response_text.lower()
        
        # Question types
        if query_lower.startswith(('what', 'how', 'why', 'when', 'where', 'who')):
            question_type = query_lower.split()[0]
            
            if question_type == "what" and ("is" in response_lower or "are" in response_lower):
                score += 0.15
            elif question_type == "how" and ("steps" in response_lower or "process" in response_lower):
                score += 0.15
            elif question_type == "why" and ("because" in response_lower or "reason" in response_lower):
                score += 0.15
            
            metrics["question_type_match"] = f"{question_type}_handled"
        
        # Context relevance
        context_keywords = context.get("keywords", [])
        if context_keywords:
            context_matches = sum(1 for keyword in context_keywords if keyword.lower() in response_lower)
            context_score = context_matches / len(context_keywords)
            score += context_score * 0.2
            metrics["context_relevance"] = context_score
        
        metrics["relevance_score"] = score
        return max(0.0, min(1.0, score)), metrics
    
    def _analyze_safety(self, response_text: str, context: Dict[str, Any]) -> tuple:
        """Analyze safety dimension"""
        
        score = 0.95  # Start with high safety score
        metrics = {}
        
        response_lower = response_text.lower()
        
        # Harmful content detection (basic)
        harmful_indicators = [
            "violence", "harm", "dangerous", "illegal", "inappropriate",
            "offensive", "hate", "discrimination", "explicit"
        ]
        
        harmful_count = sum(1 for indicator in harmful_indicators if indicator in response_lower)
        if harmful_count > 0:
            score -= harmful_count * 0.2
            metrics["harmful_content_detected"] = harmful_count
        
        # Bias detection (basic)
        bias_indicators = [
            "always", "never", "all", "none", "everyone", "no one"
        ]
        
        bias_count = sum(1 for indicator in bias_indicators if f" {indicator} " in f" {response_lower} ")
        if bias_count > 2:
            score -= 0.1
            metrics["potential_bias"] = bias_count
        
        # Professional tone check
        unprofessional_words = ["stupid", "dumb", "idiotic", "crazy", "insane"]
        unprofessional_count = sum(1 for word in unprofessional_words if word in response_lower)
        
        if unprofessional_count > 0:
            score -= unprofessional_count * 0.15
            metrics["unprofessional_language"] = unprofessional_count
        
        # Positive safety indicators
        disclaimer_phrases = ["please note", "it's important to", "be careful", "consult with"]
        disclaimer_count = sum(1 for phrase in disclaimer_phrases if phrase in response_lower)
        
        if disclaimer_count > 0:
            score += min(0.05, disclaimer_count * 0.02)
            metrics["safety_disclaimers"] = disclaimer_count
        
        metrics["safety_score"] = score
        return max(0.0, min(1.0, score)), metrics
    
    def _analyze_compliance(self, response_text: str, context: Dict[str, Any]) -> tuple:
        """Analyze compliance dimension"""
        
        score = 0.8  # Base compliance score
        metrics = {}
        
        # Enterprise compliance indicators
        response_lower = response_text.lower()
        
        # Professional language
        professional_indicators = [
            "please", "thank you", "kindly", "respectfully", "professional"
        ]
        professional_count = sum(1 for indicator in professional_indicators if indicator in response_lower)
        
        if professional_count > 0:
            score += min(0.1, professional_count * 0.03)
            metrics["professional_language"] = professional_count
        
        # Policy compliance (basic check)
        policy_violations = [
            "confidential", "internal only", "do not share", "classified"
        ]
        violation_count = sum(1 for violation in policy_violations if violation in response_lower)
        
        if violation_count > 0:
            score -= violation_count * 0.3
            metrics["policy_violations"] = violation_count
        
        # Data privacy compliance
        privacy_concerns = [
            "personal information", "private data", "confidential data"
        ]
        privacy_mentions = sum(1 for concern in privacy_concerns if concern in response_lower)
        
        if privacy_mentions > 0:
            # Check if proper handling is mentioned
            proper_handling = any(phrase in response_lower for phrase in [
                "protect", "secure", "privacy", "confidential"
            ])
            
            if proper_handling:
                score += 0.05
                metrics["privacy_awareness"] = "Good"
            else:
                score -= 0.1
                metrics["privacy_awareness"] = "Insufficient"
        
        # Regulatory compliance indicators
        regulatory_phrases = [
            "according to regulations", "compliance with", "regulatory requirements"
        ]
        regulatory_count = sum(1 for phrase in regulatory_phrases if phrase in response_lower)
        
        if regulatory_count > 0:
            score += regulatory_count * 0.05
            metrics["regulatory_awareness"] = regulatory_count
        
        metrics["compliance_score"] = score
        return max(0.0, min(1.0, score)), metrics
    
    def _calculate_overall_score(self, dimension_scores: Dict[str, float]) -> float:
        """Calculate weighted overall quality score"""
        
        total_score = 0.0
        total_weight = 0.0
        
        for dimension, score in dimension_scores.items():
            weight = self.quality_dimensions.get(dimension, {}).get("weight", 0.1)
            total_score += score * weight
            total_weight += weight
        
        return total_score / max(total_weight, 0.1)
    
    def _determine_quality_level(self, overall_score: float) -> str:
        """Determine quality level from overall score"""
        
        for level, config in sorted(self.quality_levels.items(), 
                                  key=lambda x: x[1]["min_score"], reverse=True):
            if overall_score >= config["min_score"]:
                return level
        
        return "unacceptable"
    
    def _check_compliance(self, dimension_scores: Dict[str, float], 
                         quality_requirements: Dict[str, str]) -> Dict[str, bool]:
        """Check compliance with enterprise thresholds"""
        
        compliance = {}
        
        # Check enterprise thresholds
        for threshold_name, threshold_value in self.enterprise_thresholds.items():
            if threshold_name == "minimum_quality":
                overall_score = self._calculate_overall_score(dimension_scores)
                compliance[threshold_name] = overall_score >= threshold_value
            elif threshold_name.endswith("_threshold"):
                dimension = threshold_name.replace("_threshold", "")
                if dimension in dimension_scores:
                    compliance[threshold_name] = dimension_scores[dimension] >= threshold_value
        
        # Check custom requirements
        for req_name, req_value in quality_requirements.items():
            try:
                req_threshold = float(req_value)
                if req_name in dimension_scores:
                    compliance[f"custom_{req_name}"] = dimension_scores[req_name] >= req_threshold
            except ValueError:
                pass  # Skip non-numeric requirements
        
        return compliance
    
    def _identify_improvement_areas(self, dimension_scores: Dict[str, float], 
                                   detailed_metrics: Dict[str, Any]) -> List[str]:
        """Identify areas for improvement"""
        
        improvements = []
        
        # Check each dimension against thresholds
        for dimension, score in dimension_scores.items():
            threshold_key = f"{dimension}_threshold"
            threshold = self.enterprise_thresholds.get(threshold_key, 0.7)
            
            if score < threshold:
                improvements.append(f"Improve {dimension} (current: {score:.2f}, target: {threshold:.2f})")
        
        # Check overall quality
        overall_score = self._calculate_overall_score(dimension_scores)
        if overall_score < self.enterprise_thresholds["minimum_quality"]:
            improvements.append(f"Overall quality below minimum standard ({overall_score:.2f})")
        
        # Specific improvements based on metrics
        for dimension, metrics in detailed_metrics.items():
            if "readability_concern" in metrics:
                improvements.append(f"Address readability: {metrics['readability_concern']}")
            
            if "harmful_content_detected" in metrics and metrics["harmful_content_detected"] > 0:
                improvements.append("Remove harmful content")
            
            if "policy_violations" in metrics and metrics["policy_violations"] > 0:
                improvements.append("Address policy violations")
        
        return improvements[:5]  # Limit to top 5 improvements
    
    def _check_thresholds(self, dimension_scores: Dict[str, float], 
                         overall_score: float) -> Dict[str, bool]:
        """Check if quality thresholds are met"""
        
        results = {}
        
        # Overall threshold
        results["minimum_quality"] = overall_score >= self.enterprise_thresholds["minimum_quality"]
        
        # Dimension thresholds
        for dimension, score in dimension_scores.items():
            threshold_key = f"{dimension}_threshold"
            if threshold_key in self.enterprise_thresholds:
                results[threshold_key] = score >= self.enterprise_thresholds[threshold_key]
        
        # Critical thresholds (must pass)
        results["safety_critical"] = dimension_scores.get("safety", 0) >= 0.9
        results["compliance_critical"] = dimension_scores.get("compliance", 0) >= 0.8
        
        return results
    
    def _emergency_validation(self, response_text: str) -> Dict[str, Any]:
        """Emergency fallback validation"""
        
        basic_score = 0.6 if len(response_text) > 50 else 0.3
        
        return {
            "dimension_scores": {
                "accuracy": basic_score,
                "completeness": basic_score,
                "clarity": basic_score,
                "relevance": basic_score,
                "safety": 0.9,  # Assume safe unless detected otherwise
                "compliance": 0.7
            },
            "overall_score": basic_score,
            "quality_level": "acceptable" if basic_score >= 0.6 else "poor",
            "compliance_status": {"emergency": True},
            "improvement_areas": ["Manual quality review required"],
            "threshold_results": {"emergency_mode": True}
        }

# Initialize validator
quality_validator = QualityValidator()

@quality_router.post("/validate", response_model=QualityResponse)
async def validate_quality(request: QualityRequest):
    """Perform comprehensive quality validation"""
    
    start_time = time.time()
    
    try:
        # Generate quality ID
        quality_id = f"qual_{int(time.time())}_{hash(request.response_text) % 1000}"
        
        # Perform validation
        validation = await quality_validator.validate_quality(request)
        
        processing_time = time.time() - start_time
        
        response = QualityResponse(
            quality_id=quality_id,
            overall_score=validation.get("overall_score", 0.5),
            quality_level=validation.get("quality_level", "acceptable"),
            dimension_scores=validation.get("dimension_scores", {}),
            quality_metrics=validation.get("detailed_metrics", {}),
            compliance_status=validation.get("compliance_status", {}),
            improvement_areas=validation.get("improvement_areas", []),
            passed_thresholds=validation.get("threshold_results", {}),
            processing_time_seconds=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(
            f"✅ Quality validation: {validation.get('quality_level', 'unknown')} "
            f"({validation.get('overall_score', 0):.2f})"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Quality validation error: {e}")
        raise HTTPException(status_code=500, detail=f"Quality validation failed: {str(e)}")

@quality_router.get("/dimensions")
async def get_quality_dimensions():
    """Get quality dimensions and their configurations"""
    
    return {
        "dimensions": quality_validator.quality_dimensions,
        "thresholds": quality_validator.enterprise_thresholds,
        "quality_levels": quality_validator.quality_levels,
        "timestamp": datetime.now().isoformat()
    }

@quality_router.post("/quick")
async def quick_quality_check(response_text: str, min_threshold: float = 0.75):
    """Quick quality check for simple validation"""
    
    try:
        request = QualityRequest(response_text=response_text)
        validation = await quality_validator.validate_quality(request)
        
        overall_score = validation.get("overall_score", 0.5)
        quality_level = validation.get("quality_level", "acceptable")
        
        return {
            "overall_score": overall_score,
            "quality_level": quality_level,
            "passes_threshold": overall_score >= min_threshold,
            "recommendation": "approve" if overall_score >= min_threshold else "review_needed",
            "top_issue": validation.get("improvement_areas", ["None"])[0] if validation.get("improvement_areas") else "None",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Quick quality check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@quality_router.post("/batch")
async def batch_quality_validation(responses: List[str], threshold: float = 0.75):
    """Validate multiple responses in batch"""
    
    try:
        results = []
        
        for i, response_text in enumerate(responses[:20]):  # Limit to 20 responses
            request = QualityRequest(response_text=response_text)
            validation = await quality_validator.validate_quality(request)
            
            results.append({
                "index": i,
                "overall_score": validation.get("overall_score", 0.5),
                "quality_level": validation.get("quality_level", "acceptable"),
                "passes_threshold": validation.get("overall_score", 0.5) >= threshold,
                "top_improvement": validation.get("improvement_areas", ["None"])[0] if validation.get("improvement_areas") else "None"
            })
        
        # Summary statistics
        scores = [r["overall_score"] for r in results]
        passed_count = sum(1 for r in results if r["passes_threshold"])
        
        return {
            "results": results,
            "summary": {
                "total_responses": len(results),
                "passed_threshold": passed_count,
                "pass_rate": passed_count / len(results) if results else 0,
                "average_score": sum(scores) / len(scores) if scores else 0,
                "min_score": min(scores) if scores else 0,
                "max_score": max(scores) if scores else 0
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Batch quality validation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))