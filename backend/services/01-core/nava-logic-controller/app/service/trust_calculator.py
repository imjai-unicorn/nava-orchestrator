# backend/services/shared/common/trust_calculator.py
"""
Trust Calculator
Advanced trust scoring and reliability assessment system
"""

import logging
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)

class TrustDimension(str, Enum):
    """Trust evaluation dimensions"""
    ACCURACY = "accuracy"           # Factual correctness
    RELIABILITY = "reliability"     # Consistency over time
    SAFETY = "safety"              # Safety and bias assessment
    COMPLETENESS = "completeness"   # Response completeness
    RELEVANCE = "relevance"        # Relevance to user needs
    TRANSPARENCY = "transparency"   # Decision explainability
    ROBUSTNESS = "robustness"      # Performance under stress
    COMPLIANCE = "compliance"      # Regulatory compliance
    CONSISTENCY = "consistency"    # Internal logical consistency
    TIMELINESS = "timeliness"      # Response time performance

class TrustLevel(str, Enum):
    """Trust level classifications"""
    VERY_HIGH = "very_high"        # 90-100%
    HIGH = "high"                  # 80-89%
    MEDIUM = "medium"              # 60-79%
    LOW = "low"                    # 40-59%
    VERY_LOW = "very_low"          # Below 40%
    UNVERIFIED = "unverified"      # Insufficient data

class EvidenceType(str, Enum):
    """Types of trust evidence"""
    USER_FEEDBACK = "user_feedback"      # Direct user ratings
    PERFORMANCE_METRIC = "performance"   # System performance data
    VALIDATION_RESULT = "validation"     # Automated validation
    EXPERT_REVIEW = "expert_review"      # Expert assessment
    PEER_COMPARISON = "peer_comparison"  # Comparison with other models
    HISTORICAL_DATA = "historical"      # Historical performance
    COMPLIANCE_AUDIT = "compliance"     # Compliance verification
    STRESS_TEST = "stress_test"         # Performance under load

@dataclass
class TrustEvidence:
    """Evidence for trust calculation"""
    evidence_id: str
    evidence_type: EvidenceType
    dimension: TrustDimension
    value: float  # 0.0 to 1.0
    confidence: float  # Confidence in this evidence
    weight: float = 1.0  # Weight in calculation
    
    # Context
    source: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Temporal information
    timestamp: datetime = field(default_factory=datetime.now)
    validity_period: Optional[timedelta] = None
    
    # Metadata
    sample_size: Optional[int] = None
    methodology: Optional[str] = None
    verified: bool = False

@dataclass
class TrustScore:
    """Trust score for a specific dimension"""
    dimension: TrustDimension
    score: float  # 0.0 to 1.0
    confidence: float  # Confidence in the score
    level: TrustLevel
    
    # Supporting data
    evidence_count: int
    sample_size: int
    calculation_method: str
    
    # Temporal information
    calculated_at: datetime = field(default_factory=datetime.now)
    valid_until: Optional[datetime] = None
    
    # Breakdown
    contributing_factors: Dict[str, float] = field(default_factory=dict)
    uncertainty_factors: List[str] = field(default_factory=list)

@dataclass
class CompositeTrustScore:
    """Composite trust score across all dimensions"""
    overall_score: float
    overall_level: TrustLevel
    overall_confidence: float
    
    # Dimension scores
    dimension_scores: Dict[TrustDimension, TrustScore]
    
    # Calculation metadata
    calculation_method: str
    weights_used: Dict[TrustDimension, float]
    evidence_summary: Dict[EvidenceType, int]
    
    # Temporal information
    calculated_at: datetime = field(default_factory=datetime.now)
    
    # Risk assessment
    risk_factors: List[str] = field(default_factory=list)
    mitigation_suggestions: List[str] = field(default_factory=list)

class TrustCalculator:
    """Advanced trust scoring and reliability assessment system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Trust calculation settings
        self.dimension_weights = self._load_dimension_weights()
        self.evidence_weights = self._load_evidence_weights()
        self.decay_factors = self._load_decay_factors()
        
        # Evidence storage
        self.evidence_store: Dict[str, List[TrustEvidence]] = defaultdict(list)
        self.trust_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Configuration
        self.min_evidence_count = self.config.get('min_evidence_count', 5)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.temporal_window_days = self.config.get('temporal_window_days', 30)
        
        logger.info("🔒 Trust Calculator initialized with advanced scoring algorithms")
    
    def add_evidence(self, 
                    entity_id: str,
                    evidence: TrustEvidence) -> bool:
        """Add trust evidence for an entity"""
        
        try:
            # Validate evidence
            if not self._validate_evidence(evidence):
                return False
            
            # Store evidence
            self.evidence_store[entity_id].append(evidence)
            
            # Maintain evidence window
            self._cleanup_old_evidence(entity_id)
            
            logger.debug(f"✅ Added trust evidence for {entity_id}: {evidence.dimension} = {evidence.value:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add evidence for {entity_id}: {e}")
            return False
    
    def calculate_trust_score(self, 
                            entity_id: str,
                            dimension: TrustDimension,
                            temporal_window: Optional[timedelta] = None) -> Optional[TrustScore]:
        """Calculate trust score for a specific dimension"""
        
        # Get relevant evidence
        evidence_list = self._get_relevant_evidence(entity_id, dimension, temporal_window)
        
        if len(evidence_list) < self.min_evidence_count:
            logger.debug(f"Insufficient evidence for {entity_id}.{dimension}: {len(evidence_list)} < {self.min_evidence_count}")
            return None
        
        # Calculate weighted score
        weighted_score, confidence = self._calculate_weighted_score(evidence_list)
        
        # Determine trust level
        trust_level = self._determine_trust_level(weighted_score)
        
        # Calculate contributing factors
        contributing_factors = self._analyze_contributing_factors(evidence_list)
        
        # Identify uncertainty factors
        uncertainty_factors = self._identify_uncertainty_factors(evidence_list, confidence)
        
        # Calculate validity period
        validity_period = self._calculate_validity_period(evidence_list)
        
        trust_score = TrustScore(
            dimension=dimension,
            score=weighted_score,
            confidence=confidence,
            level=trust_level,
            evidence_count=len(evidence_list),
            sample_size=sum(e.sample_size or 1 for e in evidence_list),
            calculation_method="weighted_temporal_evidence",
            valid_until=datetime.now() + validity_period if validity_period else None,
            contributing_factors=contributing_factors,
            uncertainty_factors=uncertainty_factors
        )
        
        # Store in history
        self.trust_history[f"{entity_id}:{dimension}"].append({
            'timestamp': datetime.now(),
            'score': weighted_score,
            'confidence': confidence,
            'evidence_count': len(evidence_list)
        })
        
        return trust_score
    
    def calculate_composite_trust(self, 
                                entity_id: str,
                                dimensions: Optional[List[TrustDimension]] = None,
                                custom_weights: Optional[Dict[TrustDimension, float]] = None) -> Optional[CompositeTrustScore]:
        """Calculate composite trust score across multiple dimensions"""
        
        dimensions = dimensions or list(TrustDimension)
        weights = custom_weights or self.dimension_weights
        
        # Calculate individual dimension scores
        dimension_scores = {}
        valid_scores = []
        
        for dimension in dimensions:
            score = self.calculate_trust_score(entity_id, dimension)
            if score and score.confidence >= self.confidence_threshold:
                dimension_scores[dimension] = score
                valid_scores.append((dimension, score.score, weights.get(dimension, 1.0)))
        
        if len(valid_scores) < 3:  # Need at least 3 dimensions for composite score
            logger.debug(f"Insufficient valid dimensions for composite trust: {len(valid_scores)}")
            return None
        
        # Calculate weighted composite score
        total_weighted_score = sum(score * weight for _, score, weight in valid_scores)
        total_weight = sum(weight for _, _, weight in valid_scores)
        composite_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Calculate composite confidence
        confidences = [dimension_scores[dim].confidence for dim, _, _ in valid_scores]
        composite_confidence = statistics.harmonic_mean(confidences) if confidences else 0.0
        
        # Determine overall trust level
        overall_level = self._determine_trust_level(composite_score)
        
        # Analyze evidence summary
        evidence_summary = self._summarize_evidence(entity_id, dimensions)
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(dimension_scores, entity_id)
        
        # Generate mitigation suggestions
        mitigation_suggestions = self._generate_mitigation_suggestions(dimension_scores, risk_factors)
        
        composite_trust = CompositeTrustScore(
            overall_score=composite_score,
            overall_level=overall_level,
            overall_confidence=composite_confidence,
            dimension_scores=dimension_scores,
            calculation_method="weighted_harmonic_composite",
            weights_used=weights,
            evidence_summary=evidence_summary,
            risk_factors=risk_factors,
            mitigation_suggestions=mitigation_suggestions
        )
        
        return composite_trust
    
    def track_trust_trend(self, 
                         entity_id: str,
                         dimension: TrustDimension,
                         lookback_days: int = 30) -> Dict[str, Any]:
        """Analyze trust score trends over time"""
        
        history_key = f"{entity_id}:{dimension}"
        history = list(self.trust_history[history_key])
        
        if len(history) < 2:
            return {"trend": "insufficient_data", "confidence": 0.0}
        
        # Filter by time window
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_history = [h for h in history if h['timestamp'] >= cutoff_date]
        
        if len(recent_history) < 2:
            return {"trend": "insufficient_recent_data", "confidence": 0.0}
        
        # Calculate trend
        scores = [h['score'] for h in recent_history]
        timestamps = [h['timestamp'].timestamp() for h in recent_history]
        
        # Linear regression for trend
        n = len(scores)
        sum_x = sum(timestamps)
        sum_y = sum(scores)
        sum_xy = sum(x * y for x, y in zip(timestamps, scores))
        sum_x2 = sum(x * x for x in timestamps)
        
        # Calculate slope (trend direction)
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            slope = 0
        else:
            slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        # Interpret trend
        if abs(slope) < 1e-10:  # Very small change
            trend_direction = "stable"
        elif slope > 0:
            trend_direction = "improving"
        else:
            trend_direction = "declining"
        
        # Calculate trend strength
        score_range = max(scores) - min(scores)
        trend_strength = min(1.0, abs(slope) * 86400 * 365)  # Normalize to per-year change
        
        # Calculate confidence in trend
        trend_confidence = min(1.0, len(recent_history) / 10)  # More data = higher confidence
        
        return {
            "trend": trend_direction,
            "strength": trend_strength,
            "confidence": trend_confidence,
            "slope": slope,
            "score_range": score_range,
            "data_points": len(recent_history),
            "current_score": scores[-1] if scores else 0.0,
            "score_change": scores[-1] - scores[0] if len(scores) >= 2 else 0.0
        }
    
    def compare_entities(self, 
                        entity_ids: List[str],
                        dimensions: Optional[List[TrustDimension]] = None) -> Dict[str, Any]:
        """Compare trust scores across multiple entities"""
        
        dimensions = dimensions or list(TrustDimension)
        comparison_results = {}
        
        # Calculate composite scores for all entities
        entity_scores = {}
        for entity_id in entity_ids:
            composite = self.calculate_composite_trust(entity_id, dimensions)
            if composite:
                entity_scores[entity_id] = composite
        
        if len(entity_scores) < 2:
            return {"error": "Insufficient entities with valid trust scores"}
        
        # Overall ranking
        ranked_entities = sorted(
            entity_scores.items(),
            key=lambda x: x[1].overall_score,
            reverse=True
        )
        
        comparison_results["ranking"] = [
            {
                "entity_id": entity_id,
                "overall_score": score.overall_score,
                "overall_level": score.overall_level,
                "confidence": score.overall_confidence
            }
            for entity_id, score in ranked_entities
        ]
        
        # Dimension-wise comparison
        dimension_comparison = {}
        for dimension in dimensions:
            dim_scores = {}
            for entity_id, composite in entity_scores.items():
                if dimension in composite.dimension_scores:
                    dim_scores[entity_id] = composite.dimension_scores[dimension].score
            
            if dim_scores:
                best_entity = max(dim_scores.keys(), key=lambda x: dim_scores[x])
                worst_entity = min(dim_scores.keys(), key=lambda x: dim_scores[x])
                
                dimension_comparison[dimension] = {
                    "best": {"entity": best_entity, "score": dim_scores[best_entity]},
                    "worst": {"entity": worst_entity, "score": dim_scores[worst_entity]},
                    "average": statistics.mean(dim_scores.values()),
                    "spread": max(dim_scores.values()) - min(dim_scores.values())
                }
        
        comparison_results["dimension_analysis"] = dimension_comparison
        
        # Statistical analysis
        all_scores = [score.overall_score for _, score in entity_scores.items()]
        comparison_results["statistics"] = {
            "mean": statistics.mean(all_scores),
            "median": statistics.median(all_scores),
            "std_dev": statistics.stdev(all_scores) if len(all_scores) > 1 else 0,
            "range": max(all_scores) - min(all_scores)
        }
        
        return comparison_results
    
    def get_trust_recommendations(self, 
                                entity_id: str,
                                target_level: TrustLevel = TrustLevel.HIGH) -> List[Dict[str, Any]]:
        """Generate recommendations to improve trust"""
        
        composite = self.calculate_composite_trust(entity_id)
        if not composite:
            return [{"recommendation": "Collect more evidence to establish baseline trust"}]
        
        recommendations = []
        target_score = self._trust_level_to_score(target_level)
        
        # Identify lowest scoring dimensions
        weak_dimensions = []
        for dimension, score in composite.dimension_scores.items():
            if score.score < target_score:
                weak_dimensions.append((dimension, score.score, target_score - score.score))
        
        # Sort by improvement needed
        weak_dimensions.sort(key=lambda x: x[2], reverse=True)
        
        # Generate specific recommendations
        for dimension, current_score, gap in weak_dimensions[:3]:  # Top 3 areas for improvement
            recommendations.append({
                "dimension": dimension,
                "current_score": current_score,
                "target_score": target_score,
                "improvement_needed": gap,
                "priority": "high" if gap > 0.3 else "medium" if gap > 0.15 else "low",
                "specific_actions": self._get_improvement_actions(dimension, current_score),
                "estimated_timeline": self._estimate_improvement_timeline(dimension, gap)
            })
        
        # General recommendations
        if composite.overall_confidence < 0.8:
            recommendations.append({
                "type": "data_collection",
                "recommendation": "Collect more evidence to increase confidence in trust assessment",
                "priority": "medium",
                "actions": ["Increase feedback collection", "Add automated validation", "Conduct expert reviews"]
            })
        
        return recommendations
    
    def _validate_evidence(self, evidence: TrustEvidence) -> bool:
        """Validate evidence data"""
        
        # Check value range
        if not (0.0 <= evidence.value <= 1.0):
            logger.warning(f"Evidence value out of range: {evidence.value}")
            return False
        
        # Check confidence range
        if not (0.0 <= evidence.confidence <= 1.0):
            logger.warning(f"Evidence confidence out of range: {evidence.confidence}")
            return False
        
        # Check required fields
        if not evidence.evidence_id or not evidence.source:
            logger.warning("Evidence missing required fields")
            return False
        
        return True
    
    def _get_relevant_evidence(self, 
                             entity_id: str,
                             dimension: TrustDimension,
                             temporal_window: Optional[timedelta] = None) -> List[TrustEvidence]:
        """Get relevant evidence for calculation"""
        
        all_evidence = self.evidence_store.get(entity_id, [])
        
        # Filter by dimension
        relevant_evidence = [e for e in all_evidence if e.dimension == dimension]
        
        # Apply temporal window
        if temporal_window:
            cutoff_date = datetime.now() - temporal_window
            relevant_evidence = [e for e in relevant_evidence if e.timestamp >= cutoff_date]
        else:
            # Default temporal window
            cutoff_date = datetime.now() - timedelta(days=self.temporal_window_days)
            relevant_evidence = [e for e in relevant_evidence if e.timestamp >= cutoff_date]
        
        # Sort by timestamp (newest first)
        relevant_evidence.sort(key=lambda x: x.timestamp, reverse=True)
        
        return relevant_evidence
    
    def _calculate_weighted_score(self, evidence_list: List[TrustEvidence]) -> Tuple[float, float]:
        """Calculate weighted score and confidence"""
        
        if not evidence_list:
            return 0.0, 0.0
        
        total_weighted_score = 0.0
        total_weight = 0.0
        confidence_scores = []
        
        for evidence in evidence_list:
            # Calculate temporal weight (newer evidence has higher weight)
            age_days = (datetime.now() - evidence.timestamp).days
            temporal_weight = self._calculate_temporal_weight(age_days)
            
            # Calculate evidence type weight
            type_weight = self.evidence_weights.get(evidence.evidence_type, 1.0)
            
            # Combined weight
            combined_weight = evidence.weight * temporal_weight * type_weight * evidence.confidence
            
            total_weighted_score += evidence.value * combined_weight
            total_weight += combined_weight
            confidence_scores.append(evidence.confidence)
        
        # Calculate final score
        final_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Calculate overall confidence (harmonic mean of individual confidences)
        overall_confidence = statistics.harmonic_mean(confidence_scores) if confidence_scores else 0.0
        
        # Adjust confidence based on evidence count
        evidence_count_factor = min(1.0, len(evidence_list) / 10)  # Full confidence at 10+ evidence points
        adjusted_confidence = overall_confidence * evidence_count_factor
        
        return final_score, adjusted_confidence
    
    def _calculate_temporal_weight(self, age_days: int) -> float:
        """Calculate temporal weight based on evidence age"""
        
        # Exponential decay function
        decay_rate = self.decay_factors.get("temporal_decay", 0.1)
        return math.exp(-decay_rate * age_days / 30)  # 30-day half-life
    
    def _determine_trust_level(self, score: float) -> TrustLevel:
        """Determine trust level from numeric score"""
        
        if score >= 0.90:
            return TrustLevel.VERY_HIGH
        elif score >= 0.80:
            return TrustLevel.HIGH
        elif score >= 0.60:
            return TrustLevel.MEDIUM
        elif score >= 0.40:
            return TrustLevel.LOW
        else:
            return TrustLevel.VERY_LOW
    
    def _trust_level_to_score(self, level: TrustLevel) -> float:
        """Convert trust level to numeric score"""
        
        level_scores = {
            TrustLevel.VERY_HIGH: 0.95,
            TrustLevel.HIGH: 0.85,
            TrustLevel.MEDIUM: 0.70,
            TrustLevel.LOW: 0.50,
            TrustLevel.VERY_LOW: 0.30,
            TrustLevel.UNVERIFIED: 0.0
        }
        
        return level_scores.get(level, 0.0)
    
    def _analyze_contributing_factors(self, evidence_list: List[TrustEvidence]) -> Dict[str, float]:
        """Analyze what factors contribute most to the trust score"""
        
        factor_contributions = defaultdict(float)
        total_weight = 0.0
        
        for evidence in evidence_list:
            weight = evidence.weight * evidence.confidence
            factor_contributions[f"evidence_type:{evidence.evidence_type}"] += weight
            factor_contributions[f"source:{evidence.source}"] += weight
            total_weight += weight
        
        # Normalize to percentages
        if total_weight > 0:
            for factor in factor_contributions:
                factor_contributions[factor] = factor_contributions[factor] / total_weight
        
        # Return top contributing factors
        sorted_factors = sorted(factor_contributions.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_factors[:5])  # Top 5 factors
    
    def _identify_uncertainty_factors(self, evidence_list: List[TrustEvidence], confidence: float) -> List[str]:
        """Identify factors that create uncertainty in the trust assessment"""
        
        uncertainty_factors = []
        
        # Low confidence evidence
        low_confidence_count = sum(1 for e in evidence_list if e.confidence < 0.7)
        if low_confidence_count > len(evidence_list) * 0.3:
            uncertainty_factors.append("High proportion of low-confidence evidence")
        
        # Old evidence
        old_evidence_count = sum(1 for e in evidence_list 
                                if (datetime.now() - e.timestamp).days > 60)
        if old_evidence_count > len(evidence_list) * 0.5:
            uncertainty_factors.append("Evidence may be outdated")
        
        # Limited evidence types
        evidence_types = set(e.evidence_type for e in evidence_list)
        if len(evidence_types) < 3:
            uncertainty_factors.append("Limited diversity in evidence types")
        
        # Small sample sizes
        small_sample_count = sum(1 for e in evidence_list 
                               if e.sample_size and e.sample_size < 10)
        if small_sample_count > len(evidence_list) * 0.4:
            uncertainty_factors.append("Small sample sizes in evidence")
        
        # Overall low confidence
        if confidence < 0.6:
            uncertainty_factors.append("Overall low confidence in assessment")
        
        return uncertainty_factors
    
    def _calculate_validity_period(self, evidence_list: List[TrustEvidence]) -> Optional[timedelta]:
        """Calculate how long the trust score remains valid"""
        
        if not evidence_list:
            return None
        
        # Base validity on newest evidence
        newest_evidence = max(evidence_list, key=lambda x: x.timestamp)
        
        # Evidence-type specific validity periods
        validity_periods = {
            EvidenceType.USER_FEEDBACK: timedelta(days=30),
            EvidenceType.PERFORMANCE_METRIC: timedelta(days=7),
            EvidenceType.VALIDATION_RESULT: timedelta(days=14),
            EvidenceType.EXPERT_REVIEW: timedelta(days=90),
            EvidenceType.COMPLIANCE_AUDIT: timedelta(days=365),
            EvidenceType.STRESS_TEST: timedelta(days=30)
        }
        
        return validity_periods.get(newest_evidence.evidence_type, timedelta(days=30))
    
    def _summarize_evidence(self, entity_id: str, dimensions: List[TrustDimension]) -> Dict[EvidenceType, int]:
        """Summarize evidence by type across dimensions"""
        
        evidence_summary = defaultdict(int)
        
        for dimension in dimensions:
            evidence_list = self._get_relevant_evidence(entity_id, dimension)
            for evidence in evidence_list:
                evidence_summary[evidence.evidence_type] += 1
        
        return dict(evidence_summary)
    
    def _identify_risk_factors(self, dimension_scores: Dict[TrustDimension, TrustScore], entity_id: str) -> List[str]:
        """Identify trust-related risk factors"""
        
        risk_factors = []
        
        # Low scores in critical dimensions
        critical_dimensions = [TrustDimension.SAFETY, TrustDimension.COMPLIANCE, TrustDimension.ACCURACY]
        for dimension in critical_dimensions:
            if dimension in dimension_scores:
                score = dimension_scores[dimension]
                if score.score < 0.7:
                    risk_factors.append(f"Low {dimension} score ({score.score:.2f})")
        
        # High uncertainty
        low_confidence_dimensions = [
            dim for dim, score in dimension_scores.items() 
            if score.confidence < 0.6
        ]
        if len(low_confidence_dimensions) > len(dimension_scores) * 0.3:
            risk_factors.append("High uncertainty in multiple dimensions")
        
        # Declining trends
        for dimension in dimension_scores.keys():
            trend = self.track_trust_trend(entity_id, dimension)
            if trend.get("trend") == "declining" and trend.get("confidence", 0) > 0.7:
                risk_factors.append(f"Declining trend in {dimension}")
        
        return risk_factors
    
    def _generate_mitigation_suggestions(self, 
                                       dimension_scores: Dict[TrustDimension, TrustScore],
                                       risk_factors: List[str]) -> List[str]:
        """Generate suggestions to mitigate identified risks"""
        
        suggestions = []
        
        # Address low scores
        low_scoring_dimensions = [
            dim for dim, score in dimension_scores.items() 
            if score.score < 0.7
        ]
        
        for dimension in low_scoring_dimensions:
            suggestions.extend(self._get_improvement_actions(dimension, dimension_scores[dimension].score))
        
        # Address uncertainty
        if "uncertainty" in " ".join(risk_factors).lower():
            suggestions.append("Increase evidence collection and validation frequency")
            suggestions.append("Implement more automated monitoring and testing")
        
        # Address declining trends
        if "declining" in " ".join(risk_factors).lower():
            suggestions.append("Investigate root causes of performance degradation")
            suggestions.append("Implement corrective measures and monitor closely")
        
        return list(set(suggestions))  # Remove duplicates
    
    def _get_improvement_actions(self, dimension: TrustDimension, current_score: float) -> List[str]:
        """Get specific improvement actions for a dimension"""
        
        improvement_actions = {
            TrustDimension.ACCURACY: [
                "Implement fact-checking mechanisms",
                "Add source verification",
                "Increase training data quality",
                "Regular accuracy testing"
            ],
            TrustDimension.RELIABILITY: [
                "Improve system monitoring",
                "Add redundancy and failover",
                "Regular performance testing",
                "Implement circuit breakers"
            ],
            TrustDimension.SAFETY: [
                "Enhance bias detection",
                "Implement safety filters",
                "Regular safety audits",
                "User safety feedback collection"
            ],
            TrustDimension.COMPLIANCE: [
                "Regular compliance audits",
                "Update policies and procedures",
                "Staff compliance training",
                "Automated compliance monitoring"
            ],
            TrustDimension.TRANSPARENCY: [
                "Improve decision explanations",
                "Add reasoning traces",
                "Enhance documentation",
                "User education programs"
            ]
        }
        
        actions = improvement_actions.get(dimension, ["Collect more evidence", "Regular monitoring"])
        
        # Prioritize based on current score
        if current_score < 0.5:
            return actions  # All actions needed
        elif current_score < 0.7:
            return actions[:2]  # Top 2 actions
        else:
            return actions[:1]  # Top action only
    
    def _estimate_improvement_timeline(self, dimension: TrustDimension, improvement_gap: float) -> str:
        """Estimate timeline for improvement"""
        
        # Base timeline on improvement gap and dimension complexity
        complexity_factors = {
            TrustDimension.ACCURACY: 1.2,
            TrustDimension.SAFETY: 1.5,
            TrustDimension.COMPLIANCE: 2.0,
            TrustDimension.RELIABILITY: 1.0,
            TrustDimension.TRANSPARENCY: 0.8
        }
        
        base_weeks = improvement_gap * 20  # 20 weeks for full improvement
        complexity_factor = complexity_factors.get(dimension, 1.0)
        estimated_weeks = base_weeks * complexity_factor
        
        if estimated_weeks < 4:
            return "1-4 weeks"
        elif estimated_weeks < 12:
            return "1-3 months"
        elif estimated_weeks < 24:
            return "3-6 months"
        else:
            return "6+ months"
    
    def _cleanup_old_evidence(self, entity_id: str):
        """Remove evidence outside the retention window"""
        
        cutoff_date = datetime.now() - timedelta(days=365)  # 1 year retention
        
        if entity_id in self.evidence_store:
            self.evidence_store[entity_id] = [
                e for e in self.evidence_store[entity_id]
                if e.timestamp >= cutoff_date
            ]
    
    def _load_dimension_weights(self) -> Dict[TrustDimension, float]:
        """Load dimension weights for composite scoring"""
        
        return {
            TrustDimension.ACCURACY: 1.5,
            TrustDimension.SAFETY: 1.8,
            TrustDimension.RELIABILITY: 1.3,
            TrustDimension.COMPLIANCE: 1.6,
            TrustDimension.COMPLETENESS: 1.0,
            TrustDimension.RELEVANCE: 1.1,
            TrustDimension.TRANSPARENCY: 0.9,
            TrustDimension.ROBUSTNESS: 1.2,
            TrustDimension.CONSISTENCY: 1.1,
            TrustDimension.TIMELINESS: 0.8
        }
    
    def _load_evidence_weights(self) -> Dict[EvidenceType, float]:
        """Load evidence type weights"""
        
        return {
            EvidenceType.USER_FEEDBACK: 1.2,
            EvidenceType.PERFORMANCE_METRIC: 1.0,
            EvidenceType.VALIDATION_RESULT: 1.1,
            EvidenceType.EXPERT_REVIEW: 1.5,
            EvidenceType.PEER_COMPARISON: 0.9,
            EvidenceType.HISTORICAL_DATA: 0.8,
            EvidenceType.COMPLIANCE_AUDIT: 1.8,
            EvidenceType.STRESS_TEST: 1.3
        }
    
    def _load_decay_factors(self) -> Dict[str, float]:
        """Load temporal decay factors"""
        
        return {
            "temporal_decay": 0.1,  # 10% decay per month
            "confidence_decay": 0.05,  # 5% confidence decay per month
            "relevance_decay": 0.15  # 15% relevance decay per month
        }

# Utility functions
def create_trust_evidence(evidence_type: EvidenceType,
                        dimension: TrustDimension,
                        value: float,
                        confidence: float,
                        source: str,
                        **kwargs) -> TrustEvidence:
    """Create a trust evidence object"""
    
    import uuid
    return TrustEvidence(
        evidence_id=str(uuid.uuid4()),
        evidence_type=evidence_type,
        dimension=dimension,
        value=value,
        confidence=confidence,
        source=source,
        **kwargs
    )

def aggregate_trust_scores(scores: List[TrustScore]) -> Dict[str, float]:
    """Aggregate multiple trust scores"""
    
    if not scores:
        return {}
    
    return {
        "average_score": statistics.mean(s.score for s in scores),
        "median_score": statistics.median(s.score for s in scores),
        "min_score": min(s.score for s in scores),
        "max_score": max(s.score for s in scores),
        "average_confidence": statistics.mean(s.confidence for s in scores),
        "total_evidence": sum(s.evidence_count for s in scores)
    }

# Export main classes and functions
__all__ = [
    "TrustDimension",
    "TrustLevel",
    "EvidenceType",
    "TrustEvidence",
    "TrustScore",
    "CompositeTrustScore",
    "TrustCalculator",
    "create_trust_evidence",
    "aggregate_trust_scores"
] List[str] = field(default_factory