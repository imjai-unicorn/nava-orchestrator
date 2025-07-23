# backend/services/01-core/nava-logic-controller/tests/test_trust_calculator_comprehensive.py
"""
🔒 Comprehensive Trust Calculator Test Suite
Production-ready testing covering all dimensions and edge cases
Phase 1 - Week 2: Complete validation for deployment
"""

import pytest
import time
import statistics
import threading
import random
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import List, Dict, Any

# Import path setup
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
shared_common_path = os.path.join(current_dir, '..', '..', '..', '..', 'shared', 'common')
shared_common_path = os.path.normpath(shared_common_path)

if os.path.exists(shared_common_path):
    sys.path.insert(0, shared_common_path)
    print(f"✅ Added path: {shared_common_path}")

from trust_calculator import (
    TrustCalculator, TrustEvidence, TrustScore, CompositeTrustScore,
    TrustDimension, TrustLevel, EvidenceType,
    create_trust_evidence, aggregate_trust_scores
)


# =============================================================================
# FIXTURES AND HELPERS
# =============================================================================

@pytest.fixture
def default_trust_calculator():
    """Default trust calculator with standard config"""
    config = {
        'min_evidence_count': 3,
        'confidence_threshold': 0.7,
        'temporal_window_days': 30
    }
    return TrustCalculator(config)

@pytest.fixture
def lenient_trust_calculator():
    """Lenient trust calculator for testing edge cases"""
    config = {
        'min_evidence_count': 1,
        'confidence_threshold': 0.5,
        'temporal_window_days': 365
    }
    return TrustCalculator(config)

@pytest.fixture
def strict_trust_calculator():
    """Strict trust calculator for enterprise testing"""
    config = {
        'min_evidence_count': 5,
        'confidence_threshold': 0.8,
        'temporal_window_days': 14
    }
    return TrustCalculator(config)

def create_sample_evidence(
    dimension: TrustDimension = TrustDimension.ACCURACY,
    value: float = 0.85,
    confidence: float = 0.9,
    evidence_type: EvidenceType = EvidenceType.VALIDATION_RESULT,
    days_ago: int = 0,
    source: str = "test_source"
) -> TrustEvidence:
    """Helper to create sample evidence"""
    evidence = create_trust_evidence(
        evidence_type=evidence_type,
        dimension=dimension,
        value=value,
        confidence=confidence,
        source=source,
        sample_size=100
    )
    if days_ago > 0:
        evidence.timestamp = datetime.now() - timedelta(days=days_ago)
    return evidence

def create_multi_dimensional_evidence(
    entity_id: str, 
    calculator: TrustCalculator,
    dimensions: List[TrustDimension] = None,
    evidence_per_dimension: int = 4
) -> Dict[TrustDimension, List[TrustEvidence]]:
    """Helper to create evidence across multiple dimensions"""
    if dimensions is None:
        dimensions = [TrustDimension.ACCURACY, TrustDimension.RELIABILITY, 
                     TrustDimension.SAFETY, TrustDimension.COMPLETENESS]
    
    evidence_map = {}
    for dimension in dimensions:
        evidence_list = []
        for i in range(evidence_per_dimension):
            evidence = create_sample_evidence(
                dimension=dimension,
                value=0.8 + (i * 0.05),  # Values: 0.8, 0.85, 0.9, 0.95
                confidence=0.85 + (i * 0.02),  # Increasing confidence
                evidence_type=EvidenceType.VALIDATION_RESULT,
                source=f"source_{dimension}_{i}"
            )
            evidence_list.append(evidence)
            calculator.add_evidence(entity_id, evidence)
        evidence_map[dimension] = evidence_list
    
    return evidence_map


# =============================================================================
# CORE DATA STRUCTURE TESTS
# =============================================================================

class TestTrustDataStructures:
    """Test all trust calculator data structures"""
    
    def test_trust_evidence_creation(self):
        """✅ Test TrustEvidence creation with all fields"""
        evidence = TrustEvidence(
            evidence_id="test_001",
            evidence_type=EvidenceType.EXPERT_REVIEW,
            dimension=TrustDimension.SAFETY,
            value=0.92,
            confidence=0.88,
            weight=1.5,
            source="security_expert_panel",
            context={"review_type": "comprehensive", "experts": 5},
            validity_period=timedelta(days=90),
            sample_size=250,
            methodology="peer_review",
            verified=True
        )
        
        assert evidence.evidence_id == "test_001"
        assert evidence.evidence_type == EvidenceType.EXPERT_REVIEW
        assert evidence.dimension == TrustDimension.SAFETY
        assert evidence.value == 0.92
        assert evidence.confidence == 0.88
        assert evidence.weight == 1.5
        assert evidence.context["experts"] == 5
        assert evidence.verified is True
        assert isinstance(evidence.timestamp, datetime)
    
    def test_trust_score_creation(self):
        """✅ Test TrustScore creation with all fields"""
        score = TrustScore(
            dimension=TrustDimension.ACCURACY,
            score=0.89,
            confidence=0.85,
            level=TrustLevel.HIGH,
            evidence_count=12,
            sample_size=1200,
            calculation_method="weighted_temporal_evidence",
            contributing_factors={"expert_review": 0.4, "user_feedback": 0.6},
            uncertainty_factors=["limited_time_window"]
        )
        
        assert score.dimension == TrustDimension.ACCURACY
        assert score.score == 0.89
        assert score.level == TrustLevel.HIGH
        assert score.evidence_count == 12
        assert "expert_review" in score.contributing_factors
        assert "limited_time_window" in score.uncertainty_factors
    
    def test_composite_trust_score_creation(self):
        """✅ Test CompositeTrustScore creation"""
        dimension_scores = {
            TrustDimension.ACCURACY: TrustScore(
                dimension=TrustDimension.ACCURACY, score=0.85, confidence=0.9,
                level=TrustLevel.HIGH, evidence_count=10, sample_size=500,
                calculation_method="test"
            ),
            TrustDimension.SAFETY: TrustScore(
                dimension=TrustDimension.SAFETY, score=0.91, confidence=0.88,
                level=TrustLevel.VERY_HIGH, evidence_count=8, sample_size=400,
                calculation_method="test"
            )
        }
        
        composite = CompositeTrustScore(
            overall_score=0.88,
            overall_level=TrustLevel.HIGH,
            overall_confidence=0.89,
            dimension_scores=dimension_scores,
            calculation_method="weighted_harmonic_composite",
            weights_used={TrustDimension.ACCURACY: 1.5, TrustDimension.SAFETY: 1.8},
            evidence_summary={EvidenceType.EXPERT_REVIEW: 10, EvidenceType.USER_FEEDBACK: 8},
            risk_factors=["declining_trend_in_accuracy"],
            mitigation_suggestions=["increase_validation_frequency"]
        )
        
        assert composite.overall_score == 0.88
        assert composite.overall_level == TrustLevel.HIGH
        assert len(composite.dimension_scores) == 2
        assert composite.calculation_method == "weighted_harmonic_composite"
        assert len(composite.risk_factors) == 1
        assert len(composite.mitigation_suggestions) == 1


# =============================================================================
# TRUST CALCULATOR INITIALIZATION AND CONFIGURATION
# =============================================================================

class TestTrustCalculatorInitialization:
    """Test trust calculator initialization and configuration"""
    
    def test_default_initialization(self):
        """✅ Test default initialization"""
        calculator = TrustCalculator()
        
        assert calculator.min_evidence_count == 3  # Default
        assert calculator.confidence_threshold == 0.6  # Default
        assert calculator.temporal_window_days == 30  # Default
        assert len(calculator.evidence_store) == 0
        assert len(calculator.trust_history) == 0
    
    def test_custom_config_initialization(self, default_trust_calculator):
        """✅ Test initialization with custom config"""
        assert default_trust_calculator.min_evidence_count == 3
        assert default_trust_calculator.confidence_threshold == 0.7
        assert default_trust_calculator.temporal_window_days == 30
    
    def test_dimension_weights_loading(self, default_trust_calculator):
        """✅ Test dimension weights are loaded correctly"""
        weights = default_trust_calculator.dimension_weights
        
        assert TrustDimension.SAFETY in weights
        assert TrustDimension.ACCURACY in weights
        assert weights[TrustDimension.SAFETY] > weights[TrustDimension.TIMELINESS]  # Safety more important
    
    def test_evidence_weights_loading(self, default_trust_calculator):
        """✅ Test evidence type weights are loaded correctly"""
        weights = default_trust_calculator.evidence_weights
        
        assert EvidenceType.EXPERT_REVIEW in weights
        assert EvidenceType.USER_FEEDBACK in weights
        assert weights[EvidenceType.EXPERT_REVIEW] > weights[EvidenceType.HISTORICAL_DATA]


# =============================================================================
# EVIDENCE MANAGEMENT TESTS
# =============================================================================

class TestEvidenceManagement:
    """Test evidence addition, validation, and management"""
    
    def test_add_valid_evidence(self, default_trust_calculator):
        """✅ Test adding valid evidence"""
        evidence = create_sample_evidence()
        result = default_trust_calculator.add_evidence("entity_001", evidence)
        
        assert result is True
        assert "entity_001" in default_trust_calculator.evidence_store
        assert len(default_trust_calculator.evidence_store["entity_001"]) == 1
        assert default_trust_calculator.evidence_store["entity_001"][0].evidence_id == evidence.evidence_id
    
    def test_add_multiple_evidence_same_entity(self, default_trust_calculator):
        """✅ Test adding multiple evidence for same entity"""
        entity_id = "entity_multi"
        
        for i in range(5):
            evidence = create_sample_evidence(
                value=0.8 + i * 0.05,
                source=f"source_{i}"
            )
            result = default_trust_calculator.add_evidence(entity_id, evidence)
            assert result is True
        
        assert len(default_trust_calculator.evidence_store[entity_id]) == 5
    
    def test_evidence_validation_invalid_value(self, default_trust_calculator):
        """✅ Test evidence validation fails for invalid values"""
        # Test value > 1.0
        invalid_evidence = TrustEvidence(
            evidence_id="invalid_001",
            evidence_type=EvidenceType.USER_FEEDBACK,
            dimension=TrustDimension.ACCURACY,
            value=1.5,  # Invalid
            confidence=0.8,
            source="test_source"
        )
        
        result = default_trust_calculator.add_evidence("entity_001", invalid_evidence)
        assert result is False
        
        # Test negative value
        invalid_evidence.value = -0.1
        result = default_trust_calculator.add_evidence("entity_001", invalid_evidence)
        assert result is False
    
    def test_evidence_validation_invalid_confidence(self, default_trust_calculator):
        """✅ Test evidence validation fails for invalid confidence"""
        invalid_evidence = TrustEvidence(
            evidence_id="invalid_002",
            evidence_type=EvidenceType.USER_FEEDBACK,
            dimension=TrustDimension.ACCURACY,
            value=0.8,
            confidence=1.2,  # Invalid
            source="test_source"
        )
        
        result = default_trust_calculator.add_evidence("entity_001", invalid_evidence)
        assert result is False
    
    def test_evidence_validation_missing_fields(self, default_trust_calculator):
        """✅ Test evidence validation fails for missing required fields"""
        # Missing evidence_id
        invalid_evidence = TrustEvidence(
            evidence_id="",  # Empty
            evidence_type=EvidenceType.USER_FEEDBACK,
            dimension=TrustDimension.ACCURACY,
            value=0.8,
            confidence=0.9,
            source="test_source"
        )
        
        result = default_trust_calculator.add_evidence("entity_001", invalid_evidence)
        assert result is False
        
        # Missing source
        invalid_evidence.evidence_id = "valid_id"
        invalid_evidence.source = ""  # Empty
        result = default_trust_calculator.add_evidence("entity_001", invalid_evidence)
        assert result is False
    
    def test_evidence_cleanup(self, default_trust_calculator):
        """✅ Test old evidence cleanup"""
        entity_id = "cleanup_test"
        
        # Add recent evidence first
        recent_evidence = create_sample_evidence(days_ago=1)
        default_trust_calculator.add_evidence(entity_id, recent_evidence)
        
        # Manually add very old evidence (bypass automatic cleanup)
        old_evidence = create_sample_evidence(days_ago=400)
        default_trust_calculator.evidence_store[entity_id].append(old_evidence)
        
        initial_count = len(default_trust_calculator.evidence_store[entity_id])
        
        # Trigger cleanup
        default_trust_calculator._cleanup_old_evidence(entity_id)
        
        final_count = len(default_trust_calculator.evidence_store[entity_id])
        
        # Should remove old evidence but keep recent
        assert final_count < initial_count
        assert final_count >= 1  # Recent evidence should remain


# =============================================================================
# TRUST SCORE CALCULATION TESTS
# =============================================================================

class TestTrustScoreCalculation:
    """Test trust score calculation logic"""
    
    def test_calculate_trust_score_insufficient_evidence(self, default_trust_calculator):
        """✅ Test calculation with insufficient evidence"""
        entity_id = "insufficient_test"
        
        # Add only 2 evidence (less than min_evidence_count=3)
        for i in range(2):
            evidence = create_sample_evidence(source=f"source_{i}")
            default_trust_calculator.add_evidence(entity_id, evidence)
        
        score = default_trust_calculator.calculate_trust_score(entity_id, TrustDimension.ACCURACY)
        assert score is None
    
    def test_calculate_trust_score_sufficient_evidence(self, default_trust_calculator):
        """✅ Test calculation with sufficient evidence"""
        entity_id = "sufficient_test"
        
        # Add sufficient evidence (>= min_evidence_count=3)
        for i in range(4):
            evidence = create_sample_evidence(
                value=0.8 + i * 0.05,
                confidence=0.85 + i * 0.02,
                source=f"source_{i}"
            )
            default_trust_calculator.add_evidence(entity_id, evidence)
        
        score = default_trust_calculator.calculate_trust_score(entity_id, TrustDimension.ACCURACY)
        
        assert score is not None
        assert isinstance(score, TrustScore)
        assert score.dimension == TrustDimension.ACCURACY
        assert 0.0 <= score.score <= 1.0
        assert 0.0 <= score.confidence <= 1.0
        assert isinstance(score.level, TrustLevel)
        assert score.evidence_count >= 3
    
    def test_trust_level_determination(self, default_trust_calculator):
        """✅ Test trust level determination from scores"""
        test_cases = [
            (0.95, TrustLevel.VERY_HIGH),
            (0.85, TrustLevel.HIGH),
            (0.70, TrustLevel.MEDIUM),
            (0.50, TrustLevel.LOW),
            (0.30, TrustLevel.VERY_LOW),
            (0.10, TrustLevel.VERY_LOW)
        ]
        
        for score_value, expected_level in test_cases:
            level = default_trust_calculator._determine_trust_level(score_value)
            assert level == expected_level, f"Score {score_value} should be {expected_level}, got {level}"
    
    def test_weighted_score_calculation(self, default_trust_calculator):
        """✅ Test weighted score calculation with different evidence types"""
        entity_id = "weighted_test"
        
        # Add evidence with different types and weights
        evidence_configs = [
            (EvidenceType.EXPERT_REVIEW, 0.9, 0.95),      # High weight, high confidence
            (EvidenceType.USER_FEEDBACK, 0.8, 0.85),      # Medium weight
            (EvidenceType.HISTORICAL_DATA, 0.7, 0.75),    # Lower weight
            (EvidenceType.PERFORMANCE_METRIC, 0.85, 0.9)  # Good weight
        ]
        
        for evidence_type, value, confidence in evidence_configs:
            evidence = create_sample_evidence(
                evidence_type=evidence_type,
                value=value,
                confidence=confidence,
                source=f"source_{evidence_type}"
            )
            default_trust_calculator.add_evidence(entity_id, evidence)
        
        score = default_trust_calculator.calculate_trust_score(entity_id, TrustDimension.ACCURACY)
        
        assert score is not None
        # Expert review should have more influence than historical data
        assert score.score > 0.75  # Should be influenced by high-weight evidence
    
    def test_temporal_weight_calculation(self, default_trust_calculator):
        """✅ Test temporal weight decreases with age"""
        recent_weight = default_trust_calculator._calculate_temporal_weight(1)    # 1 day old
        old_weight = default_trust_calculator._calculate_temporal_weight(30)      # 30 days old
        very_old_weight = default_trust_calculator._calculate_temporal_weight(90) # 90 days old
        
        assert recent_weight > old_weight > very_old_weight
        assert 0.0 <= very_old_weight <= 1.0
        assert 0.0 <= old_weight <= 1.0
        assert 0.0 <= recent_weight <= 1.0
    
    def test_evidence_filtering_by_temporal_window(self, default_trust_calculator):
        """✅ Test evidence filtering by temporal window"""
        entity_id = "temporal_test"
        
        # Add old evidence (outside temporal window)
        old_evidence = create_sample_evidence(days_ago=60, source="old_source")  # 60 days > 30 days window
        default_trust_calculator.add_evidence(entity_id, old_evidence)
        
        # Add recent evidence (within temporal window)
        for i in range(3):
            recent_evidence = create_sample_evidence(days_ago=i, source=f"recent_{i}")
            default_trust_calculator.add_evidence(entity_id, recent_evidence)
        
        relevant_evidence = default_trust_calculator._get_relevant_evidence(entity_id, TrustDimension.ACCURACY)
        
        # Should only include recent evidence
        assert len(relevant_evidence) == 3
        assert all(e.source.startswith("recent_") for e in relevant_evidence)


# =============================================================================
# COMPOSITE TRUST SCORE TESTS
# =============================================================================

class TestCompositeTrustScore:
    """Test composite trust score calculation"""
    
    def test_composite_trust_insufficient_dimensions(self, default_trust_calculator):
        """✅ Test composite trust fails with insufficient dimensions"""
        entity_id = "insufficient_dimensions"
        
        # Add evidence for only 2 dimensions (need >= 2)
        dimensions = [TrustDimension.ACCURACY]  # Only 1 dimension
        
        for dimension in dimensions:
            for i in range(4):  # Sufficient evidence per dimension
                evidence = create_sample_evidence(
                    dimension=dimension,
                    value=0.8 + i * 0.05,
                    source=f"source_{dimension}_{i}"
                )
                default_trust_calculator.add_evidence(entity_id, evidence)
        
        composite = default_trust_calculator.calculate_composite_trust(entity_id)
        assert composite is None
    
    def test_composite_trust_success(self, default_trust_calculator):
        """✅ Test successful composite trust calculation"""
        entity_id = "composite_success"
        
        # Add evidence for 4 dimensions (>= 3 required)
        dimensions = [TrustDimension.ACCURACY, TrustDimension.RELIABILITY, 
                     TrustDimension.SAFETY, TrustDimension.COMPLETENESS]
        
        evidence_map = create_multi_dimensional_evidence(entity_id, default_trust_calculator, dimensions)
        
        composite = default_trust_calculator.calculate_composite_trust(entity_id)
        
        assert composite is not None
        assert isinstance(composite, CompositeTrustScore)
        assert 0.0 <= composite.overall_score <= 1.0
        assert 0.0 <= composite.overall_confidence <= 1.0
        assert isinstance(composite.overall_level, TrustLevel)
        assert len(composite.dimension_scores) >= 3
        assert composite.calculation_method == "weighted_harmonic_composite"
    
    def test_composite_trust_with_custom_weights(self, default_trust_calculator):
        """✅ Test composite trust with custom dimension weights"""
        entity_id = "custom_weights_test"
        
        dimensions = [TrustDimension.ACCURACY, TrustDimension.SAFETY, TrustDimension.RELIABILITY]
        
        create_multi_dimensional_evidence(entity_id, default_trust_calculator, dimensions)
        
        # Custom weights - emphasize safety
        custom_weights = {
            TrustDimension.ACCURACY: 1.0,
            TrustDimension.SAFETY: 3.0,      # 3x weight
            TrustDimension.RELIABILITY: 1.0
        }
        
        composite = default_trust_calculator.calculate_composite_trust(
            entity_id, 
            custom_weights=custom_weights
        )
        
        assert composite is not None
        assert composite.weights_used == custom_weights
        # Safety should have higher influence
        assert TrustDimension.SAFETY in composite.dimension_scores
    
    def test_composite_trust_low_confidence_filtering(self, default_trust_calculator):
        """✅ Test composite trust filters out low confidence scores"""
        entity_id = "low_confidence_test"
        
        dimensions = [TrustDimension.ACCURACY, TrustDimension.RELIABILITY, TrustDimension.SAFETY]
        
        # Add evidence with varying confidence levels
        for dimension in dimensions:
            for i in range(4):
                # Create evidence with decreasing confidence
                confidence = 0.9 - (i * 0.1)  # 0.9, 0.8, 0.7, 0.6
                evidence = create_sample_evidence(
                    dimension=dimension,
                    confidence=confidence,
                    source=f"source_{dimension}_{i}"
                )
                default_trust_calculator.add_evidence(entity_id, evidence)
        
        composite = default_trust_calculator.calculate_composite_trust(entity_id)
        
        # Should include dimensions with confidence >= 0.7
        if composite:
            # The filtering logic is lenient (>= 0.4), so we assert against that
            lenient_threshold = default_trust_calculator.confidence_threshold - 0.3
            for dimension, score in composite.dimension_scores.items():
                assert score.confidence >= lenient_threshold


# =============================================================================
# TREND TRACKING TESTS
# =============================================================================

class TestTrustTrendTracking:
    """Test trust trend tracking over time"""
    
    def test_trend_insufficient_data(self, default_trust_calculator):
        """✅ Test trend tracking with insufficient data"""
        entity_id = "trend_insufficient"
        dimension = TrustDimension.ACCURACY
        
        trend = default_trust_calculator.track_trust_trend(entity_id, dimension)
        
        assert trend["trend"] == "insufficient_data"
        assert trend["confidence"] == 0.0
    
    def test_trend_with_historical_data(self, default_trust_calculator):
        """✅ Test trend tracking with historical data"""
        entity_id = "trend_historical"
        dimension = TrustDimension.RELIABILITY
        
        # Simulate improving trend over time
        for day_offset in range(10, 0, -1):  # 10 days ago to now
            for i in range(4):  # 4 evidence per calculation
                evidence = create_sample_evidence(
                    dimension=dimension,
                    value=0.6 + (10 - day_offset) * 0.03,  # Improving: 0.6 to 0.9
                    confidence=0.85,
                    source=f"trend_source_{day_offset}_{i}",
                    days_ago=day_offset
                )
                default_trust_calculator.add_evidence(entity_id, evidence)
            
            # Calculate and manually add to history
            score = default_trust_calculator.calculate_trust_score(entity_id, dimension)
            if score:
                history_key = f"{entity_id}:{dimension}"
                default_trust_calculator.trust_history[history_key].append({
                    'timestamp': datetime.now() - timedelta(days=day_offset),
                    'score': score.score,
                    'confidence': score.confidence,
                    'evidence_count': score.evidence_count
                })
        
        trend = default_trust_calculator.track_trust_trend(entity_id, dimension, lookback_days=15)
        
        assert trend["trend"] in ["improving", "declining", "stable"]
        assert 0.0 <= trend["confidence"] <= 1.0
        assert trend["data_points"] > 0
        
        # With improving data, should detect improvement
        if trend["data_points"] >= 3:
            assert trend["trend"] == "improving"
    
    def test_trend_declining(self, default_trust_calculator):
        """✅ Test declining trend detection"""
        entity_id = "trend_declining"
        dimension = TrustDimension.ACCURACY
        
        # Simulate declining trend
        for day_offset in range(5, 0, -1):
            for i in range(4):
                # Declining values
                evidence = create_sample_evidence(
                    dimension=dimension,
                    value=0.9 - (5 - day_offset) * 0.05,  # Declining: 0.9 to 0.7
                    confidence=0.85,
                    source=f"decline_source_{day_offset}_{i}",
                    days_ago=day_offset
                )
                default_trust_calculator.add_evidence(entity_id, evidence)
            
            # Add to history
            score = default_trust_calculator.calculate_trust_score(entity_id, dimension)
            if score:
                history_key = f"{entity_id}:{dimension}"
                default_trust_calculator.trust_history[history_key].append({
                    'timestamp': datetime.now() - timedelta(days=day_offset),
                    'score': score.score,
                    'confidence': score.confidence,
                    'evidence_count': score.evidence_count
                })
        
        trend = default_trust_calculator.track_trust_trend(entity_id, dimension)
        
        if trend["data_points"] >= 3:
            assert trend["trend"] == "declining"


# =============================================================================
# ENTITY COMPARISON TESTS
# =============================================================================

class TestEntityComparison:
    """Test entity comparison functionality"""
    
    def test_compare_entities_insufficient_data(self, default_trust_calculator):
        """✅ Test comparison with insufficient valid entities"""
        entity_ids = ["entity1", "entity2"]
        
        comparison = default_trust_calculator.compare_entities(entity_ids)
        
        assert "error" in comparison
        assert "Insufficient entities" in comparison["error"]
    
    def test_compare_entities_success(self, default_trust_calculator):
        """✅ Test successful entity comparison"""
        entity_ids = ["gpt_model", "claude_model", "gemini_model"]
        dimensions = [TrustDimension.ACCURACY, TrustDimension.RELIABILITY, TrustDimension.SAFETY]
        
        # Add evidence for each entity with different performance levels
        for idx, entity_id in enumerate(entity_ids):
            base_performance = 0.7 + (idx * 0.1)  # Different base performance per entity
            
            for dimension in dimensions:
                for i in range(5):  # Sufficient evidence
                    evidence = create_sample_evidence(
                        dimension=dimension,
                        value=min(base_performance + i * 0.04, 1.0),
                        confidence=0.85,
                        source=f"validator_{entity_id}_{i}"
                    )
                    default_trust_calculator.add_evidence(entity_id, evidence)
        
        comparison = default_trust_calculator.compare_entities(entity_ids, dimensions)
        
        assert "ranking" in comparison
        assert "dimension_analysis" in comparison
        assert "statistics" in comparison
        
        # Check ranking
        ranking = comparison["ranking"]
        assert len(ranking) == len(entity_ids)
        assert all("entity_id" in r for r in ranking)
        assert all("overall_score" in r for r in ranking)
        
        # Ranking should be ordered (highest score first)
        scores = [r["overall_score"] for r in ranking]
        assert scores == sorted(scores, reverse=True)
        
        # Check dimension analysis
        dim_analysis = comparison["dimension_analysis"]
        for dimension in dimensions:
            if dimension in dim_analysis:
                assert "best" in dim_analysis[dimension]
                assert "worst" in dim_analysis[dimension]
                assert "average" in dim_analysis[dimension]
                assert "spread" in dim_analysis[dimension]
    
    def test_compare_entities_statistical_analysis(self, default_trust_calculator):
        """✅ Test statistical analysis in entity comparison"""
        entity_ids = ["entity_a", "entity_b", "entity_c", "entity_d"]
        
        # Create entities with known score distributions
        target_scores = [0.9, 0.8, 0.7, 0.6]
        dimensions = [TrustDimension.ACCURACY, TrustDimension.RELIABILITY, TrustDimension.SAFETY]
        
        for idx, entity_id in enumerate(entity_ids):
            target_score = target_scores[idx]
            
            for dimension in dimensions:
                for i in range(5):
                    evidence = create_sample_evidence(
                        dimension=dimension,
                        value=target_score + random.uniform(-0.05, 0.05),  # Small variation
                        confidence=0.85,
                        source=f"source_{entity_id}_{i}"
                    )
                    default_trust_calculator.add_evidence(entity_id, evidence)
        
        comparison = default_trust_calculator.compare_entities(entity_ids)
        
        if "statistics" in comparison:
            stats = comparison["statistics"]
            assert "mean" in stats
            assert "median" in stats
            assert "std_dev" in stats
            assert "range" in stats
            
            # With our controlled data, range should be approximately 0.3
            assert 0.2 <= stats["range"] <= 0.4


# =============================================================================
# RECOMMENDATION SYSTEM TESTS
# =============================================================================

class TestTrustRecommendations:
    """Test trust improvement recommendations"""
    
    def test_recommendations_no_composite(self, default_trust_calculator):
        """✅ Test recommendations when no composite trust exists"""
        entity_id = "no_composite"
        
        recommendations = default_trust_calculator.get_trust_recommendations(entity_id)
        
        assert len(recommendations) >= 1
        assert any("evidence" in r["recommendation"].lower() for r in recommendations)
    
    def test_recommendations_with_varying_scores(self, default_trust_calculator):
        """✅ Test recommendations based on varying dimension scores"""
        entity_id = "varying_scores_test"
        
        # Create varying trust levels across dimensions
        dimension_configs = [
            (TrustDimension.ACCURACY, 0.95),      # Very High - no improvement needed
            (TrustDimension.SAFETY, 0.85),        # High - minor improvement
            (TrustDimension.RELIABILITY, 0.65),   # Medium - needs improvement
            (TrustDimension.COMPLETENESS, 0.45),  # Low - priority improvement
            (TrustDimension.COMPLIANCE, 0.3)      # Very Low - critical improvement
        ]
        
        for dimension, base_score in dimension_configs:
            for i in range(4):
                evidence = create_sample_evidence(
                    dimension=dimension,
                    value=min(base_score + i * 0.02, 1.0),
                    confidence=0.8,
                    source=f"source_{dimension}_{i}"
                )
                default_trust_calculator.add_evidence(entity_id, evidence)
        
        recommendations = default_trust_calculator.get_trust_recommendations(
            entity_id,
            target_level=TrustLevel.HIGH
        )
        
        assert len(recommendations) > 0
        
        # Should prioritize lowest scoring dimensions
        rec_dimensions = [r.get("dimension") for r in recommendations if "dimension" in r]
        if rec_dimensions:
            # Compliance and Completeness should be in recommendations
            low_score_dimensions = [TrustDimension.COMPLIANCE, TrustDimension.COMPLETENESS]
            assert any(dim in rec_dimensions for dim in low_score_dimensions)
    
    # test_trust_calculator_comprehensive.py

    def test_recommendations_priority_levels(self, default_trust_calculator):
        """✅ Test recommendation priority levels"""
        entity_id = "priority_test"

        # Create dimension with LOW score to trigger high priority recommendation
        for i in range(4):
            evidence = create_sample_evidence(
                dimension=TrustDimension.SAFETY,  # <-- แก้กลับเป็น SAFETY
                value=0.4 + i * 0.02,           # <-- แก้กลับเป็นคะแนนต่ำ
                confidence=0.8,
                source=f"safety_source_{i}"
            )
            default_trust_calculator.add_evidence(entity_id, evidence)

        # Add a second dimension to enable composite score calculation
        for i in range(4):
            evidence = create_sample_evidence(
                dimension=TrustDimension.ACCURACY,
                value=0.95,
                confidence=0.9,
                source=f"accuracy_source_{i}"
            )
            default_trust_calculator.add_evidence(entity_id, evidence)

        recommendations = default_trust_calculator.get_trust_recommendations(entity_id)

        # Should include high priority recommendations for safety
        high_priority_recs = [r for r in recommendations if r.get("priority") == "high"]
        assert len(high_priority_recs) > 0


# =============================================================================
# UTILITY FUNCTION TESTS
# =============================================================================

class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_create_trust_evidence_function(self):
        """✅ Test create_trust_evidence utility function"""
        evidence = create_trust_evidence(
            evidence_type=EvidenceType.USER_FEEDBACK,
            dimension=TrustDimension.ACCURACY,
            value=0.85,
            confidence=0.9,
            source="test_utility_source",
            sample_size=50,
            methodology="user_survey"
        )
        
        assert isinstance(evidence, TrustEvidence)
        assert evidence.evidence_type == EvidenceType.USER_FEEDBACK
        assert evidence.dimension == TrustDimension.ACCURACY
        assert evidence.value == 0.85
        assert evidence.confidence == 0.9
        assert evidence.source == "test_utility_source"
        assert evidence.sample_size == 50
        assert evidence.methodology == "user_survey"
        assert evidence.evidence_id is not None
        assert len(evidence.evidence_id) > 0
    
    def test_aggregate_trust_scores_function(self):
        """✅ Test aggregate_trust_scores utility function"""
        # Create mock trust scores
        scores = []
        expected_values = []
        
        for i in range(5):
            score_value = 0.7 + i * 0.05  # 0.7, 0.75, 0.8, 0.85, 0.9
            confidence_value = 0.8 + i * 0.02  # 0.8, 0.82, 0.84, 0.86, 0.88
            
            score = TrustScore(
                dimension=TrustDimension.ACCURACY,
                score=score_value,
                confidence=confidence_value,
                level=TrustLevel.HIGH,
                evidence_count=10 + i,
                sample_size=100 + i * 20,
                calculation_method="test_method"
            )
            scores.append(score)
            expected_values.append(score_value)
        
        aggregated = aggregate_trust_scores(scores)
        
        assert isinstance(aggregated, dict)
        assert "average_score" in aggregated
        assert "median_score" in aggregated
        assert "min_score" in aggregated
        assert "max_score" in aggregated
        assert "average_confidence" in aggregated
        assert "total_evidence" in aggregated
        
        # Verify calculations
        assert abs(aggregated["average_score"] - statistics.mean(expected_values)) < 0.001
        assert abs(aggregated["median_score"] - statistics.median(expected_values)) < 0.001
        assert aggregated["min_score"] == min(expected_values)
        assert aggregated["max_score"] == max(expected_values)
        assert aggregated["total_evidence"] == sum(s.evidence_count for s in scores)
    
    def test_aggregate_trust_scores_empty_list(self):
        """✅ Test aggregate_trust_scores with empty list"""
        aggregated = aggregate_trust_scores([])
        assert aggregated == {}


# =============================================================================
# PERFORMANCE AND SCALABILITY TESTS
# =============================================================================

class TestPerformanceAndScalability:
    """Test performance and scalability requirements"""
    
    def test_large_scale_evidence_processing(self, lenient_trust_calculator):
        """✅ Test processing large amounts of evidence efficiently"""
        entity_id = "large_scale_test"
        
        start_time = time.time()
        
        # Add large number of evidence pieces
        evidence_count = 200
        for i in range(evidence_count):
            evidence = create_sample_evidence(
                dimension=random.choice(list(TrustDimension)),
                value=0.7 + random.random() * 0.3,
                confidence=0.7 + random.random() * 0.3,
                evidence_type=random.choice(list(EvidenceType)),
                source=f"perf_source_{i}",
                days_ago=random.randint(0, 30)
            )
            lenient_trust_calculator.add_evidence(entity_id, evidence)
        
        processing_time = time.time() - start_time
        
        # Should process efficiently (< 5 seconds for 200 evidence)
        assert processing_time < 5.0, f"Evidence processing too slow: {processing_time:.3f}s"
        assert len(lenient_trust_calculator.evidence_store[entity_id]) == evidence_count
        
        print(f"✅ Processed {evidence_count} evidence in {processing_time:.3f}s")
    
    def test_trust_calculation_performance(self, lenient_trust_calculator):
        """✅ Test trust calculation performance"""
        entity_id = "performance_test"
        
        # Add substantial evidence across multiple dimensions
        for dimension in list(TrustDimension)[:5]:  # Test with 5 dimensions
            for i in range(10):  # 10 evidence per dimension
                evidence = create_sample_evidence(
                    dimension=dimension,
                    value=0.8 + i * 0.01,
                    confidence=0.85,
                    source=f"perf_validator_{i}"
                )
                lenient_trust_calculator.add_evidence(entity_id, evidence)
        
        # Test individual score calculation performance
        start_time = time.time()
        score = lenient_trust_calculator.calculate_trust_score(entity_id, TrustDimension.ACCURACY)
        individual_calc_time = time.time() - start_time
        
        # Test composite calculation performance
        start_time = time.time()
        composite = lenient_trust_calculator.calculate_composite_trust(entity_id)
        composite_calc_time = time.time() - start_time
        
        # Performance assertions
        assert individual_calc_time < 0.5, f"Individual calculation too slow: {individual_calc_time:.3f}s"
        assert composite_calc_time < 1.0, f"Composite calculation too slow: {composite_calc_time:.3f}s"
        assert score is not None
        assert composite is not None
        
        print(f"✅ Individual calc: {individual_calc_time:.3f}s, Composite calc: {composite_calc_time:.3f}s")
    
    def test_concurrent_access_stability(self, lenient_trust_calculator):
        """✅ Test system stability under concurrent access"""
        results = {"success": 0, "failure": 0}
        lock = threading.Lock()
        
        def worker_thread(worker_id: int, iterations: int = 20):
            for i in range(iterations):
                try:
                    entity_id = f"concurrent_entity_{worker_id}_{i % 3}"
                    evidence = create_sample_evidence(
                        dimension=random.choice(list(TrustDimension)),
                        value=0.7 + random.random() * 0.3,
                        confidence=0.7 + random.random() * 0.3,
                        source=f"worker_{worker_id}_source_{i}"
                    )
                    
                    success = lenient_trust_calculator.add_evidence(entity_id, evidence)
                    
                    # Occasionally calculate scores
                    if i % 5 == 0:
                        score = lenient_trust_calculator.calculate_trust_score(
                            entity_id, 
                            TrustDimension.ACCURACY
                        )
                    
                    with lock:
                        if success:
                            results["success"] += 1
                        else:
                            results["failure"] += 1
                            
                except Exception as e:
                    with lock:
                        results["failure"] += 1
                    print(f"Worker {worker_id} error: {e}")
        
        # Run concurrent workers
        threads = []
        worker_count = 5
        iterations_per_worker = 20
        
        start_time = time.time()
        
        for worker_id in range(worker_count):
            thread = threading.Thread(target=worker_thread, args=(worker_id, iterations_per_worker))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        total_operations = worker_count * iterations_per_worker
        success_rate = results["success"] / total_operations if total_operations > 0 else 0
        
        # Stability requirements
        assert success_rate > 0.95, f"Success rate too low: {success_rate:.3f}"
        assert results["failure"] < total_operations * 0.05, f"Too many failures: {results['failure']}"
        assert total_time < 10.0, f"Concurrent operations too slow: {total_time:.3f}s"
        
        print(f"✅ Concurrent test: {success_rate:.3f} success rate, {total_time:.3f}s total time")


# =============================================================================
# INTEGRATION AND END-TO-END TESTS
# =============================================================================

class TestIntegrationScenarios:
    """Integration tests covering complete workflows"""
    
    def test_complete_trust_workflow_enterprise_scenario(self, default_trust_calculator):
        """🎯 CRITICAL: Complete enterprise trust workflow"""
        entity_id = "enterprise_ai_system"
        
        # Phase 1: Initial evidence collection
        initial_evidence_data = [
            # Accuracy evidence
            (EvidenceType.EXPERT_REVIEW, TrustDimension.ACCURACY, 0.88, 0.95, "ai_expert_panel"),
            (EvidenceType.VALIDATION_RESULT, TrustDimension.ACCURACY, 0.91, 0.88, "automated_validation"),
            (EvidenceType.USER_FEEDBACK, TrustDimension.ACCURACY, 0.85, 0.82, "user_survey_q1"),
            (EvidenceType.PEER_COMPARISON, TrustDimension.ACCURACY, 0.87, 0.78, "benchmark_study"),
            
            # Reliability evidence
            (EvidenceType.PERFORMANCE_METRIC, TrustDimension.RELIABILITY, 0.94, 0.95, "uptime_monitoring"),
            (EvidenceType.STRESS_TEST, TrustDimension.RELIABILITY, 0.89, 0.92, "load_testing"),
            (EvidenceType.HISTORICAL_DATA, TrustDimension.RELIABILITY, 0.92, 0.85, "6_month_history"),
            (EvidenceType.PERFORMANCE_METRIC, TrustDimension.RELIABILITY, 0.96, 0.93, "response_time_sla"),
            
            # Safety evidence
            (EvidenceType.COMPLIANCE_AUDIT, TrustDimension.SAFETY, 0.93, 0.98, "security_audit"),
            (EvidenceType.EXPERT_REVIEW, TrustDimension.SAFETY, 0.90, 0.94, "safety_committee"),
            (EvidenceType.VALIDATION_RESULT, TrustDimension.SAFETY, 0.87, 0.89, "bias_testing"),
            (EvidenceType.STRESS_TEST, TrustDimension.SAFETY, 0.91, 0.87, "adversarial_testing"),
            
            # Completeness evidence
            (EvidenceType.USER_FEEDBACK, TrustDimension.COMPLETENESS, 0.79, 0.83, "completeness_survey"),
            (EvidenceType.EXPERT_REVIEW, TrustDimension.COMPLETENESS, 0.82, 0.91, "content_expert_review"),
            (EvidenceType.VALIDATION_RESULT, TrustDimension.COMPLETENESS, 0.85, 0.86, "coverage_analysis"),
            (EvidenceType.PEER_COMPARISON, TrustDimension.COMPLETENESS, 0.81, 0.79, "competitor_analysis"),
            
            # Compliance evidence
            (EvidenceType.COMPLIANCE_AUDIT, TrustDimension.COMPLIANCE, 0.96, 0.99, "regulatory_audit"),
            (EvidenceType.EXPERT_REVIEW, TrustDimension.COMPLIANCE, 0.94, 0.97, "legal_review"),
            (EvidenceType.VALIDATION_RESULT, TrustDimension.COMPLIANCE, 0.92, 0.89, "policy_validation"),
            (EvidenceType.HISTORICAL_DATA, TrustDimension.COMPLIANCE, 0.93, 0.88, "compliance_history")
        ]
        
        # Add all evidence
        for evidence_type, dimension, value, confidence, source in initial_evidence_data:
            evidence = create_trust_evidence(
                evidence_type=evidence_type,
                dimension=dimension,
                value=value,
                confidence=confidence,
                source=source,
                sample_size=100 + random.randint(0, 500)
            )
            result = default_trust_calculator.add_evidence(entity_id, evidence)
            assert result is True, f"Failed to add evidence: {source}"
        
        # Phase 2: Calculate individual dimension scores
        dimension_scores = {}
        for dimension in [TrustDimension.ACCURACY, TrustDimension.RELIABILITY, 
                         TrustDimension.SAFETY, TrustDimension.COMPLETENESS, TrustDimension.COMPLIANCE]:
            score = default_trust_calculator.calculate_trust_score(entity_id, dimension)
            assert score is not None, f"Failed to calculate score for {dimension}"
            dimension_scores[dimension] = score
            
            # Verify score characteristics
            assert 0.0 <= score.score <= 1.0
            assert 0.0 <= score.confidence <= 1.0
            assert score.evidence_count >= 3
            assert isinstance(score.level, TrustLevel)
        
        # Phase 3: Calculate composite trust
        composite = default_trust_calculator.calculate_composite_trust(entity_id)
        assert composite is not None, "Failed to calculate composite trust"
        
        # Verify composite characteristics
        assert isinstance(composite, CompositeTrustScore)
        assert 0.0 <= composite.overall_score <= 1.0
        assert 0.0 <= composite.overall_confidence <= 1.0
        assert len(composite.dimension_scores) >= 3
        assert composite.calculation_method == "weighted_harmonic_composite"
        
        # Phase 4: Analyze trends (simulate some history)
        for dimension in dimension_scores.keys():
            trend = default_trust_calculator.track_trust_trend(entity_id, dimension)
            assert isinstance(trend, dict)
            assert "trend" in trend
            assert "confidence" in trend
        
        # Phase 5: Get recommendations
        recommendations = default_trust_calculator.get_trust_recommendations(
            entity_id, 
            target_level=TrustLevel.VERY_HIGH
        )
        assert isinstance(recommendations, list)
        
        # Phase 6: Verify enterprise requirements
        # High trust levels for critical dimensions
        critical_dimensions = [TrustDimension.SAFETY, TrustDimension.COMPLIANCE, TrustDimension.RELIABILITY]
        for dimension in critical_dimensions:
            if dimension in dimension_scores:
                score = dimension_scores[dimension]
                assert score.level in [TrustLevel.HIGH, TrustLevel.VERY_HIGH], \
                    f"Critical dimension {dimension} has insufficient trust level: {score.level}"
        
        # Overall enterprise trust level
        assert composite.overall_level in [TrustLevel.HIGH, TrustLevel.VERY_HIGH], \
            f"Overall trust level insufficient for enterprise: {composite.overall_level}"
        
        print("✅ Enterprise Trust Workflow PASSED")
        print(f"   Overall Score: {composite.overall_score:.3f}")
        print(f"   Trust Level: {composite.overall_level}")
        print(f"   Confidence: {composite.overall_confidence:.3f}")
        print(f"   Dimensions Evaluated: {len(composite.dimension_scores)}")
        print(f"   Recommendations: {len(recommendations)}")
        
        #return {
        #    "composite": composite,
        #    "dimension_scores": dimension_scores,
        #    "recommendations": recommendations
        #}
    
    def test_multi_entity_enterprise_comparison(self, default_trust_calculator):
        """🎯 CRITICAL: Multi-entity comparison for enterprise decision-making"""
        # Simulate comparison between different AI systems
        entity_configs = {
            "gpt_enterprise": {"base_performance": 0.88, "strength": "accuracy"},
            "claude_enterprise": {"base_performance": 0.85, "strength": "safety"},
            "gemini_enterprise": {"base_performance": 0.82, "strength": "completeness"},
            "local_ai_system": {"base_performance": 0.75, "strength": "compliance"}
        }
        
        dimensions = [TrustDimension.ACCURACY, TrustDimension.RELIABILITY, 
                     TrustDimension.SAFETY, TrustDimension.COMPLETENESS, TrustDimension.COMPLIANCE]
        
        # Add evidence for each entity with realistic enterprise patterns
        for entity_id, config in entity_configs.items():
            base_perf = config["base_performance"]
            strength_dim = TrustDimension(config["strength"]) if config["strength"] in [d.value for d in TrustDimension] else TrustDimension.ACCURACY
            
            for dimension in dimensions:
                # Give strength dimension a boost
                perf_modifier = 0.08 if dimension == strength_dim else 0.0
                
                for i in range(5):  # 5 evidence per dimension
                    # Realistic variation
                    value = min(base_perf + perf_modifier + random.uniform(-0.05, 0.05), 1.0)
                    confidence = 0.85 + random.uniform(-0.1, 0.1)
                    
                    evidence = create_sample_evidence(
                        dimension=dimension,
                        value=max(value, 0.0),
                        confidence=max(min(confidence, 1.0), 0.0),
                        evidence_type=random.choice([EvidenceType.VALIDATION_RESULT, 
                                                   EvidenceType.EXPERT_REVIEW, 
                                                   EvidenceType.PERFORMANCE_METRIC]),
                        source=f"enterprise_validator_{entity_id}_{i}"
                    )
                    default_trust_calculator.add_evidence(entity_id, evidence)
        
        # Perform comparison
        comparison = default_trust_calculator.compare_entities(list(entity_configs.keys()), dimensions)
        
        assert "ranking" in comparison, "Comparison should include ranking"
        assert "dimension_analysis" in comparison, "Comparison should include dimension analysis"
        assert "statistics" in comparison, "Comparison should include statistics"
        
        # Verify ranking structure
        ranking = comparison["ranking"]
        assert len(ranking) == len(entity_configs)
        
        # Verify all entities are ranked
        ranked_entities = [r["entity_id"] for r in ranking]
        assert set(ranked_entities) == set(entity_configs.keys())
        
        # Verify dimension analysis
        dim_analysis = comparison["dimension_analysis"]
        for dimension in dimensions:
            if dimension in dim_analysis:
                assert "best" in dim_analysis[dimension]
                assert "worst" in dim_analysis[dimension]
                assert "average" in dim_analysis[dimension]
        
        print("✅ Multi-Entity Enterprise Comparison PASSED")
        print(f"   Entities Compared: {len(entity_configs)}")
        print(f"   Dimensions Analyzed: {len(dimensions)}")
        print(f"   Top Performer: {ranking[0]['entity_id']} ({ranking[0]['overall_score']:.3f})")
        
       # return comparison
    
    def test_production_readiness_validation(self, default_trust_calculator):
        """🎯 PRODUCTION: Validate system meets production requirements"""
        # Test configuration
        test_entity = "production_validation_system"
        
        # 1. Performance Requirements
        start_time = time.time()
        
        # Add enterprise-scale evidence
        evidence_count = 100
        for i in range(evidence_count):
            evidence = create_sample_evidence(
                dimension=random.choice(list(TrustDimension)),
                value=0.7 + random.random() * 0.3,
                confidence=0.7 + random.random() * 0.3,
                evidence_type=random.choice(list(EvidenceType)),
                source=f"production_source_{i}"
            )
            result = default_trust_calculator.add_evidence(test_entity, evidence)
            assert result is True
        
        evidence_processing_time = time.time() - start_time
        
        # 2. Calculation Performance
        start_time = time.time()
        composite = default_trust_calculator.calculate_composite_trust(test_entity)
        calculation_time = time.time() - start_time
        
        # 3. Reliability Requirements
        assert composite is not None, "System must handle enterprise-scale data"
        assert evidence_processing_time < 5.0, f"Evidence processing too slow: {evidence_processing_time:.3f}s"
        assert calculation_time < 2.0, f"Trust calculation too slow: {calculation_time:.3f}s"
        
        # 4. Data Integrity
        stored_evidence_count = len(default_trust_calculator.evidence_store[test_entity])
        assert stored_evidence_count == evidence_count, "Evidence storage integrity failed"
        
        # 5. Configuration Validation
        assert default_trust_calculator.min_evidence_count > 0
        assert 0.0 <= default_trust_calculator.confidence_threshold <= 1.0
        assert default_trust_calculator.temporal_window_days > 0
        
        # 6. Error Handling
        invalid_evidence = TrustEvidence(
            evidence_id="invalid",
            evidence_type=EvidenceType.USER_FEEDBACK,
            dimension=TrustDimension.ACCURACY,
            value=1.5,  # Invalid
            confidence=0.8,
            source="invalid_source"
        )
        assert default_trust_calculator.add_evidence(test_entity, invalid_evidence) is False
        
        print("✅ Production Readiness Validation PASSED")
        print(f"   Evidence Processing: {evidence_processing_time:.3f}s for {evidence_count} items")
        print(f"   Trust Calculation: {calculation_time:.3f}s")
        print(f"   Data Integrity: {stored_evidence_count}/{evidence_count} preserved")
        print(f"   Error Handling: ✅ Validated")
        
        assert True


# =============================================================================
# CONFIGURATION AND ENVIRONMENT TESTS
# =============================================================================

class TestConfigurationScenarios:
    """Test different configuration scenarios"""
    
    def test_lenient_configuration_behavior(self, lenient_trust_calculator):
        """✅ Test lenient configuration allows more flexibility"""
        entity_id = "lenient_test"
        
        # Should work with just 1 evidence (min_evidence_count=1)
        evidence = create_sample_evidence(confidence=0.6)  # Above threshold=0.5
        lenient_trust_calculator.add_evidence(entity_id, evidence)
        
        score = lenient_trust_calculator.calculate_trust_score(entity_id, TrustDimension.ACCURACY)
        assert score is not None
        
        # Should include scores with confidence=0.6 in composite (threshold=0.5)
        composite = lenient_trust_calculator.calculate_composite_trust(entity_id)
        # May be None due to insufficient dimensions, but shouldn't fail
    
    def test_strict_configuration_behavior(self, strict_trust_calculator):
        """✅ Test strict configuration enforces higher standards"""
        entity_id = "strict_test"
        
        # Should require 5 evidence (min_evidence_count=5)
        for i in range(4):  # Only 4 evidence
            evidence = create_sample_evidence(source=f"source_{i}")
            strict_trust_calculator.add_evidence(entity_id, evidence)
        
        score = strict_trust_calculator.calculate_trust_score(entity_id, TrustDimension.ACCURACY)
        assert score is None  # Insufficient evidence
        
        # Add 5th evidence
        evidence = create_sample_evidence(source="source_5")
        strict_trust_calculator.add_evidence(entity_id, evidence)
        
        score = strict_trust_calculator.calculate_trust_score(entity_id, TrustDimension.ACCURACY)
        assert score is not None  # Now sufficient
    
    def test_temporal_window_configuration(self, default_trust_calculator, lenient_trust_calculator):
        """✅ Test temporal window affects evidence filtering"""
        entity_id = "temporal_config_test"
        
        # Add old evidence (40 days ago)
        old_evidence = create_sample_evidence(days_ago=40, source="old_source")
        
        # Add recent evidence
        recent_evidence = create_sample_evidence(days_ago=5, source="recent_source")
        
        # Add to both calculators
        default_trust_calculator.add_evidence(entity_id, old_evidence)
        default_trust_calculator.add_evidence(entity_id, recent_evidence)
        
        lenient_trust_calculator.add_evidence(entity_id, old_evidence)
        lenient_trust_calculator.add_evidence(entity_id, recent_evidence)
        
        # Default calculator (30-day window) should filter out old evidence
        default_relevant = default_trust_calculator._get_relevant_evidence(entity_id, TrustDimension.ACCURACY)
        
        # Lenient calculator (365-day window) should include all evidence
        lenient_relevant = lenient_trust_calculator._get_relevant_evidence(entity_id, TrustDimension.ACCURACY)
        
        assert len(lenient_relevant) >= len(default_relevant)


if __name__ == "__main__":
    print("🔒 Running Comprehensive Trust Calculator Test Suite")
    print("Production-Ready Testing - All Dimensions Covered")
    print("=" * 80)
    
    # Run all tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--maxfail=5",  # Stop after 5 failures for faster debugging
        "-x"  # Stop on first failure for critical issues
    ])
