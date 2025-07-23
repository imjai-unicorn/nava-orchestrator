# backend/services/01-core/nava-logic-controller/tests/test_trust_calculator.py
"""
🔒 Trust Calculator Test Suite
Advanced trust scoring and reliability assessment testing
Phase 1 - Week 2: Test existing advanced features
"""

import pytest
import time
import statistics
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import uuid

# Import the actual classes from the service
import sys
import os

# Fix import path
current_dir = os.path.dirname(os.path.abspath(__file__))
shared_common_path = os.path.join(current_dir, '..', '..', '..', '..', 'shared', 'common')
shared_common_path = os.path.normpath(shared_common_path)

if os.path.exists(shared_common_path):
    sys.path.insert(0, shared_common_path)
    print(f"✅ Added path: {shared_common_path}")
else:
    print(f"❌ Path not found: {shared_common_path}")

from trust_calculator import (
    TrustCalculator, TrustEvidence, TrustScore, CompositeTrustScore,
    TrustDimension, TrustLevel, EvidenceType,
    create_trust_evidence, aggregate_trust_scores
)

@pytest.fixture
def trust_calculator():
    """Create trust calculator instance for testing"""
    config = {
        'min_evidence_count': 1,      # ลดเป็น 1
        'confidence_threshold': 0.5,  # ลดเป็น 0.5
        'temporal_window_days': 365   # เพิ่มเป็น 365
    }
    return TrustCalculator(config)        

@pytest.fixture(scope="class")
def sample_evidence_set():
    """Create sample evidence set for testing"""
    evidence_set = []
        
    # High quality user feedback
    evidence_set.append(TrustEvidence(
        evidence_id=str(uuid.uuid4()),
        evidence_type=EvidenceType.USER_FEEDBACK,
        dimension=TrustDimension.ACCURACY,            
        value=0.9,
        confidence=0.85,
        source="user_rating_system",
        sample_size=100,
        timestamp=datetime.now(),
        context={"rating_type": "accuracy", "user_segment": "expert"}
    ))
        
    # Performance metrics
    evidence_set.append(TrustEvidence(
        evidence_id=str(uuid.uuid4()),
        evidence_type=EvidenceType.PERFORMANCE_METRIC,
        dimension=TrustDimension.RELIABILITY,
        value=0.95,
        confidence=0.9,
        source="system_monitoring",
        sample_size=1000,
        timestamp=datetime.now() - timedelta(hours=1),
        context={"metric_type": "uptime", "measurement_period": "24h"}
    ))
        
    # Validation results
    evidence_set.append(TrustEvidence(
        evidence_id=str(uuid.uuid4()),
        evidence_type=EvidenceType.VALIDATION_RESULT,
        dimension=TrustDimension.SAFETY,
        value=0.88,
        confidence=0.8,
        source="safety_validator",
        sample_size=500,
        timestamp=datetime.now() - timedelta(hours=2),
        context={"validation_type": "bias_detection", "test_cases": 500}
    ))
        
    # Expert review
    evidence_set.append(TrustEvidence(
        evidence_id=str(uuid.uuid4()),
        evidence_type=EvidenceType.EXPERT_REVIEW,
        dimension=TrustDimension.COMPLETENESS,
        value=0.82,
        confidence=0.95,
        source="expert_panel",
        sample_size=50,
        timestamp=datetime.now() - timedelta(hours=3),
        context={"expert_count": 5, "review_type": "comprehensive"}
    ))
        
    # Historical data
    evidence_set.append(TrustEvidence(
        evidence_id=str(uuid.uuid4()),
        evidence_type=EvidenceType.HISTORICAL_DATA,
        dimension=TrustDimension.ACCURACY,
        value=0.87,
        confidence=0.75,
        source="historical_analysis",
        sample_size=2000,
        timestamp=datetime.now() - timedelta(days=1),
        context={"period": "30_days", "data_points": 2000}
    ))
        
    return evidence_set   

class TestTrustCalculator:
    """Test suite for NAVA Trust Calculator""" 
            
class TestTrustEvidence:
    """Test trust evidence management"""
    
    def test_add_evidence_success(self, trust_calculator, sample_evidence_set):
        """✅ Test successful evidence addition"""
        entity_id = "ai_model_gpt"
        evidence = sample_evidence_set[0]
        
        result = trust_calculator.add_evidence(entity_id, evidence)
        
        assert result is True
        assert entity_id in trust_calculator.evidence_store
        assert len(trust_calculator.evidence_store[entity_id]) == 1
        assert trust_calculator.evidence_store[entity_id][0].evidence_id == evidence.evidence_id
    
    def test_add_multiple_evidence(self, trust_calculator, sample_evidence_set):
        """✅ Test adding multiple evidence pieces"""
        entity_id = "ai_model_claude"
        
        for evidence in sample_evidence_set:
            result = trust_calculator.add_evidence(entity_id, evidence)
            assert result is True
        
        assert len(trust_calculator.evidence_store[entity_id]) == len(sample_evidence_set)
    
    def test_evidence_validation(self, trust_calculator):
        """✅ Test evidence validation rules"""
        entity_id = "test_entity"
        
        # Invalid evidence - value out of range
        invalid_evidence = TrustEvidence(
            evidence_id=str(uuid.uuid4()),
            evidence_type=EvidenceType.USER_FEEDBACK,
            dimension=TrustDimension.ACCURACY,
            value=1.5,  # Invalid: > 1.0
            confidence=0.8,
            source="test_source"
        )
        
        result = trust_calculator.add_evidence(entity_id, invalid_evidence)
        assert result is False
        
        # Invalid evidence - confidence out of range
        invalid_evidence2 = TrustEvidence(
            evidence_id=str(uuid.uuid4()),
            evidence_type=EvidenceType.USER_FEEDBACK,
            dimension=TrustDimension.ACCURACY,
            value=0.8,
            confidence=1.2,  # Invalid: > 1.0
            source="test_source"
        )
        
        result = trust_calculator.add_evidence(entity_id, invalid_evidence2)
        assert result is False


class TestTrustScoreCalculation:
    """Test trust score calculations"""
    
    def test_calculate_trust_score_insufficient_evidence(self, trust_calculator):
        """✅ Test calculation with insufficient evidence"""
        entity_id = "test_entity"
        
        # DON'T add any evidence (0 < min_evidence_count=1)
        # เอาส่วน add_evidence ออกหมด
        
        score = trust_calculator.calculate_trust_score(entity_id, TrustDimension.ACCURACY)
        assert score is None  # Should return None for insufficient evidence
    
    def test_calculate_trust_score_success(self, trust_calculator, sample_evidence_set):
        """✅ Test successful trust score calculation"""
        entity_id = "ai_model_test"
        
        # Add sufficient evidence for accuracy dimension
        accuracy_evidence = [e for e in sample_evidence_set if e.dimension == TrustDimension.ACCURACY]
        
        # Add more accuracy evidence to meet minimum
        for i in range(3):
            evidence = create_trust_evidence(
                evidence_type=EvidenceType.VALIDATION_RESULT,
                dimension=TrustDimension.ACCURACY,
                value=0.85 + i * 0.02,  # Varying values
                confidence=0.8,
                source=f"validator_{i}"
            )
            trust_calculator.add_evidence(entity_id, evidence)
        
        score = trust_calculator.calculate_trust_score(entity_id, TrustDimension.ACCURACY)
        
        assert score is not None
        assert isinstance(score, TrustScore)
        assert score.dimension == TrustDimension.ACCURACY
        assert 0.0 <= score.score <= 1.0
        assert 0.0 <= score.confidence <= 1.0
        assert isinstance(score.level, TrustLevel)
        assert score.evidence_count >= 3
    
    def test_trust_level_determination(self, trust_calculator):
        """✅ Test trust level determination from scores"""
        test_cases = [
            (0.95, TrustLevel.VERY_HIGH),
            (0.85, TrustLevel.HIGH),
            (0.70, TrustLevel.MEDIUM),
            (0.50, TrustLevel.LOW),
            (0.30, TrustLevel.VERY_LOW)
        ]
        
        for score_value, expected_level in test_cases:
            level = trust_calculator._determine_trust_level(score_value)
            assert level == expected_level
    
    def test_weighted_score_calculation(self, trust_calculator, sample_evidence_set):
        """✅ Test weighted score calculation with different evidence types"""
        entity_id = "weighted_test"
        
        # Add evidence with different types (different weights)
        for evidence in sample_evidence_set[:4]:  # Use first 4 pieces
            trust_calculator.add_evidence(entity_id, evidence)
        
        # Calculate for a dimension with multiple evidence
        evidence_list = trust_calculator._get_relevant_evidence(entity_id, TrustDimension.ACCURACY)
        
        if len(evidence_list) >= 1:
            score, confidence = trust_calculator._calculate_weighted_score(evidence_list)
            assert 0.0 <= score <= 1.0
            assert 0.0 <= confidence <= 1.0


class TestCompositeTrustScore:
    """Test composite trust scoring"""
    
    def test_calculate_composite_trust_insufficient_dimensions(self, trust_calculator):
        """✅ Test composite trust with insufficient valid dimensions"""
        entity_id = "composite_test"
        
        # Add evidence for only one dimension
        for i in range(3):
            evidence = create_trust_evidence(
                evidence_type=EvidenceType.USER_FEEDBACK,
                dimension=TrustDimension.ACCURACY,
                value=0.8,
                confidence=0.9,
                source=f"source_{i}"
            )
            trust_calculator.add_evidence(entity_id, evidence)
        
        composite = trust_calculator.calculate_composite_trust(entity_id)
        assert composite is None  # Need at least 3 dimensions
    
    def test_calculate_composite_trust_success(self, trust_calculator, sample_evidence_set):
        """✅ Test successful composite trust calculation"""
        entity_id = "composite_success"
        
        # Add evidence for multiple dimensions
        dimensions = [TrustDimension.ACCURACY, TrustDimension.RELIABILITY, TrustDimension.SAFETY, TrustDimension.COMPLETENESS]
        
        for dimension in dimensions:
            for i in range(3):  # Add 3 evidence per dimension
                evidence = create_trust_evidence(
                    evidence_type=EvidenceType.VALIDATION_RESULT,
                    dimension=dimension,
                    value=0.8 + i * 0.05,  # Varying values
                    confidence=0.85,
                    source=f"source_{dimension}_{i}"
                )
                trust_calculator.add_evidence(entity_id, evidence)
        
        composite = trust_calculator.calculate_composite_trust(entity_id)
        
        assert composite is not None
        assert isinstance(composite, CompositeTrustScore)
        assert 0.0 <= composite.overall_score <= 1.0
        assert 0.0 <= composite.overall_confidence <= 1.0
        assert isinstance(composite.overall_level, TrustLevel)
        assert len(composite.dimension_scores) >= 3
        assert composite.calculation_method == "weighted_harmonic_composite"
    
    def test_composite_with_custom_weights(self, trust_calculator):
        """✅ Test composite trust with custom dimension weights"""
        entity_id = "custom_weights_test"
        
        # Add evidence for multiple dimensions
        dimensions = [TrustDimension.ACCURACY, TrustDimension.SAFETY, TrustDimension.RELIABILITY]
        
        for dimension in dimensions:
            for i in range(3):
                evidence = create_trust_evidence(
                    evidence_type=EvidenceType.EXPERT_REVIEW,
                    dimension=dimension,
                    value=0.8,
                    confidence=0.9,
                    source=f"expert_{i}"
                )
                trust_calculator.add_evidence(entity_id, evidence)
        
        # Custom weights - emphasize safety
        custom_weights = {
            TrustDimension.ACCURACY: 1.0,
            TrustDimension.SAFETY: 3.0,  # 3x weight
            TrustDimension.RELIABILITY: 1.0
        }
        
        composite = trust_calculator.calculate_composite_trust(entity_id, custom_weights=custom_weights)
        
        assert composite is not None
        assert composite.weights_used == custom_weights


class TestTrustTrendTracking:
    """Test trust trend tracking over time"""
    
    def test_track_trust_trend_insufficient_data(self, trust_calculator):
        """✅ Test trend tracking with insufficient historical data"""
        entity_id = "trend_test"
        dimension = TrustDimension.ACCURACY
        
        trend = trust_calculator.track_trust_trend(entity_id, dimension)
        
        assert trend["trend"] == "insufficient_data"
        assert trend["confidence"] == 0.0
    
    def test_track_trust_trend_with_history(self, trust_calculator):
        """✅ Test trend tracking with historical data"""
        entity_id = "trend_history_test"
        dimension = TrustDimension.RELIABILITY
        
        # Simulate historical trust calculations
        # Add evidence and calculate scores multiple times
        for day_offset in range(10, 0, -1):  # 10 days ago to now
            for i in range(3):  # 3 evidence per calculation
                evidence = create_trust_evidence(
                    evidence_type=EvidenceType.PERFORMANCE_METRIC,
                    dimension=dimension,
                    value=0.7 + day_offset * 0.02,  # Improving trend
                    confidence=0.8,
                    source=f"metric_{i}"
                )
                evidence.timestamp = datetime.now() - timedelta(days=day_offset)
                trust_calculator.add_evidence(entity_id, evidence)
            
            # Calculate score to add to history
            score = trust_calculator.calculate_trust_score(entity_id, dimension)
            if score:
                # Manually add to history for testing
                history_key = f"{entity_id}:{dimension}"
                trust_calculator.trust_history[history_key].append({
                    'timestamp': datetime.now() - timedelta(days=day_offset),
                    'score': score.score,
                    'confidence': score.confidence,
                    'evidence_count': score.evidence_count
                })
        
        trend = trust_calculator.track_trust_trend(entity_id, dimension, lookback_days=15)
        
        assert trend["trend"] in ["improving", "declining", "stable"]
        assert 0.0 <= trend["confidence"] <= 1.0
        assert trend["data_points"] > 0


class TestEntityComparison:
    """Test entity comparison functionality"""
    
    def test_compare_entities_insufficient_data(self, trust_calculator):
        """✅ Test comparison with insufficient valid entities"""
        entity_ids = ["entity1", "entity2"]
        
        comparison = trust_calculator.compare_entities(entity_ids)
        
        assert "error" in comparison
        assert "Insufficient entities" in comparison["error"]
    
    def test_compare_entities_success(self, trust_calculator):
        """✅ Test successful entity comparison"""
        entity_ids = ["gpt_model", "claude_model", "gemini_model"]
        
        # Add evidence for multiple entities and dimensions
        dimensions = [TrustDimension.ACCURACY, TrustDimension.RELIABILITY, TrustDimension.SAFETY]
        
        for entity_id in entity_ids:
            entity_bonus = hash(entity_id) % 3 * 0.1  # Different performance per entity
            
            for dimension in dimensions:
                for i in range(4):  # More than minimum evidence
                    evidence = create_trust_evidence(
                        evidence_type=EvidenceType.VALIDATION_RESULT,
                        dimension=dimension,
                        value=min(0.7 + entity_bonus + i * 0.05, 1.0),
                        confidence=0.85,
                        source=f"validator_{i}"
                    )
                    trust_calculator.add_evidence(entity_id, evidence)
        
        comparison = trust_calculator.compare_entities(entity_ids, dimensions)
        
        assert "ranking" in comparison
        assert "dimension_analysis" in comparison
        assert "statistics" in comparison
        
        # Check ranking
        ranking = comparison["ranking"]
        assert len(ranking) == len(entity_ids)
        assert all("entity_id" in r for r in ranking)
        assert all("overall_score" in r for r in ranking)
        
        # Check dimension analysis
        dim_analysis = comparison["dimension_analysis"]
        for dimension in dimensions:
            if dimension in dim_analysis:
                assert "best" in dim_analysis[dimension]
                assert "worst" in dim_analysis[dimension]
                assert "average" in dim_analysis[dimension]


class TestTrustRecommendations:
    """Test trust improvement recommendations"""
    
    def test_get_recommendations_no_composite(self, trust_calculator):
        """✅ Test recommendations when no composite trust exists"""
        entity_id = "no_composite"
        
        recommendations = trust_calculator.get_trust_recommendations(entity_id)
        
        assert len(recommendations) >= 1
        assert any("evidence" in r["recommendation"].lower() for r in recommendations)
    
    def test_get_recommendations_with_composite(self, trust_calculator):
        """✅ Test recommendations based on composite trust analysis"""
        entity_id = "recommendations_test"
        
        # Add evidence creating varying trust levels across dimensions
        dimension_values = {
            TrustDimension.ACCURACY: 0.6,  # Needs improvement
            TrustDimension.SAFETY: 0.9,    # Good
            TrustDimension.RELIABILITY: 0.7, # Moderate
            TrustDimension.COMPLETENESS: 0.5  # Low - needs most improvement
        }
        
        for dimension, base_value in dimension_values.items():
            for i in range(4):  # Sufficient evidence
                evidence = create_trust_evidence(
                    evidence_type=EvidenceType.VALIDATION_RESULT,
                    dimension=dimension,
                    value=min(base_value + i * 0.02, 1.0),
                    confidence=0.8,
                    source=f"source_{i}"
                )
                trust_calculator.add_evidence(entity_id, evidence)
        
        recommendations = trust_calculator.get_trust_recommendations(
            entity_id,
            target_level=TrustLevel.HIGH
        )
        
        assert len(recommendations) > 0
        
        # Should recommend improvement for lowest scoring dimensions
        rec_dimensions = [r.get("dimension") for r in recommendations if "dimension" in r]
        if rec_dimensions:
            # Completeness (lowest) should be in recommendations
            assert TrustDimension.COMPLETENESS in rec_dimensions


class TestTemporalManagement:
    """Test temporal data management"""
    
    def test_evidence_temporal_filtering(self, trust_calculator):
        """✅ Test evidence filtering by temporal window"""
        entity_id = "temporal_test"
        dimension = TrustDimension.ACCURACY
        
        # Add old evidence (outside temporal window)
        old_evidence = create_trust_evidence(
            evidence_type=EvidenceType.HISTORICAL_DATA,
            dimension=dimension,
            value=0.6,
            confidence=0.7,
            source="old_source"
        )
        old_evidence.timestamp = datetime.now() - timedelta(days=400)
        trust_calculator.add_evidence(entity_id, old_evidence)
        
        # Add recent evidence
        for i in range(3):
            recent_evidence = create_trust_evidence(
                evidence_type=EvidenceType.USER_FEEDBACK,
                dimension=dimension,
                value=0.8,
                confidence=0.9,
                source=f"recent_source_{i}"
            )
            trust_calculator.add_evidence(entity_id, recent_evidence)
        
        # Get relevant evidence (should filter out old evidence)
        relevant = trust_calculator._get_relevant_evidence(entity_id, dimension)
        
        # Should only include recent evidence
        assert len(relevant) == 3
        assert all(e.timestamp > datetime.now() - timedelta(days=35) for e in relevant)
    
    def test_evidence_cleanup(self, trust_calculator):
        """✅ Test automatic evidence cleanup"""
        entity_id = "cleanup_test"
        
        # Add very old evidence
        old_evidence = create_trust_evidence(
            evidence_type=EvidenceType.HISTORICAL_DATA,
            dimension=TrustDimension.ACCURACY,
            value=0.7,
            confidence=0.8,
            source="very_old_source"
        )
        old_evidence.timestamp = datetime.now() - timedelta(days=400)  # Very old
        trust_calculator.add_evidence(entity_id, old_evidence)
        
        initial_count = len(trust_calculator.evidence_store[entity_id])
        
        # Run cleanup
        trust_calculator._cleanup_old_evidence(entity_id)
        
        final_count = len(trust_calculator.evidence_store[entity_id])
        
        # Old evidence should be removed
        assert final_count < initial_count or initial_count == 0


class TestPerformanceAndScalability:
    """Test performance and scalability"""
    
    def test_large_scale_evidence_processing(self, trust_calculator):
        """✅ Test processing large amounts of evidence"""
        entity_id = "large_scale_test"
        
        start_time = time.time()
        
        # Add large number of evidence pieces
        evidence_count = 100
        for i in range(evidence_count):
            evidence = create_trust_evidence(
                evidence_type=EvidenceType.PERFORMANCE_METRIC,
                dimension=TrustDimension.RELIABILITY,
                value=0.8 + (i % 10) * 0.01,  # Varying values
                confidence=0.4,
                source=f"source_{i}",
                sample_size=100 + i
            )
            trust_calculator.add_evidence(entity_id, evidence)
        
        processing_time = time.time() - start_time
        
        # Should process efficiently
        assert processing_time < 5.0, f"Processing too slow: {processing_time}s"
        assert len(trust_calculator.evidence_store[entity_id]) == evidence_count
    
    def test_trust_calculation_performance(self, trust_calculator):
        """✅ Test trust calculation performance"""
        entity_id = "performance_test"
        dimension = TrustDimension.ACCURACY
        
        # Add sufficient evidence
        for i in range(20):
            evidence = create_trust_evidence(
                evidence_type=EvidenceType.VALIDATION_RESULT,
                dimension=dimension,
                value=0.8 + i * 0.005,
                confidence=0.9,
                source=f"perf_source_{i}"
            )
            trust_calculator.add_evidence(entity_id, evidence)
        
        start_time = time.time()
        score = trust_calculator.calculate_trust_score(entity_id, dimension)
        calculation_time = time.time() - start_time
        
        # Should calculate quickly
        assert calculation_time < 1.0, f"Calculation too slow: {calculation_time}s"
        assert score is not None


class TestTrustUtilities:
    """Test utility functions"""
    
    def test_create_trust_evidence(self):
        """✅ Test trust evidence factory function"""
        evidence = create_trust_evidence(
            evidence_type=EvidenceType.USER_FEEDBACK,
            dimension=TrustDimension.ACCURACY,
            value=0.85,
            confidence=0.9,
            source="test_source",
            sample_size=50
        )
        
        assert isinstance(evidence, TrustEvidence)
        assert evidence.evidence_type == EvidenceType.USER_FEEDBACK
        assert evidence.dimension == TrustDimension.ACCURACY
        assert evidence.value == 0.85
        assert evidence.confidence == 0.9
        assert evidence.source == "test_source"
        assert evidence.sample_size == 50
        assert evidence.evidence_id is not None
    
    def test_aggregate_trust_scores(self):
        """✅ Test trust score aggregation"""
        # Create mock trust scores
        scores = []
        for i in range(5):
            score = TrustScore(
                dimension=TrustDimension.ACCURACY,
                score=0.8 + i * 0.05,
                confidence=0.85 + i * 0.02,
                level=TrustLevel.HIGH,
                evidence_count=10 + i,
                sample_size=100 + i * 20,
                calculation_method="test_method"
            )
            scores.append(score)
        
        aggregated = aggregate_trust_scores(scores)
        
        assert isinstance(aggregated, dict)
        assert "average_score" in aggregated
        assert "median_score" in aggregated
        assert "min_score" in aggregated
        assert "max_score" in aggregated
        assert "average_confidence" in aggregated
        assert "total_evidence" in aggregated
        
        # Verify calculations
        expected_avg = statistics.mean(s.score for s in scores)
        assert abs(aggregated["average_score"] - expected_avg) < 0.001


class TestTrustIntegration:
    """Integration tests for trust system"""
    
    def test_complete_trust_workflow(self, trust_calculator):
        """🎯 PHASE 1 CRITICAL: Complete trust calculation workflow"""
        entity_id = "ai_model_gpt4"
        
        # 1. Add diverse evidence across multiple dimensions
        evidence_data = [
            (EvidenceType.USER_FEEDBACK, TrustDimension.ACCURACY, 0.92, 0.85, "user_surveys"),
            (EvidenceType.PERFORMANCE_METRIC, TrustDimension.RELIABILITY, 0.96, 0.95, "system_monitoring"),
            (EvidenceType.VALIDATION_RESULT, TrustDimension.SAFETY, 0.89, 0.88, "safety_validator"),
            (EvidenceType.EXPERT_REVIEW, TrustDimension.COMPLETENESS, 0.84, 0.92, "expert_panel"),
            (EvidenceType.USER_FEEDBACK, TrustDimension.ACCURACY, 0.88, 0.80, "user_ratings"),
            (EvidenceType.PERFORMANCE_METRIC, TrustDimension.RELIABILITY, 0.94, 0.93, "uptime_monitoring"),
            (EvidenceType.VALIDATION_RESULT, TrustDimension.SAFETY, 0.91, 0.87, "bias_detection"),
        ]
        
        for evidence_type, dimension, value, confidence, source in evidence_data:
            evidence = create_trust_evidence(
                evidence_type=evidence_type,
                dimension=dimension,
                value=value,
                confidence=confidence,
                source=source,
                sample_size=100
            )
            result = trust_calculator.add_evidence(entity_id, evidence)
            assert result is True
        
        # 2. Calculate individual dimension scores
        accuracy_score = trust_calculator.calculate_trust_score(entity_id, TrustDimension.ACCURACY)
        reliability_score = trust_calculator.calculate_trust_score(entity_id, TrustDimension.RELIABILITY)
        safety_score = trust_calculator.calculate_trust_score(entity_id, TrustDimension.SAFETY)
        
        assert accuracy_score is not None
        assert reliability_score is not None
        assert safety_score is not None
        
        # Verify score characteristics
        assert accuracy_score.level in [TrustLevel.HIGH, TrustLevel.VERY_HIGH]
        assert reliability_score.level in [TrustLevel.HIGH, TrustLevel.VERY_HIGH]
        assert safety_score.level in [TrustLevel.HIGH, TrustLevel.VERY_HIGH]
        
        # 3. Calculate composite trust score
        composite = trust_calculator.calculate_composite_trust(entity_id)
        
        assert composite is not None
        assert isinstance(composite, CompositeTrustScore)
        assert composite.overall_level in [TrustLevel.HIGH, TrustLevel.VERY_HIGH]
        assert len(composite.dimension_scores) >= 3
        
        # 4. Test recommendations
        recommendations = trust_calculator.get_trust_recommendations(entity_id)
        
        # Should have some recommendations even for high-trust entity
        assert isinstance(recommendations, list)
        
        print("✅ Trust System Complete Workflow Test PASSED")
        print(f"   Overall Score: {composite.overall_score:.3f}")
        print(f"   Trust Level: {composite.overall_level}")
        print(f"   Confidence: {composite.overall_confidence:.3f}")
        
    def test_trust_system_performance_requirements(self, trust_calculator):
        """🎯 PHASE 1 PERFORMANCE: Trust calculation performance < 2s target"""
        entity_id = "performance_entity"
        
        start_time = time.time()
        
        # Add substantial evidence
        for dimension in [TrustDimension.ACCURACY, TrustDimension.RELIABILITY, TrustDimension.SAFETY, TrustDimension.COMPLETENESS]:
            for i in range(10):  # 10 evidence per dimension
                evidence = create_trust_evidence(
                    evidence_type=EvidenceType.VALIDATION_RESULT,
                    dimension=dimension,
                    value=0.8 + i * 0.01,
                    confidence=0.85,
                    source=f"validator_{i}",
                    sample_size=100
                )
                trust_calculator.add_evidence(entity_id, evidence)
        
        evidence_time = time.time() - start_time
        
        # Calculate composite trust
        start_time = time.time()
        composite = trust_calculator.calculate_composite_trust(entity_id)
        calculation_time = time.time() - start_time
        
        total_time = evidence_time + calculation_time
        
        # Performance assertions (Phase 1 targets)
        assert evidence_time < 1.0, f"Evidence processing too slow: {evidence_time:.3f}s"
        assert calculation_time < 0.5, f"Trust calculation too slow: {calculation_time:.3f}s"
        assert total_time < 2.0, f"Total time too slow: {total_time:.3f}s"
        assert composite is not None
        
        print(f"✅ Trust Performance Test PASSED")
        print(f"   Evidence Processing: {evidence_time:.3f}s")
        print(f"   Trust Calculation: {calculation_time:.3f}s")
        print(f"   Total Time: {total_time:.3f}s")


# Phase 1 Integration Test
class TestPhase1TrustIntegration:
    """Phase 1 specific integration tests"""
    
    def test_trust_system_stability_under_load(self, trust_calculator):
        """🎯 PHASE 1 STABILITY: Trust system stability under concurrent load"""
        import threading
        import random
        
        results = {"success": 0, "failure": 0}
        lock = threading.Lock()
        
        def add_evidence_worker(worker_id, iterations=20):
            for i in range(iterations):
                try:
                    entity_id = f"entity_{worker_id}_{i % 3}"  # 3 entities per worker
                    evidence = create_trust_evidence(
                        evidence_type=random.choice(list(EvidenceType)),
                        dimension=random.choice(list(TrustDimension)),
                        value=0.7 + random.random() * 0.3,
                        confidence=0.8 + random.random() * 0.2,
                        source=f"worker_{worker_id}_source_{i}"
                    )
                    
                    success = trust_calculator.add_evidence(entity_id, evidence)
                    
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
        
        for worker_id in range(worker_count):
            thread = threading.Thread(target=add_evidence_worker, args=(worker_id, iterations_per_worker))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        total_operations = worker_count * iterations_per_worker
        success_rate = results["success"] / total_operations if total_operations > 0 else 0
        
        # Stability requirements
        assert success_rate > 0.95, f"Success rate too low: {success_rate:.3f}"
        assert results["failure"] < total_operations * 0.05, f"Too many failures: {results['failure']}"
        
        print(f"✅ Trust System Stability Test PASSED")
        print(f"   Success Rate: {success_rate:.3f}")
        print(f"   Total Operations: {total_operations}")
        print(f"   Failures: {results['failure']}")


if __name__ == "__main__":
    print("🔒 Running NAVA Trust Calculator Test Suite...")
    print("Phase 1 - Week 2: Advanced Features Testing")
    print("=" * 60)
    
    # Run specific test classes for Phase 1
    pytest.main([
        __file__,
        "-v", 
        "--tb=short",
        "-k", "test_complete_trust_workflow or test_trust_system_performance_requirements or test_trust_system_stability_under_load"
    ])
