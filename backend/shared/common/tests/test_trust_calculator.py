# backend/services/shared/common/tests/test_trust_calculator.py
"""
Test suite for trust calculator
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Import the models we're testing
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from trust_calculator import (
    TrustDimension, TrustLevel, EvidenceType,
    TrustEvidence, TrustScore, CompositeTrustScore, TrustCalculator,
    create_trust_evidence, aggregate_trust_scores
)

class TestTrustEvidence:
    """Test TrustEvidence dataclass"""
    
    def test_trust_evidence_creation(self):
        """Test creating a TrustEvidence instance"""
        evidence = TrustEvidence(
            evidence_id="evidence_001",
            evidence_type=EvidenceType.USER_FEEDBACK,
            dimension=TrustDimension.ACCURACY,
            value=0.85,
            confidence=0.9,
            source="user_rating_system"
        )
        
        assert evidence.evidence_id == "evidence_001"
        assert evidence.evidence_type == EvidenceType.USER_FEEDBACK
        assert evidence.dimension == TrustDimension.ACCURACY
        assert evidence.value == 0.85
        assert evidence.confidence == 0.9
        assert evidence.weight == 1.0  # Default value
        assert isinstance(evidence.timestamp, datetime)
    
    def test_trust_evidence_with_optional_fields(self):
        """Test trust evidence with optional fields"""
        evidence = TrustEvidence(
            evidence_id="evidence_002",
            evidence_type=EvidenceType.EXPERT_REVIEW,
            dimension=TrustDimension.SAFETY,
            value=0.95,
            confidence=0.8,
            weight=1.5,
            source="security_expert",
            context={"review_type": "security_audit"},
            validity_period=timedelta(days=90),
            sample_size=100,
            methodology="manual_review",
            verified=True
        )
        
        assert evidence.weight == 1.5
        assert evidence.context["review_type"] == "security_audit"
        assert evidence.validity_period == timedelta(days=90)
        assert evidence.sample_size == 100
        assert evidence.verified

class TestTrustScore:
    """Test TrustScore dataclass"""
    
    def test_trust_score_creation(self):
        """Test creating a TrustScore instance"""
        score = TrustScore(
            dimension=TrustDimension.RELIABILITY,
            score=0.78,
            confidence=0.85,
            level=TrustLevel.HIGH,
            evidence_count=15,
            sample_size=500,
            calculation_method="weighted_evidence"
        )
        
        assert score.dimension == TrustDimension.RELIABILITY
        assert score.score == 0.78
        assert score.confidence == 0.85
        assert score.level == TrustLevel.HIGH
        assert score.evidence_count == 15
        assert score.sample_size == 500
        assert isinstance(score.calculated_at, datetime)

class TestCompositeTrustScore:
    """Test CompositeTrustScore dataclass"""
    
    def test_composite_trust_score_creation(self):
        """Test creating a CompositeTrustScore instance"""
        dimension_scores = {
            TrustDimension.ACCURACY: TrustScore(
                dimension=TrustDimension.ACCURACY,
                score=0.85,
                confidence=0.9,
                level=TrustLevel.HIGH,
                evidence_count=10,
                sample_size=200,
                calculation_method="test"
            ),
            TrustDimension.SAFETY: TrustScore(
                dimension=TrustDimension.SAFETY,
                score=0.92,
                confidence=0.95,
                level=TrustLevel.VERY_HIGH,
                evidence_count=8,
                sample_size=150,
                calculation_method="test"
            )
        }
        
        composite = CompositeTrustScore(
            overall_score=0.88,
            overall_level=TrustLevel.HIGH,
            overall_confidence=0.92,
            dimension_scores=dimension_scores,
            calculation_method="weighted_composite",
            weights_used={TrustDimension.ACCURACY: 1.0, TrustDimension.SAFETY: 1.2},
            evidence_summary={EvidenceType.USER_FEEDBACK: 10, EvidenceType.EXPERT_REVIEW: 8}
        )
        
        assert composite.overall_score == 0.88
        assert composite.overall_level == TrustLevel.HIGH
        assert len(composite.dimension_scores) == 2
        assert TrustDimension.ACCURACY in composite.dimension_scores

class TestTrustCalculator:
    """Test TrustCalculator class"""
    
    @pytest.fixture
    def trust_calculator(self):
        """Fixture providing a trust calculator instance"""
        config = {
            'min_evidence_count': 3,
            'confidence_threshold': 0.6,
            'temporal_window_days': 30
        }
        return TrustCalculator(config)
    
    @pytest.fixture
    def sample_evidence_list(self):
        """Fixture providing sample evidence"""
        evidence_list = []
        
        # Create evidence for accuracy dimension
        for i in range(5):
            evidence = TrustEvidence(
                evidence_id=f"acc_evidence_{i}",
                evidence_type=EvidenceType.USER_FEEDBACK,
                dimension=TrustDimension.ACCURACY,
                value=0.8 + (i * 0.05),  # Values from 0.8 to 1.0
                confidence=0.9,
                source=f"user_system_{i}",
                timestamp=datetime.now() - timedelta(days=i)
            )
            evidence_list.append(evidence)
        
        return evidence_list
    
    def test_trust_calculator_initialization(self, trust_calculator):
        """Test trust calculator initialization"""
        assert trust_calculator.min_evidence_count == 3
        assert trust_calculator.confidence_threshold == 0.6
        assert trust_calculator.temporal_window_days == 30
        assert len(trust_calculator.evidence_store) == 0
        assert len(trust_calculator.trust_history) == 0
    
    def test_add_evidence_success(self, trust_calculator):
        """Test successful evidence addition"""
        evidence = TrustEvidence(
            evidence_id="test_evidence",
            evidence_type=EvidenceType.PERFORMANCE_METRIC,
            dimension=TrustDimension.RELIABILITY,
            value=0.75,
            confidence=0.8,
            source="monitoring_system"
        )
        
        result = trust_calculator.add_evidence("entity_123", evidence)
        
        assert result
        assert len(trust_calculator.evidence_store["entity_123"]) == 1
        assert trust_calculator.evidence_store["entity_123"][0].evidence_id == "test_evidence"
    
    def test_add_evidence_validation_failure(self, trust_calculator):
        """Test evidence addition with validation failure"""
        # Invalid evidence with value > 1.0
        invalid_evidence = TrustEvidence(
            evidence_id="invalid_evidence",
            evidence_type=EvidenceType.USER_FEEDBACK,
            dimension=TrustDimension.ACCURACY,
            value=1.5,  # Invalid value > 1.0
            confidence=0.8,
            source="test_source"
        )
        
        result = trust_calculator.add_evidence("entity_123", invalid_evidence)
        
        assert not result
    
    def test_validate_evidence_success(self, trust_calculator):
        """Test successful evidence validation"""
        valid_evidence = TrustEvidence(
            evidence_id="valid_test",
            evidence_type=EvidenceType.VALIDATION_RESULT,
            dimension=TrustDimension.COMPLETENESS,
            value=0.85,
            confidence=0.75,
            source="validation_system"
        )
        
        is_valid = trust_calculator._validate_evidence(valid_evidence)
        assert is_valid
    
    def test_validate_evidence_invalid_value(self, trust_calculator):
        """Test evidence validation with invalid value"""
        invalid_evidence = TrustEvidence(
            evidence_id="invalid_test",
            evidence_type=EvidenceType.USER_FEEDBACK,
            dimension=TrustDimension.ACCURACY,
            value=-0.1,  # Invalid negative value
            confidence=0.8,
            source="test_source"
        )
        
        is_valid = trust_calculator._validate_evidence(invalid_evidence)
        assert not is_valid
    
    def test_validate_evidence_missing_fields(self, trust_calculator):
        """Test evidence validation with missing required fields"""
        incomplete_evidence = TrustEvidence(
            evidence_id="",  # Empty evidence ID
            evidence_type=EvidenceType.USER_FEEDBACK,
            dimension=TrustDimension.ACCURACY,
            value=0.8,
            confidence=0.9,
            source=""  # Empty source
        )
        
        