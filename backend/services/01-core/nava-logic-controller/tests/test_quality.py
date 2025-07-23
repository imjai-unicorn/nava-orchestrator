# backend/services/01-core/nava-logic-controller/tests/test_quality.py
"""
Test Quality Models and Validation
Testing quality assessment and improvement functionality
"""

import pytest
from datetime import datetime
from typing import List, Dict, Any

# Import quality models
try:
    from app.models.quality import (
        QualityDimension, QualityLevel, ValidationStatus,
        QualityScore, QualityThreshold, QualityAssessment, QualityGate,
        QualityImprovement, QualityMetrics,
        ENTERPRISE_QUALITY_THRESHOLDS, QUALITY_GATES,
        calculate_overall_quality_score, determine_quality_level,
        validate_against_gate, get_quality_gate, generate_improvement_suggestions
    )
except ImportError:
    from ..app.models.quality import (
        QualityDimension, QualityLevel, ValidationStatus,
        QualityScore, QualityThreshold, QualityAssessment, QualityGate,
        QualityImprovement, QualityMetrics,
        ENTERPRISE_QUALITY_THRESHOLDS, QUALITY_GATES,
        calculate_overall_quality_score, determine_quality_level,
        validate_against_gate, get_quality_gate, generate_improvement_suggestions
    )

class TestQualityModels:
    """Test quality data models"""
    
    def test_quality_score_creation(self):
        """Test QualityScore model creation"""
        score = QualityScore(
            dimension=QualityDimension.ACCURACY,
            score=0.85,
            confidence=0.90,
            reasoning="Response demonstrates high factual accuracy",
            assessed_by="ai_system"
        )
        
        assert score.dimension == QualityDimension.ACCURACY
        assert score.score == 0.85
        assert score.confidence == 0.90
        assert "accuracy" in score.reasoning.lower()
        assert isinstance(score.assessed_at, datetime)
    
    def test_quality_threshold_validation(self):
        """Test QualityThreshold validation"""
        # Valid threshold
        threshold = QualityThreshold(
            dimension=QualityDimension.COMPLETENESS,
            minimum_score=0.80,
            warning_score=0.70,
            weight=1.2
        )
        
        assert threshold.warning_score < threshold.minimum_score
        
        # Invalid threshold - warning >= minimum should raise error
        with pytest.raises(ValueError):
            QualityThreshold(
                dimension=QualityDimension.COMPLETENESS,
                minimum_score=0.70,
                warning_score=0.80,  # Warning score higher than minimum
                weight=1.0
            )
    
    def test_quality_assessment_creation(self):
        """Test QualityAssessment model creation"""
        scores = [
            QualityScore(
                dimension=QualityDimension.ACCURACY,
                score=0.85,
                confidence=0.90,
                reasoning="High accuracy demonstrated",
                assessed_by="ai"
            ),
            QualityScore(
                dimension=QualityDimension.RELEVANCE,
                score=0.80,
                confidence=0.85,
                reasoning="Good relevance to query",
                assessed_by="ai"
            ),
            QualityScore(
                dimension=QualityDimension.CLARITY,
                score=0.75,
                confidence=0.85,
                reasoning="Clear and understandable",
                assessed_by="ai"
            )
        ]
        
        # Step 2: Calculate overall score
        thresholds = [ENTERPRISE_QUALITY_THRESHOLDS[dim] for dim in [
            QualityDimension.ACCURACY,
            QualityDimension.RELEVANCE,
            QualityDimension.CLARITY
        ]]
        
        overall_score = calculate_overall_quality_score(scores, thresholds)
        quality_level = determine_quality_level(overall_score)
        
        # Step 3: Create assessment
        assessment = QualityAssessment(
            assessment_id="e2e_test_001",
            response_id="resp_e2e_001",
            dimension_scores=scores,
            overall_score=overall_score,
            quality_level=quality_level,
            thresholds_used=thresholds,
            assessment_method="comprehensive",
            passed_validation=overall_score >= 0.75
        )
        
        # Step 4: Validate against gate
        gate = get_quality_gate("standard")
        passed, issues = validate_against_gate(assessment, gate)
        
        # Step 5: Generate improvements if needed
        improvements = generate_improvement_suggestions(assessment)
        
        # Assertions
        assert assessment.overall_score > 0.0
        assert assessment.quality_level in [QualityLevel.GOOD, QualityLevel.EXCELLENT, QualityLevel.SATISFACTORY]
        assert len(assessment.dimension_scores) == 3
        
        if not passed:
            assert len(improvements) > 0
    
    def test_quality_trend_analysis(self):
        """Test quality trend analysis simulation"""
        # Simulate week-over-week quality metrics
        weekly_metrics = []
        
        for week in range(4):
            metrics = QualityMetrics(
                metrics_id=f"week_{week}",
                time_period="week",
                total_assessments=50 + week * 10,
                average_quality_score=0.75 + week * 0.02,  # Improving trend
                pass_rate=0.80 + week * 0.03,
                dimension_averages={
                    QualityDimension.ACCURACY: 0.80 + week * 0.02,
                    QualityDimension.RELEVANCE: 0.75 + week * 0.025,
                    QualityDimension.CLARITY: 0.70 + week * 0.03
                }
            )
            weekly_metrics.append(metrics)
        
        # Verify improvement trend
        assert weekly_metrics[3].average_quality_score > weekly_metrics[0].average_quality_score
        assert weekly_metrics[3].pass_rate > weekly_metrics[0].pass_rate
        
        # Check dimension improvements
        for dimension in [QualityDimension.ACCURACY, QualityDimension.RELEVANCE, QualityDimension.CLARITY]:
            assert weekly_metrics[3].dimension_averages[dimension] > weekly_metrics[0].dimension_averages[dimension]

class TestQualityThresholds:
    """Test enterprise quality thresholds"""
    
    def test_enterprise_thresholds_exist(self):
        """Test that enterprise thresholds are defined"""
        assert QualityDimension.ACCURACY in ENTERPRISE_QUALITY_THRESHOLDS
        assert QualityDimension.SAFETY in ENTERPRISE_QUALITY_THRESHOLDS
        assert QualityDimension.COMPLIANCE in ENTERPRISE_QUALITY_THRESHOLDS
        
        # Check safety and compliance have high thresholds
        safety_threshold = ENTERPRISE_QUALITY_THRESHOLDS[QualityDimension.SAFETY]
        compliance_threshold = ENTERPRISE_QUALITY_THRESHOLDS[QualityDimension.COMPLIANCE]
        
        assert safety_threshold.minimum_score >= 0.90
        assert compliance_threshold.minimum_score >= 0.85
        assert safety_threshold.weight >= 1.5  # Safety should have high weight
    
    def test_threshold_consistency(self):
        """Test threshold configuration consistency"""
        for dimension, threshold in ENTERPRISE_QUALITY_THRESHOLDS.items():
            # Warning should be less than minimum
            assert threshold.warning_score < threshold.minimum_score
            
            # Scores should be in valid range
            assert 0.0 <= threshold.warning_score <= 1.0
            assert 0.0 <= threshold.minimum_score <= 1.0
            
            # Weight should be positive
            assert threshold.weight > 0

class TestQualityCalculations:
    """Test quality calculation functions"""
    
    def test_overall_score_calculation(self):
        """Test overall quality score calculation with weights"""
        scores = [
            QualityScore(
                dimension=QualityDimension.ACCURACY,
                score=0.90,
                confidence=0.95,
                reasoning="High accuracy",
                assessed_by="ai"
            ),
            QualityScore(
                dimension=QualityDimension.RELEVANCE,
                score=0.80,
                confidence=0.90,
                reasoning="Good relevance", 
                assessed_by="ai"
            )
        ]
        
        thresholds = [
            QualityThreshold(
                dimension=QualityDimension.ACCURACY,
                minimum_score=0.80,
                warning_score=0.70,
                weight=2.0  # Higher weight
            ),
            QualityThreshold(
                dimension=QualityDimension.RELEVANCE,
                minimum_score=0.75,
                warning_score=0.65,
                weight=1.0  # Lower weight
            )
        ]
        
        overall_score = calculate_overall_quality_score(scores, thresholds)
        
        # Should be weighted average: (0.90 * 2.0 + 0.80 * 1.0) / (2.0 + 1.0) = 0.867
        expected_score = (0.90 * 2.0 + 0.80 * 1.0) / 3.0
        assert abs(overall_score - expected_score) < 0.01
    
    def test_quality_level_determination(self):
        """Test quality level determination from scores"""
        assert determine_quality_level(0.95) == QualityLevel.EXCELLENT
        assert determine_quality_level(0.85) == QualityLevel.GOOD
        assert determine_quality_level(0.75) == QualityLevel.SATISFACTORY
        assert determine_quality_level(0.65) == QualityLevel.NEEDS_IMPROVEMENT
        assert determine_quality_level(0.50) == QualityLevel.POOR
    
    def test_empty_scores_handling(self):
        """Test handling of empty score lists"""
        overall_score = calculate_overall_quality_score([], [])
        assert overall_score == 0.0

class TestQualityGates:
    """Test quality gate functionality"""
    
    def test_quality_gates_exist(self):
        """Test that quality gates are defined"""
        assert "standard" in QUALITY_GATES
        assert "enterprise" in QUALITY_GATES
        assert "creative" in QUALITY_GATES
        
        standard_gate = get_quality_gate("standard")
        assert standard_gate is not None
        assert len(standard_gate.required_dimensions) >= 3
    
    def test_gate_validation_pass(self):
        """Test quality gate validation - passing case"""
        gate = get_quality_gate("standard")
        
        # Create assessment that should pass
        scores = []
        for dimension in gate.required_dimensions:
            threshold = next(t for t in gate.thresholds if t.dimension == dimension)
            scores.append(QualityScore(
                dimension=dimension,
                score=threshold.minimum_score + 0.1,  # Above minimum
                confidence=0.90,
                reasoning=f"Good {dimension.value}",
                assessed_by="ai"
            ))
        
        assessment = QualityAssessment(
            assessment_id="test_pass",
            response_id="resp_001",
            dimension_scores=scores,
            overall_score=0.85,
            quality_level=QualityLevel.GOOD,
            thresholds_used=gate.thresholds,
            assessment_method="automated",
            passed_validation=True
        )
        
        passed, issues = validate_against_gate(assessment, gate)
        assert passed == True
        assert len(issues) == 0  # No issues for passing assessment
    
    def test_gate_validation_fail(self):
        """Test quality gate validation - failing case"""
        gate = get_quality_gate("enterprise")
        
        # Create assessment that should fail
        scores = []
        for dimension in gate.required_dimensions:
            threshold = next(t for t in gate.thresholds if t.dimension == dimension)
            scores.append(QualityScore(
                dimension=dimension,
                score=threshold.minimum_score - 0.1,  # Below minimum
                confidence=0.80,
                reasoning=f"Poor {dimension.value}",
                assessed_by="ai"
            ))
        
        assessment = QualityAssessment(
            assessment_id="test_fail",
            response_id="resp_001",
            dimension_scores=scores,
            overall_score=0.60,
            quality_level=QualityLevel.NEEDS_IMPROVEMENT,
            thresholds_used=gate.thresholds,
            assessment_method="automated",
            passed_validation=False
        )
        
        passed, issues = validate_against_gate(assessment, gate)
        assert passed == False
        assert len(issues) > 0  # Should have failure issues
    
    def test_enterprise_gate_strictness(self):
        """Test that enterprise gate is stricter than standard"""
        standard_gate = get_quality_gate("standard")
        enterprise_gate = get_quality_gate("enterprise")
        
        # Enterprise should require more dimensions
        assert len(enterprise_gate.required_dimensions) >= len(standard_gate.required_dimensions)
        
        # Enterprise should be in strict mode
        assert enterprise_gate.strict_mode == True
        assert enterprise_gate.compliance_required == True
        assert enterprise_gate.audit_trail == True

class TestQualityImprovement:
    """Test quality improvement suggestions"""
    
    def test_improvement_suggestion_generation(self):
        """Test generation of improvement suggestions"""
        # Create assessment with room for improvement
        scores = [
            QualityScore(
                dimension=QualityDimension.ACCURACY,
                score=0.65,  # Below good threshold
                confidence=0.80,
                reasoning="Some inaccuracies detected",
                assessed_by="ai"
            ),
            QualityScore(
                dimension=QualityDimension.CLARITY,
                score=0.55,  # Poor score
                confidence=0.75,
                reasoning="Response lacks clarity",
                assessed_by="ai"
            )
        ]
        
        assessment = QualityAssessment(
            assessment_id="improve_test",
            response_id="resp_001",
            dimension_scores=scores,
            overall_score=0.60,
            quality_level=QualityLevel.NEEDS_IMPROVEMENT,
            thresholds_used=[],
            assessment_method="automated",
            passed_validation=False
        )
        
        improvements = generate_improvement_suggestions(assessment)
        
        assert len(improvements) == 2  # One for each poor dimension
        
        # Check accuracy improvement
        accuracy_improvement = next(i for i in improvements if i.dimension == QualityDimension.ACCURACY)
        assert accuracy_improvement.current_score == 0.65
        assert accuracy_improvement.target_score > 0.65
        assert len(accuracy_improvement.suggestions) > 0
        assert accuracy_improvement.priority in ["high", "medium", "low"]
        
        # Check clarity improvement
        clarity_improvement = next(i for i in improvements if i.dimension == QualityDimension.CLARITY)
        assert clarity_improvement.current_score == 0.55
        assert clarity_improvement.priority == "high"  # Poor score should be high priority
    
    def test_improvement_priority_assignment(self):
        """Test improvement priority assignment logic"""
        # Poor score should get high priority
        poor_assessment = QualityAssessment(
            assessment_id="poor_test",
            response_id="resp_001",
            dimension_scores=[
                QualityScore(
                    dimension=QualityDimension.ACCURACY,
                    score=0.50,  # Poor
                    confidence=0.80,
                    reasoning="Poor accuracy",
                    assessed_by="ai"
                )
            ],
            overall_score=0.50,
            quality_level=QualityLevel.POOR,
            thresholds_used=[],
            assessment_method="automated",
            passed_validation=False
        )
        
        improvements = generate_improvement_suggestions(poor_assessment)
        assert improvements[0].priority == "high"
        assert improvements[0].estimated_effort in ["medium", "high"]

class TestQualityMetrics:
    """Test quality metrics and analytics"""
    
    def test_quality_metrics_creation(self):
        """Test QualityMetrics model creation"""
        metrics = QualityMetrics(
            metrics_id="metrics_001",
            time_period="week",
            total_assessments=100,
            average_quality_score=0.82,
            pass_rate=0.85
        )
        
        assert metrics.total_assessments == 100
        assert metrics.average_quality_score == 0.82
        assert metrics.pass_rate == 0.85
        assert isinstance(metrics.created_at, datetime)
    
    def test_dimension_metrics_tracking(self):
        """Test dimension-specific metrics tracking"""
        metrics = QualityMetrics(
            metrics_id="dim_metrics_001",
            time_period="month",
            dimension_averages={
                QualityDimension.ACCURACY: 0.85,
                QualityDimension.RELEVANCE: 0.80,
                QualityDimension.CLARITY: 0.75
            }
        )
        
        assert QualityDimension.ACCURACY in metrics.dimension_averages
        assert metrics.dimension_averages[QualityDimension.ACCURACY] == 0.85

class TestQualityIntegration:
    """Test quality system integration scenarios"""
    
    def test_end_to_end_quality_assessment(self):
        """Test complete quality assessment workflow"""
        # Step 1: Create quality scores
        scores = [
            QualityScore(
                dimension=QualityDimension.ACCURACY,
                score=0.85,
                confidence=0.90,
                reasoning="High accuracy demonstrated",
                assessed_by="ai"
            )
        ]
        
        # Complete the test method
        assessment = QualityAssessment(
            assessment_id="e2e_final_test",
            response_id="resp_final_001",
            dimension_scores=scores,
            overall_score=0.85,
            quality_level=QualityLevel.GOOD,
            thresholds_used=[],
            assessment_method="comprehensive",
            passed_validation=True
        )
        
        assert assessment.overall_score == 0.85
        assert len(assessment.dimension_scores) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])