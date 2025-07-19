# backend/services/01-core/nava-logic-controller/app/core/quality_validator.py
"""
Core Quality Validator - Lightweight version for Phase 1
Basic quality validation integrated into NAVA Logic Controller
"""

import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class CoreQualityValidator:
    """
    Lightweight quality validator for core NAVA operations
    Simplified version of the full quality service for Phase 1
    """
    
    def __init__(self):
        self.enterprise_thresholds = {
            "minimum_quality": 0.75,
            "safety_threshold": 0.95,
            "compliance_threshold": 0.90
        }
        
        self.validation_enabled = True
        logger.info("✅ Core Quality Validator initialized")
    
    def quick_validate(self, response_text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Quick quality validation for immediate use
        Returns basic quality assessment without deep analysis
        """
        if not self.validation_enabled:
            return self._bypass_validation(response_text)
        
        try:
            start_time = time.time()
            
            # Basic quality checks
            basic_score = self._calculate_basic_score(response_text)
            safety_score = self._check_basic_safety(response_text)
            compliance_score = self._check_basic_compliance(response_text)
            
            # Overall assessment
            overall_score = (basic_score * 0.6 + safety_score * 0.3 + compliance_score * 0.1)
            quality_level = self._determine_quality_level(overall_score)
            
            # Check if passes thresholds
            passes_minimum = overall_score >= self.enterprise_thresholds["minimum_quality"]
            passes_safety = safety_score >= self.enterprise_thresholds["safety_threshold"]
            passes_compliance = compliance_score >= self.enterprise_thresholds["compliance_threshold"]
            
            processing_time = time.time() - start_time
            
            result = {
                "overall_score": round(overall_score, 3),
                "quality_level": quality_level,
                "dimension_scores": {
                    "basic_quality": round(basic_score, 3),
                    "safety": round(safety_score, 3),
                    "compliance": round(compliance_score, 3)
                },
                "threshold_checks": {
                    "passes_minimum": passes_minimum,
                    "passes_safety": passes_safety,
                    "passes_compliance": passes_compliance,
                    "overall_pass": passes_minimum and passes_safety and passes_compliance
                },
                "recommendations": self._get_basic_recommendations(overall_score, safety_score, compliance_score),
                "processing_time_ms": round(processing_time * 1000, 2),
                "validator_version": "core_v1.0",
                "timestamp": datetime.now().isoformat()
            }
            
            logger.debug(f"✅ Quick validation: {quality_level} ({overall_score:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Core quality validation error: {e}")
            return self._emergency_validation(response_text)
    
    def _calculate_basic_score(self, response_text: str) -> float:
        """Calculate basic quality score based on simple metrics"""
        score = 0.6  # Base score
        
        # Length check
        length = len(response_text)
        if length < 20:
            score -= 0.3  # Too short
        elif length > 50:
            score += 0.1  # Good length
        elif length > 200:
            score += 0.2  # Comprehensive
        
        # Basic content checks
        if response_text.strip():
            score += 0.1  # Non-empty
        
        # Word count
        words = response_text.split()
        if len(words) >= 10:
            score += 0.1  # Sufficient words
        
        # Sentence structure
        sentences = [s.strip() for s in response_text.split('.') if s.strip()]
        if len(sentences) >= 2:
            score += 0.05  # Multiple sentences
        
        # Check for professional language
        professional_indicators = ['please', 'thank you', 'however', 'therefore', 'additionally']
        professional_count = sum(1 for word in professional_indicators if word.lower() in response_text.lower())
        if professional_count > 0:
            score += 0.05
        
        return min(1.0, max(0.0, score))
    
    def _check_basic_safety(self, response_text: str) -> float:
        """Basic safety check"""
        score = 0.96  # Start with high safety score
        
        response_lower = response_text.lower()
        
        # Check for obvious harmful content
        harmful_keywords = [
            'violence', 'harm', 'dangerous', 'illegal', 'inappropriate',
            'hate', 'offensive', 'explicit', 'abuse'
        ]
        
        harmful_count = sum(1 for keyword in harmful_keywords if keyword in response_lower)
        if harmful_count > 0:
            score -= harmful_count * 0.2
        
        # Check for unprofessional language
        unprofessional = ['stupid', 'dumb', 'idiotic', 'crazy', 'insane']
        unprofessional_count = sum(1 for word in unprofessional if word in response_lower)
        if unprofessional_count > 0:
            score -= unprofessional_count * 0.15
        
        # Positive safety indicators
        safety_indicators = ['please note', 'be careful', 'important to', 'recommend']
        safety_count = sum(1 for indicator in safety_indicators if indicator in response_lower)
        if safety_count > 0:
            score += min(0.05, safety_count * 0.02)
        
        return min(1.0, max(0.0, score))
    
    def _check_basic_compliance(self, response_text: str) -> float:
        """Basic compliance check"""
        score = 0.8  # Base compliance score
        
        response_lower = response_text.lower()
        
        # Professional language indicators
        professional_terms = ['please', 'thank you', 'kindly', 'respectfully']
        professional_count = sum(1 for term in professional_terms if term in response_lower)
        if professional_count > 0:
            score += min(0.1, professional_count * 0.03)
        
        # Policy violation indicators
        violation_terms = ['confidential', 'internal only', 'do not share', 'classified']
        violation_count = sum(1 for term in violation_terms if term in response_lower)
        if violation_count > 0:
            score -= violation_count * 0.2
        
        # Appropriate disclaimers
        disclaimer_terms = ['according to', 'based on', 'please consult', 'recommend']
        disclaimer_count = sum(1 for term in disclaimer_terms if term in response_lower)
        if disclaimer_count > 0:
            score += min(0.05, disclaimer_count * 0.02)
        
        return min(1.0, max(0.0, score))
    
    def _determine_quality_level(self, overall_score: float) -> str:
        """Determine quality level from score"""
        if overall_score >= 0.90:
            return "excellent"
        elif overall_score >= 0.75:
            return "good"
        elif overall_score >= 0.60:
            return "acceptable"
        elif overall_score >= 0.40:
            return "poor"
        else:
            return "unacceptable"
    
    def _get_basic_recommendations(self, basic_score: float, safety_score: float, compliance_score: float) -> list:
        """Get basic improvement recommendations"""
        recommendations = []
        
        if basic_score < 0.7:
            recommendations.append("Improve response comprehensiveness and clarity")
        
        if safety_score < 0.9:
            recommendations.append("Review content for safety and appropriateness")
        
        if compliance_score < 0.8:
            recommendations.append("Ensure professional tone and policy compliance")
        
        if not recommendations:
            recommendations.append("Quality meets standards")
        
        return recommendations
    
    def _bypass_validation(self, response_text: str) -> Dict[str, Any]:
        """Bypass validation when disabled"""
        return {
            "overall_score": 0.75,
            "quality_level": "bypassed",
            "dimension_scores": {
                "basic_quality": 0.75,
                "safety": 0.95,
                "compliance": 0.80
            },
            "threshold_checks": {
                "passes_minimum": True,
                "passes_safety": True,
                "passes_compliance": True,
                "overall_pass": True
            },
            "recommendations": ["Validation bypassed"],
            "processing_time_ms": 0.1,
            "validator_version": "core_v1.0_bypass",
            "timestamp": datetime.now().isoformat()
        }
    
    def _emergency_validation(self, response_text: str) -> Dict[str, Any]:
        """Emergency fallback validation"""
        basic_score = 0.6 if len(response_text) > 50 else 0.3
        
        return {
            "overall_score": basic_score,
            "quality_level": "emergency",
            "dimension_scores": {
                "basic_quality": basic_score,
                "safety": 0.9,
                "compliance": 0.7
            },
            "threshold_checks": {
                "passes_minimum": basic_score >= 0.6,
                "passes_safety": True,
                "passes_compliance": True,
                "overall_pass": basic_score >= 0.6
            },
            "recommendations": ["Emergency validation - manual review recommended"],
            "processing_time_ms": 1.0,
            "validator_version": "core_v1.0_emergency",
            "timestamp": datetime.now().isoformat()
        }
    
    def set_validation_enabled(self, enabled: bool):
        """Enable or disable validation"""
        self.validation_enabled = enabled
        logger.info(f"✅ Core validation {'enabled' if enabled else 'disabled'}")
    
    def get_thresholds(self) -> Dict[str, float]:
        """Get current thresholds"""
        return self.enterprise_thresholds.copy()
    
    def update_threshold(self, threshold_name: str, value: float):
        """Update a specific threshold"""
        if threshold_name in self.enterprise_thresholds:
            self.enterprise_thresholds[threshold_name] = value
            logger.info(f"✅ Updated {threshold_name} threshold to {value}")
        else:
            logger.warning(f"⚠️ Unknown threshold: {threshold_name}")

# Global instance for core usage
core_quality_validator = CoreQualityValidator()

def quick_quality_check(response_text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenient function for quick quality checks
    Used throughout the core logic controller
    """
    return core_quality_validator.quick_validate(response_text, context)

def is_quality_acceptable(response_text: str, min_threshold: float = 0.75) -> bool:
    """
    Simple boolean check for quality acceptance
    Used for quick pass/fail decisions
    """
    try:
        result = core_quality_validator.quick_validate(response_text)
        return result.get("overall_score", 0.0) >= min_threshold
    except Exception as e:
        logger.error(f"❌ Quality acceptance check failed: {e}")
        return True  # Fail open for system stability

def get_quality_score(response_text: str) -> float:
    """
    Get just the quality score
    Used for scoring and ranking responses
    """
    try:
        result = core_quality_validator.quick_validate(response_text)
        return result.get("overall_score", 0.75)
    except Exception as e:
        logger.error(f"❌ Quality score check failed: {e}")
        return 0.75  # Default acceptable score

# Configuration functions
def set_quality_validation(enabled: bool):
    """Enable or disable quality validation globally"""
    core_quality_validator.set_validation_enabled(enabled)

def update_quality_threshold(threshold_name: str, value: float):
    """Update quality threshold"""
    core_quality_validator.update_threshold(threshold_name, value)