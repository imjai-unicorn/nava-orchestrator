# backend/services/01-core/nava-logic-controller/app/core/complexity_analyzer.py
"""
Complexity Analyzer - Week 3 Component  
Analyzes request complexity to optimize processing and resource allocation
"""

import logging
import re
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

class ComplexityLevel(Enum):
    """Request complexity levels"""
    SIMPLE = "simple"                    # Basic Q&A, greetings, simple facts
    BASIC_COMPLEX = "basic_complex"      # Simple analysis, explanations  
    INTERMEDIATE = "intermediate"        # Multi-step reasoning, comparisons
    ADVANCED = "advanced"               # Deep analysis, research, planning
    EXPERT = "expert"                   # Complex research, strategy, multi-domain

class ProcessingType(Enum):
    """Processing type recommendations"""
    SINGLE_MODEL = "single_model"       # One AI model sufficient
    SEQUENTIAL = "sequential"           # Multiple models in sequence
    PARALLEL = "parallel"              # Multiple models in parallel
    HYBRID = "hybrid"                  # Combination approach

@dataclass
class ComplexityMetrics:
    """Complexity analysis metrics"""
    overall_complexity: float           # 0.0 - 1.0 complexity score
    complexity_level: ComplexityLevel
    processing_type: ProcessingType
    estimated_time_seconds: float
    recommended_models: List[str]
    resource_requirements: Dict[str, str]
    confidence: float
    
@dataclass
class ComplexityAnalysis:
    """Complete complexity analysis result"""
    metrics: ComplexityMetrics
    analysis_details: Dict[str, Any]
    reasoning: Dict[str, Any]
    recommendations: Dict[str, Any]
    processing_time: float

class ComplexityAnalyzer:
    """Advanced request complexity analyzer"""
    
    def __init__(self):
        # Complexity indicators
        self.simple_patterns = [
            r'\b(hello|hi|hey|thanks|thank you)\b',
            r'\bwhat is\b',
            r'\bdefine\b',
            r'\btell me about\b'
        ]
        
        self.complex_patterns = [
            r'\b(analyze|compare|evaluate|assess|research)\b',
            r'\b(strategy|plan|approach|methodology)\b',
            r'\b(pros and cons|advantages and disadvantages)\b',
            r'\b(step by step|detailed|comprehensive)\b'
        ]
        
        self.expert_patterns = [
            r'\b(implement|design|architect|optimize)\b',
            r'\b(multi|multiple|various|different)\b.*\b(approaches|methods|strategies)\b',
            r'\b(enterprise|business|commercial|industrial)\b',
            r'\b(integration|coordination|orchestration)\b'
        ]
        
        # Complexity keywords with weights
        self.complexity_keywords = {
            # Simple (weight 0.1-0.3)
            'what': 0.1, 'how': 0.2, 'when': 0.1, 'where': 0.1, 'who': 0.1,
            'define': 0.2, 'explain': 0.3, 'tell': 0.1, 'show': 0.2,
            
            # Intermediate (weight 0.4-0.6)
            'analyze': 0.5, 'compare': 0.5, 'contrast': 0.5, 'evaluate': 0.6,
            'discuss': 0.4, 'examine': 0.5, 'investigate': 0.5, 'review': 0.4,
            
            # Complex (weight 0.7-0.9)
            'design': 0.7, 'develop': 0.7, 'create': 0.6, 'build': 0.6,
            'implement': 0.8, 'optimize': 0.8, 'strategy': 0.7, 'plan': 0.7,
            'architect': 0.9, 'enterprise': 0.8, 'integrate': 0.8, 'coordinate': 0.8
        }
        
        # Domain complexity multipliers
        self.domain_multipliers = {
            'technical': 1.3,
            'business': 1.2,
            'scientific': 1.4,
            'creative': 1.1,
            'educational': 1.0,
            'conversational': 0.8
        }
        
        # Model recommendations based on complexity
        self.model_recommendations = {
            ComplexityLevel.SIMPLE: {
                'primary': ['gpt'],
                'processing_type': ProcessingType.SINGLE_MODEL,
                'estimated_time': 2.0
            },
            ComplexityLevel.BASIC_COMPLEX: {
                'primary': ['gpt', 'claude'],
                'processing_type': ProcessingType.SINGLE_MODEL,
                'estimated_time': 5.0
            },
            ComplexityLevel.INTERMEDIATE: {
                'primary': ['claude', 'gemini'],
                'processing_type': ProcessingType.SEQUENTIAL,
                'estimated_time': 10.0
            },
            ComplexityLevel.ADVANCED: {
                'primary': ['claude', 'gemini', 'gpt'],
                'processing_type': ProcessingType.PARALLEL,
                'estimated_time': 20.0
            },
            ComplexityLevel.EXPERT: {
                'primary': ['claude', 'gemini', 'gpt'],
                'processing_type': ProcessingType.HYBRID,
                'estimated_time': 30.0
            }
        }
    
    async def analyze_complexity(
        self, 
        message: str, 
        context: Optional[Dict[str, Any]] = None,
        user_history: Optional[List[Dict[str, Any]]] = None
    ) -> ComplexityAnalysis:
        """Main complexity analysis function"""
        
        start_time = time.time()
        
        try:
            if not message or not message.strip():
                return self._create_simple_analysis(start_time, "Empty message")
            
            # Normalize message
            normalized_message = self._normalize_message(message)
            
            # Multi-dimensional analysis
            analysis_components = {
                'lexical': self._analyze_lexical_complexity(normalized_message),
                'semantic': self._analyze_semantic_complexity(normalized_message),
                'structural': self._analyze_structural_complexity(normalized_message),
                'contextual': self._analyze_contextual_complexity(normalized_message, context),
                'domain': self._analyze_domain_complexity(normalized_message),
                'intent': self._analyze_intent_complexity(normalized_message)
            }
            
            # Calculate overall complexity
            overall_score = self._calculate_overall_complexity(analysis_components)
            
            # Determine complexity level
            complexity_level = self._determine_complexity_level(overall_score)
            
            # Get recommendations
            recommendations = self._get_recommendations(complexity_level, analysis_components)
            
            # Build metrics
            metrics = ComplexityMetrics(
                overall_complexity=overall_score,
                complexity_level=complexity_level,
                processing_type=recommendations['processing_type'],
                estimated_time_seconds=recommendations['estimated_time'],
                recommended_models=recommendations['models'],
                resource_requirements=recommendations['resources'],
                confidence=self._calculate_confidence(analysis_components)
            )
            
            # Build complete analysis
            analysis = ComplexityAnalysis(
                metrics=metrics,
                analysis_details=analysis_components,
                reasoning=self._build_reasoning(analysis_components, overall_score),
                recommendations=recommendations,
                processing_time=time.time() - start_time
            )
            
            logger.info(
                f"🔍 Complexity analysis: {complexity_level.value}, "
                f"score={overall_score:.2f}, "
                f"processing={recommendations['processing_type'].value}"
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Complexity analysis error: {e}")
            return self._create_error_analysis(start_time, str(e))
    
    def _normalize_message(self, message: str) -> str:
        """Normalize message for analysis"""
        # Convert to lowercase and clean
        normalized = message.lower().strip()
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized
    
    def _analyze_lexical_complexity(self, message: str) -> Dict[str, Any]:
        """Analyze lexical complexity (vocabulary, keywords)"""
        
        words = message.split()
        unique_words = set(words)
        
        # Keyword analysis
        complexity_score = 0.0
        matched_keywords = []
        
        for word in words:
            if word in self.complexity_keywords:
                weight = self.complexity_keywords[word]
                complexity_score += weight
                matched_keywords.append((word, weight))
        
        # Normalize by word count
        if words:
            complexity_score = complexity_score / len(words)
        
        return {
            'word_count': len(words),
            'unique_word_ratio': len(unique_words) / max(len(words), 1),
            'complexity_score': min(complexity_score, 1.0),
            'matched_keywords': matched_keywords,
            'avg_word_length': sum(len(word) for word in words) / max(len(words), 1)
        }
    
    def _analyze_semantic_complexity(self, message: str) -> Dict[str, Any]:
        """Analyze semantic complexity (meaning, concepts)"""
        
        # Pattern matching for complexity levels
        simple_matches = sum(1 for pattern in self.simple_patterns if re.search(pattern, message))
        complex_matches = sum(1 for pattern in self.complex_patterns if re.search(pattern, message))
        expert_matches = sum(1 for pattern in self.expert_patterns if re.search(pattern, message))
        
        total_matches = simple_matches + complex_matches + expert_matches
        
        if total_matches == 0:
            semantic_score = 0.5  # Default medium complexity
        else:
            semantic_score = (
                (simple_matches * 0.2) + 
                (complex_matches * 0.6) + 
                (expert_matches * 0.9)
            ) / total_matches
        
        return {
            'simple_patterns': simple_matches,
            'complex_patterns': complex_matches,
            'expert_patterns': expert_matches,
            'semantic_score': semantic_score,
            'concept_density': min(total_matches / 10.0, 1.0)  # Normalize concept density
        }
    
    def _analyze_structural_complexity(self, message: str) -> Dict[str, Any]:
        """Analyze structural complexity (sentence structure, questions)"""
        
        sentences = [s.strip() for s in message.split('.') if s.strip()]
        questions = [s for s in sentences if '?' in s]
        
        # Structural indicators
        has_multiple_questions = len(questions) > 1
        has_long_sentences = any(len(s.split()) > 20 for s in sentences)
        has_lists = any(indicator in message for indicator in ['1.', '2.', '-', '•', 'first', 'second'])
        has_conditions = any(word in message for word in ['if', 'when', 'unless', 'provided', 'assuming'])
        
        # Calculate structural score
        structural_indicators = [
            has_multiple_questions,
            has_long_sentences,
            has_lists,
            has_conditions,
            len(sentences) > 3
        ]
        
        structural_score = sum(structural_indicators) / len(structural_indicators)
        
        return {
            'sentence_count': len(sentences),
            'question_count': len(questions),
            'avg_sentence_length': sum(len(s.split()) for s in sentences) / max(len(sentences), 1),
            'has_multiple_questions': has_multiple_questions,
            'has_long_sentences': has_long_sentences,
            'has_lists': has_lists,
            'has_conditions': has_conditions,
            'structural_score': structural_score
        }
    
    def _analyze_contextual_complexity(self, message: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze contextual complexity (context requirements)"""
        
        context = context or {}
        
        # Context indicators
        requires_history = any(word in message for word in ['previous', 'earlier', 'before', 'continue', 'follow up'])
        requires_external_info = any(word in message for word in ['current', 'latest', 'recent', 'today', 'now'])
        has_user_context = bool(context.get('user_preferences') or context.get('conversation_history'))
        
        contextual_score = 0.3  # Base score
        
        if requires_history:
            contextual_score += 0.2
        if requires_external_info:
            contextual_score += 0.3
        if has_user_context:
            contextual_score += 0.2
        
        return {
            'requires_history': requires_history,
            'requires_external_info': requires_external_info,
            'has_user_context': has_user_context,
            'context_richness': len(context),
            'contextual_score': min(contextual_score, 1.0)
        }
    
    def _analyze_domain_complexity(self, message: str) -> Dict[str, Any]:
        """Analyze domain-specific complexity"""
        
        # Domain indicators
        domains = {
            'technical': ['code', 'programming', 'software', 'algorithm', 'database', 'api', 'framework'],
            'business': ['strategy', 'market', 'revenue', 'profit', 'investment', 'growth', 'management'],
            'scientific': ['research', 'study', 'analysis', 'data', 'experiment', 'hypothesis', 'theory'],
            'creative': ['design', 'art', 'creative', 'story', 'content', 'writing', 'visual'],
            'educational': ['learn', 'teach', 'explain', 'understand', 'knowledge', 'concept'],
            'conversational': ['hello', 'thanks', 'help', 'please', 'chat', 'talk']
        }
        
        domain_scores = {}
        for domain, keywords in domains.items():
            score = sum(1 for keyword in keywords if keyword in message) / len(keywords)
            domain_scores[domain] = score
        
        # Determine primary domain
        primary_domain = max(domain_scores.items(), key=lambda x: x[1])
        domain_multiplier = self.domain_multipliers.get(primary_domain[0], 1.0)
        
        return {
            'domain_scores': domain_scores,
            'primary_domain': primary_domain[0],
            'domain_confidence': primary_domain[1],
            'domain_multiplier': domain_multiplier,
            'multi_domain': sum(1 for score in domain_scores.values() if score > 0.1) > 1
        }
    
    def _analyze_intent_complexity(self, message: str) -> Dict[str, Any]:
        """Analyze intent complexity (what user wants to achieve)"""
        
        # Intent categories with complexity scores
        intent_patterns = {
            'information_seeking': {
                'patterns': [r'\bwhat\b', r'\bhow\b', r'\bwhen\b', r'\bwhere\b', r'\bwho\b'],
                'complexity': 0.3
            },
            'explanation_request': {
                'patterns': [r'\bexplain\b', r'\bdescribe\b', r'\btell me about\b'],
                'complexity': 0.4
            },
            'analysis_request': {
                'patterns': [r'\banalyze\b', r'\bcompare\b', r'\bevaluate\b', r'\bassess\b'],
                'complexity': 0.7
            },
            'creation_request': {
                'patterns': [r'\bcreate\b', r'\bwrite\b', r'\bgenerate\b', r'\bbuild\b'],
                'complexity': 0.6
            },
            'problem_solving': {
                'patterns': [r'\bsolve\b', r'\bfix\b', r'\bhelp with\b', r'\btroubleshoot\b'],
                'complexity': 0.8
            },
            'planning_request': {
                'patterns': [r'\bplan\b', r'\bstrategy\b', r'\bapproach\b', r'\bmethodology\b'],
                'complexity': 0.9
            }
        }
        
        intent_scores = {}
        for intent, data in intent_patterns.items():
            score = sum(1 for pattern in data['patterns'] if re.search(pattern, message))
            if score > 0:
                intent_scores[intent] = data['complexity'] * min(score, 1.0)
        
        # Determine primary intent
        if intent_scores:
            primary_intent = max(intent_scores.items(), key=lambda x: x[1])
            intent_complexity = primary_intent[1]
        else:
            primary_intent = ('information_seeking', 0.3)
            intent_complexity = 0.3
        
        return {
            'intent_scores': intent_scores,
            'primary_intent': primary_intent[0],
            'intent_complexity': intent_complexity,
            'multiple_intents': len(intent_scores) > 1
        }
    
    def _calculate_overall_complexity(self, components: Dict[str, Any]) -> float:
        """Calculate overall complexity score from components"""
        
        # Component weights
        weights = {
            'lexical': 0.15,
            'semantic': 0.25,
            'structural': 0.15,
            'contextual': 0.15,
            'domain': 0.15,
            'intent': 0.15
        }
        
        # Calculate weighted score
        total_score = 0.0
        for component, weight in weights.items():
            if component in components:
                if component == 'lexical':
                    score = components[component]['complexity_score']
                elif component == 'semantic':
                    score = components[component]['semantic_score']
                elif component == 'structural':
                    score = components[component]['structural_score']
                elif component == 'contextual':
                    score = components[component]['contextual_score']
                elif component == 'domain':
                    score = components[component]['domain_confidence'] * components[component]['domain_multiplier']
                elif component == 'intent':
                    score = components[component]['intent_complexity']
                else:
                    score = 0.5
                
                total_score += score * weight
        
        return min(max(total_score, 0.0), 1.0)
    
    def _determine_complexity_level(self, score: float) -> ComplexityLevel:
        """Determine complexity level from score"""
        
        if score < 0.2:
            return ComplexityLevel.SIMPLE
        elif score < 0.4:
            return ComplexityLevel.BASIC_COMPLEX
        elif score < 0.6:
            return ComplexityLevel.INTERMEDIATE
        elif score < 0.8:
            return ComplexityLevel.ADVANCED
        else:
            return ComplexityLevel.EXPERT
    
    def _get_recommendations(self, complexity_level: ComplexityLevel, components: Dict[str, Any]) -> Dict[str, Any]:
        """Get processing recommendations based on complexity"""
        
        base_recommendations = self.model_recommendations[complexity_level]
        
        # Adjust based on specific analysis
        recommendations = {
            'models': base_recommendations['primary'].copy(),
            'processing_type': base_recommendations['processing_type'],
            'estimated_time': base_recommendations['estimated_time'],
            'resources': self._determine_resource_requirements(complexity_level, components),
            'optimization_suggestions': self._get_optimization_suggestions(complexity_level, components)
        }
        
        # Specific adjustments
        domain_info = components.get('domain', {})
        if domain_info.get('primary_domain') == 'technical':
            if 'claude' not in recommendations['models']:
                recommendations['models'].append('claude')
        
        if domain_info.get('multi_domain'):
            recommendations['processing_type'] = ProcessingType.PARALLEL
            recommendations['estimated_time'] *= 1.3
        
        return recommendations
    
    def _determine_resource_requirements(self, complexity_level: ComplexityLevel, components: Dict[str, Any]) -> Dict[str, str]:
        """Determine resource requirements"""
        
        requirements = {
            ComplexityLevel.SIMPLE: {'cpu': 'low', 'memory': 'low', 'time': 'minimal'},
            ComplexityLevel.BASIC_COMPLEX: {'cpu': 'low', 'memory': 'medium', 'time': 'short'},
            ComplexityLevel.INTERMEDIATE: {'cpu': 'medium', 'memory': 'medium', 'time': 'medium'},
            ComplexityLevel.ADVANCED: {'cpu': 'high', 'memory': 'high', 'time': 'long'},
            ComplexityLevel.EXPERT: {'cpu': 'high', 'memory': 'high', 'time': 'extended'}
        }
        
        base_requirements = requirements[complexity_level]
        
        # Adjust based on specific needs
        if components.get('contextual', {}).get('requires_external_info'):
            base_requirements['network'] = 'required'
        
        if components.get('domain', {}).get('multi_domain'):
            base_requirements['coordination'] = 'required'
        
        return base_requirements
    
    def _get_optimization_suggestions(self, complexity_level: ComplexityLevel, components: Dict[str, Any]) -> List[str]:
        """Get optimization suggestions"""
        
        suggestions = []
        
        if complexity_level in [ComplexityLevel.ADVANCED, ComplexityLevel.EXPERT]:
            suggestions.append("Consider breaking down into sub-tasks")
            suggestions.append("Use parallel processing for efficiency")
        
        if components.get('structural', {}).get('has_multiple_questions'):
            suggestions.append("Process questions sequentially for better results")
        
        if components.get('contextual', {}).get('requires_external_info'):
            suggestions.append("Gather current information before processing")
        
        if components.get('domain', {}).get('multi_domain'):
            suggestions.append("Use domain-specific models for each area")
        
        return suggestions
    
    def _calculate_confidence(self, components: Dict[str, Any]) -> float:
        """Calculate confidence in complexity analysis"""
        
        # Base confidence
        confidence = 0.8
        
        # Adjust based on analysis quality
        lexical = components.get('lexical', {})
        if lexical.get('word_count', 0) < 5:
            confidence -= 0.2  # Very short messages harder to analyze
        
        semantic = components.get('semantic', {})
        if semantic.get('simple_patterns', 0) + semantic.get('complex_patterns', 0) + semantic.get('expert_patterns', 0) == 0:
            confidence -= 0.1  # No clear patterns
        
        domain = components.get('domain', {})
        if domain.get('domain_confidence', 0) > 0.3:
            confidence += 0.1  # Clear domain identification
        
        return min(max(confidence, 0.3), 1.0)
    
    def _build_reasoning(self, components: Dict[str, Any], overall_score: float) -> Dict[str, Any]:
        """Build reasoning explanation for complexity determination"""
        
        reasoning = {
            'overall_score': overall_score,
            'key_factors': [],
            'dominant_indicators': [],
            'considerations': []
        }
        
        # Identify key factors
        lexical = components.get('lexical', {})
        if lexical.get('complexity_score', 0) > 0.5:
            reasoning['key_factors'].append(f"High lexical complexity ({lexical['complexity_score']:.2f})")
        
        semantic = components.get('semantic', {})
        if semantic.get('expert_patterns', 0) > 0:
            reasoning['key_factors'].append(f"Expert-level patterns detected ({semantic['expert_patterns']})")
        
        domain = components.get('domain', {})
        if domain.get('multi_domain'):
            reasoning['key_factors'].append("Multi-domain request requiring coordination")
        
        intent = components.get('intent', {})
        if intent.get('primary_intent') in ['planning_request', 'problem_solving']:
            reasoning['key_factors'].append(f"Complex intent: {intent['primary_intent']}")
        
        return reasoning
    
    def _create_simple_analysis(self, start_time: float, reason: str) -> ComplexityAnalysis:
        """Create simple complexity analysis for edge cases"""
        
        metrics = ComplexityMetrics(
            overall_complexity=0.1,
            complexity_level=ComplexityLevel.SIMPLE,
            processing_type=ProcessingType.SINGLE_MODEL,
            estimated_time_seconds=2.0,
            recommended_models=['gpt'],
            resource_requirements={'cpu': 'low', 'memory': 'low', 'time': 'minimal'},
            confidence=0.9
        )
        
        return ComplexityAnalysis(
            metrics=metrics,
            analysis_details={'reason': reason},
            reasoning={'simple_case': True, 'reason': reason},
            recommendations={'models': ['gpt'], 'processing_type': ProcessingType.SINGLE_MODEL},
            processing_time=time.time() - start_time
        )
    
    def _create_error_analysis(self, start_time: float, error: str) -> ComplexityAnalysis:
        """Create error fallback analysis"""
        
        metrics = ComplexityMetrics(
            overall_complexity=0.5,
            complexity_level=ComplexityLevel.BASIC_COMPLEX,
            processing_type=ProcessingType.SINGLE_MODEL,
            estimated_time_seconds=5.0,
            recommended_models=['gpt'],
            resource_requirements={'cpu': 'medium', 'memory': 'medium'},
            confidence=0.3
        )
        
        return ComplexityAnalysis(
            metrics=metrics,
            analysis_details={'error': error},
            reasoning={'error_fallback': True, 'error': error},
            recommendations={'models': ['gpt'], 'processing_type': ProcessingType.SINGLE_MODEL},
            processing_time=time.time() - start_time
        )

# Global analyzer instance
complexity_analyzer = ComplexityAnalyzer()

# Helper functions for external use
async def analyze_request_complexity(
    message: str,
    context: Optional[Dict[str, Any]] = None,
    user_history: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Helper function to analyze request complexity"""
    
    try:
        analysis = await complexity_analyzer.analyze_complexity(message, context, user_history)
        
        return {
            'complexity_level': analysis.metrics.complexity_level.value,
            'complexity_score': analysis.metrics.overall_complexity,
            'processing_type': analysis.metrics.processing_type.value,
            'estimated_time': analysis.metrics.estimated_time_seconds,
            'recommended_models': analysis.metrics.recommended_models,
            'resource_requirements': analysis.metrics.resource_requirements,
            'confidence': analysis.metrics.confidence,
            'analysis_details': analysis.analysis_details,
            'recommendations': analysis.recommendations,
            'processing_time': analysis.processing_time
        }
        
    except Exception as e:
        logger.error(f"❌ Complexity analysis helper error: {e}")
        return {
            'complexity_level': 'basic_complex',
            'complexity_score': 0.5,
            'processing_type': 'single_model',
            'estimated_time': 5.0,
            'recommended_models': ['gpt'],
            'resource_requirements': {'cpu': 'medium', 'memory': 'medium'},
            'confidence': 0.3,
            'error': str(e),
            'processing_time': 0.0
        }

def get_quick_complexity_estimate(message: str) -> str:
    """Quick complexity estimate for simple use cases"""
    
    message_lower = message.lower()
    
    # Quick patterns
    if any(word in message_lower for word in ['hello', 'hi', 'thanks', 'what is']):
        return 'simple'
    elif any(word in message_lower for word in ['analyze', 'compare', 'explain', 'strategy']):
        return 'complex'
    elif any(word in message_lower for word in ['implement', 'design', 'optimize', 'enterprise']):
        return 'expert'
    else:
        return 'intermediate'