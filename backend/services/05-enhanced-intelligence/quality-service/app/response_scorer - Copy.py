# backend/services/05-enhanced-intelligence/quality-service/app/response_scorer.py
"""
Response Scorer
Advanced response scoring engine with comprehensive quality assessment
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging
import time
import math
import re
from datetime import datetime

logger = logging.getLogger(__name__)

# Create router
scorer_router = APIRouter()

# Models
class ScoringRequest(BaseModel):
    response_text: str = Field(..., min_length=1, max_length=50000)
    criteria: Optional[Dict[str, float]] = Field(None, description="Custom scoring criteria weights")
    reference_response: Optional[str] = Field(None, description="Reference response for comparison")
    scoring_mode: str = Field(default="comprehensive", description="comprehensive, quick, comparative")
    context: Optional[Dict[str, Any]] = Field(None)

class ScoringResponse(BaseModel):
    scoring_id: str
    overall_score: float
    detailed_scores: Dict[str, float]
    scoring_breakdown: Dict[str, Any]
    percentile_rank: Optional[float]
    score_explanation: List[str]
    recommendations: List[str]
    processing_time_seconds: float
    timestamp: str

class ResponseScorer:
    """Advanced Response Scoring Engine"""
    
    def __init__(self):
        self.scoring_criteria = {
            "content_quality": {
                "weight": 0.25,
                "factors": ["information_density", "accuracy_indicators", "depth_of_coverage"],
                "optimal_range": [0.7, 1.0]
            },
            "language_quality": {
                "weight": 0.20,
                "factors": ["grammar", "vocabulary", "fluency", "coherence"],
                "optimal_range": [0.8, 1.0]
            },
            "structure_quality": {
                "weight": 0.20,
                "factors": ["organization", "logical_flow", "clarity"],
                "optimal_range": [0.75, 1.0]
            },
            "relevance_score": {
                "weight": 0.15,
                "factors": ["topic_relevance", "context_appropriateness"],
                "optimal_range": [0.8, 1.0]
            },
            "engagement_score": {
                "weight": 0.10,
                "factors": ["readability", "interest_level", "user_value"],
                "optimal_range": [0.6, 1.0]
            },
            "technical_score": {
                "weight": 0.10,
                "factors": ["formatting", "length_appropriateness", "completeness"],
                "optimal_range": [0.7, 1.0]
            }
        }
        
        self.score_ranges = {
            "excellent": {"min": 0.90, "max": 1.00, "description": "Exceptional quality"},
            "very_good": {"min": 0.80, "max": 0.89, "description": "High quality"},
            "good": {"min": 0.70, "max": 0.79, "description": "Good quality"},
            "satisfactory": {"min": 0.60, "max": 0.69, "description": "Acceptable quality"},
            "needs_improvement": {"min": 0.40, "max": 0.59, "description": "Below standard"},
            "poor": {"min": 0.00, "max": 0.39, "description": "Significant issues"}
        }
        
        # Historical data for percentile ranking
        self.score_history = []
        self.max_history_size = 1000
    
    async def score_response(self, request: ScoringRequest) -> Dict[str, Any]:
        """Perform comprehensive response scoring"""
        
        try:
            # Select scoring mode
            if request.scoring_mode == "quick":
                return await self._quick_scoring(request)
            elif request.scoring_mode == "comparative":
                return await self._comparative_scoring(request)
            else:
                return await self._comprehensive_scoring(request)
                
        except Exception as e:
            logger.error(f"❌ Response scoring error: {e}")
            return self._emergency_scoring(request.response_text)
    
    async def _comprehensive_scoring(self, request: ScoringRequest) -> Dict[str, Any]:
        """Comprehensive scoring analysis"""
        
        detailed_scores = {}
        scoring_breakdown = {}
        
        # Custom criteria or default
        criteria = request.criteria or {k: v["weight"] for k, v in self.scoring_criteria.items()}
        
        # Score each criterion
        for criterion, weight in criteria.items():
            if criterion in self.scoring_criteria:
                score, breakdown = self._score_criterion(
                    criterion,
                    request.response_text,
                    request.context or {}
                )
                detailed_scores[criterion] = score
                scoring_breakdown[criterion] = breakdown
        
        # Calculate overall score
        overall_score = sum(score * criteria.get(criterion, 0.1) 
                          for criterion, score in detailed_scores.items())
        
        # Normalize if weights don't sum to 1
        total_weight = sum(criteria.values())
        if total_weight != 1.0:
            overall_score = overall_score / total_weight
        
        # Calculate percentile rank
        percentile = self._calculate_percentile(overall_score)
        
        # Generate explanations
        explanations = self._generate_score_explanations(detailed_scores, scoring_breakdown)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(detailed_scores, overall_score)
        
        # Update history
        self._update_score_history(overall_score)
        
        return {
            "overall_score": overall_score,
            "detailed_scores": detailed_scores,
            "scoring_breakdown": scoring_breakdown,
            "percentile_rank": percentile,
            "explanations": explanations,
            "recommendations": recommendations,
            "score_category": self._get_score_category(overall_score)
        }
    
    def _score_criterion(self, criterion: str, response_text: str, context: Dict[str, Any]) -> tuple:
        """Score a specific criterion"""
        
        if criterion == "content_quality":
            return self._score_content_quality(response_text, context)
        elif criterion == "language_quality":
            return self._score_language_quality(response_text, context)
        elif criterion == "structure_quality":
            return self._score_structure_quality(response_text, context)
        elif criterion == "relevance_score":
            return self._score_relevance(response_text, context)
        elif criterion == "engagement_score":
            return self._score_engagement(response_text, context)
        elif criterion == "technical_score":
            return self._score_technical_aspects(response_text, context)
        else:
            return 0.5, {"method": "default_scoring"}
    
    def _score_content_quality(self, response_text: str, context: Dict[str, Any]) -> tuple:
        """Score content quality"""
        
        score = 0.6  # Base score
        breakdown = {}
        
        # Information density
        word_count = len(response_text.split())
        char_count = len(response_text)
        
        if word_count > 0:
            avg_word_length = char_count / word_count
            breakdown["avg_word_length"] = avg_word_length
            
            # Optimal word length 4-6 characters
            if 4 <= avg_word_length <= 6:
                score += 0.15
            elif avg_word_length > 8:
                score -= 0.1
        
        # Content depth indicators
        depth_indicators = ["because", "therefore", "however", "furthermore", "specifically", "detailed"]
        depth_count = sum(1 for indicator in depth_indicators if indicator in response_text.lower())
        
        if depth_count > 2:
            score += 0.2
            breakdown["depth_indicators"] = depth_count
        elif depth_count == 0:
            score -= 0.15
        
        # Factual content indicators
        factual_indicators = ["data", "research", "study", "according to", "evidence"]
        factual_count = sum(1 for indicator in factual_indicators if indicator in response_text.lower())
        
        if factual_count > 0:
            score += 0.1
            breakdown["factual_indicators"] = factual_count
        
        # Examples and explanations
        has_examples = "example" in response_text.lower() or "for instance" in response_text.lower()
        has_explanations = "explanation" in response_text.lower() or "explain" in response_text.lower()
        
        if has_examples:
            score += 0.05
        if has_explanations:
            score += 0.05
        
        breakdown["content_quality_score"] = score
        return max(0.0, min(1.0, score)), breakdown
    
    def _score_language_quality(self, response_text: str, context: Dict[str, Any]) -> tuple:
        """Score language quality"""
        
        score = 0.7  # Base score
        breakdown = {}
        
        # Basic language metrics
        sentences = [s.strip() for s in response_text.split('.') if s.strip()]
        words = response_text.split()
        
        if sentences and words:
            avg_sentence_length = len(words) / len(sentences)
            breakdown["avg_sentence_length"] = avg_sentence_length
            
            # Optimal sentence length 10-20 words
            if 10 <= avg_sentence_length <= 20:
                score += 0.15
            elif avg_sentence_length > 30:
                score -= 0.2
            elif avg_sentence_length < 5:
                score -= 0.1
        
        # Vocabulary sophistication
        long_words = [w for w in words if len(w) > 6]
        vocab_sophistication = len(long_words) / max(len(words), 1)
        
        breakdown["vocab_sophistication"] = vocab_sophistication
        if 0.2 <= vocab_sophistication <= 0.4:
            score += 0.1  # Good balance
        elif vocab_sophistication > 0.6:
            score -= 0.1  # Too complex
        
        # Repetition check
        word_freq = {}
        for word in words:
            word_lower = word.lower().strip('.,!?')
            if len(word_lower) > 3:  # Skip short words
                word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
        
        repeated_words = [w for w, freq in word_freq.items() if freq > 3]
        if repeated_words:
            score -= len(repeated_words) * 0.02
            breakdown["repeated_words"] = len(repeated_words)
        
        # Transition words (good for flow)
        transitions = ["however", "therefore", "moreover", "furthermore", "additionally", "consequently"]
        transition_count = sum(1 for trans in transitions if trans in response_text.lower())
        
        if transition_count > 0:
            score += min(0.1, transition_count * 0.03)
            breakdown["transitions"] = transition_count
        
        breakdown["language_quality_score"] = score
        return max(0.0, min(1.0, score)), breakdown
    
    def _score_structure_quality(self, response_text: str, context: Dict[str, Any]) -> tuple:
        """Score structure quality"""
        
        score = 0.6  # Base score
        breakdown = {}
        
        # Paragraph structure
        paragraphs = [p.strip() for p in response_text.split('\n\n') if p.strip()]
        breakdown["paragraph_count"] = len(paragraphs)
        
        if len(paragraphs) > 1:
            score += 0.1  # Multiple paragraphs good for structure
        
        # List and enumeration structure
        has_bullets = '•' in response_text or '- ' in response_text
        has_numbers = bool([line for line in response_text.split('\n') if line.strip().startswith(tuple('123456789'))])
        
        if has_bullets or has_numbers:
            score += 0.15
            breakdown["structured_lists"] = True
        
        # Logical flow indicators
        flow_indicators = ["first", "second", "next", "then", "finally", "in conclusion"]
        flow_count = sum(1 for indicator in flow_indicators if indicator in response_text.lower())
        
        if flow_count > 1:
            score += 0.2
            breakdown["logical_flow_indicators"] = flow_count
        
        # Headers and sections (basic detection)
        potential_headers = [line for line in response_text.split('\n') 
                           if line.strip() and line.strip().endswith(':')]
        
        if potential_headers:
            score += 0.1
            breakdown["section_headers"] = len(potential_headers)
        
        # Balance check (no paragraph too long or too short)
        if paragraphs:
            para_lengths = [len(p.split()) for p in paragraphs]
            avg_para_length = sum(para_lengths) / len(para_lengths)
            
            if 20 <= avg_para_length <= 100:  # Good paragraph length
                score += 0.1
            
            breakdown["avg_paragraph_length"] = avg_para_length
        
        breakdown["structure_quality_score"] = score
        return max(0.0, min(1.0, score)), breakdown
    
    def _score_relevance(self, response_text: str, context: Dict[str, Any]) -> tuple:
        """Score relevance"""
        
        score = 0.7  # Base score
        breakdown = {}
        
        # Context keyword matching
        context_keywords = context.get("keywords", [])
        if context_keywords:
            keyword_matches = sum(1 for keyword in context_keywords 
                                if keyword.lower() in response_text.lower())
            relevance_ratio = keyword_matches / len(context_keywords)
            score += relevance_ratio * 0.2
            breakdown["keyword_relevance"] = relevance_ratio
        
        # Topic consistency
        original_query = context.get("original_query", "")
        if original_query:
            query_words = set(original_query.lower().split())
            response_words = set(response_text.lower().split())
            
            # Remove stop words
            stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
            query_words -= stop_words
            response_words -= stop_words
            
            if query_words:
                overlap = len(query_words.intersection(response_words)) / len(query_words)
                score += overlap * 0.3
                breakdown["query_overlap"] = overlap
        
        # On-topic indicators
        focused_phrases = ["regarding", "concerning", "about", "relates to", "pertains to"]
        focus_count = sum(1 for phrase in focused_phrases if phrase in response_text.lower())
        
        if focus_count > 0:
            score += 0.05
            breakdown["focus_indicators"] = focus_count
        
        breakdown["relevance_score"] = score
        return max(0.0, min(1.0, score)), breakdown
    
    def _score_engagement(self, response_text: str, context: Dict[str, Any]) -> tuple:
        """Score engagement level"""
        
        score = 0.6  # Base score
        breakdown = {}
        
        # Readability approximation
        words = response_text.split()
        sentences = [s for s in response_text.split('.') if s.strip()]
        
        if words and sentences:
            avg_words_per_sentence = len(words) / len(sentences)
            
            # Flesch Reading Ease approximation
            if 10 <= avg_words_per_sentence <= 15:
                score += 0.15  # Good readability
            elif avg_words_per_sentence > 25:
                score -= 0.1   # Too complex
            
            breakdown["readability_approx"] = avg_words_per_sentence
        
        # Engaging elements
        engaging_elements = ["example", "imagine", "consider", "picture", "think about"]
        engagement_count = sum(1 for element in engaging_elements if element in response_text.lower())
        
        if engagement_count > 0:
            score += engagement_count * 0.05
            breakdown["engaging_elements"] = engagement_count
        
        # Questions to reader (engagement technique)
        question_marks = response_text.count('?')
        if question_marks > 0:
            score += min(0.1, question_marks * 0.03)
            breakdown["interactive_questions"] = question_marks
        
        # Conversational tone
        conversational_words = ["you", "your", "we", "us", "our"]
        conversational_count = sum(1 for word in conversational_words 
                                 if f" {word} " in f" {response_text.lower()} ")
        
        if conversational_count > 2:
            score += 0.1
            breakdown["conversational_tone"] = conversational_count
        
        breakdown["engagement_score"] = score
        return max(0.0, min(1.0, score)), breakdown
    
    def _score_technical_aspects(self, response_text: str, context: Dict[str, Any]) -> tuple:
        """Score technical aspects"""
        
        score = 0.7  # Base score
        breakdown = {}
        
        # Length appropriateness
        char_count = len(response_text)
        word_count = len(response_text.split())
        
        breakdown["character_count"] = char_count
        breakdown["word_count"] = word_count
        
        # Appropriate length for content type
        if 100 <= char_count <= 2000:  # Good range for most responses
            score += 0.1
        elif char_count < 50:
            score -= 0.2  # Too short
        elif char_count > 5000:
            score -= 0.1  # Too long
        
        # Formatting quality
        has_proper_punctuation = response_text.count('.') + response_text.count('!') + response_text.count('?') > 0
        if has_proper_punctuation:
            score += 0.05
        
        # Consistent capitalization
        sentences = [s.strip() for s in response_text.split('.') if s.strip()]
        properly_capitalized = sum(1 for s in sentences if s and s[0].isupper())
        
        if sentences and properly_capitalized / len(sentences) > 0.8:
            score += 0.1
            breakdown["proper_capitalization"] = properly_capitalized / len(sentences)
        
        # No excessive repetition
        repeated_chars = 0
        if response_text:
            for i in range(len(response_text) - 2):
                substr = response_text[i:i+3]
                if len(set(substr)) == 1:  # All same character
                    repeated_chars = max(repeated_chars, 3)
        
        if repeated_chars <= 2:  # No excessive repetition
            score += 0.05
        else:
            score -= 0.1
            breakdown["excessive_repetition"] = repeated_chars
        
        breakdown["technical_score"] = score
        return max(0.0, min(1.0, score)), breakdown
    
    async def _quick_scoring(self, request: ScoringRequest) -> Dict[str, Any]:
        """Quick scoring for fast evaluation"""
        
        response_text = request.response_text
        
        # Basic metrics
        word_count = len(response_text.split())
        char_count = len(response_text)
        
        # Quick quality indicators
        quick_score = 0.6
        
        # Length appropriateness
        if 50 <= char_count <= 1000:
            quick_score += 0.2
        elif char_count < 20:
            quick_score -= 0.3
        
        # Basic structure
        has_periods = response_text.count('.') > 0
        has_proper_case = response_text and response_text[0].isupper()
        
        if has_periods and has_proper_case:
            quick_score += 0.1
        
        # Content indicators
        if word_count > 10:
            quick_score += 0.1
        
        return {
            "overall_score": max(0.0, min(1.0, quick_score)),
            "detailed_scores": {"quick_evaluation": quick_score},
            "scoring_breakdown": {
                "word_count": word_count,
                "char_count": char_count,
                "method": "quick_scoring"
            },
            "explanations": ["Quick evaluation based on basic metrics"],
            "recommendations": ["Consider comprehensive scoring for detailed analysis"],
            "score_category": self._get_score_category(quick_score)
        }
    
    async def _comparative_scoring(self, request: ScoringRequest) -> Dict[str, Any]:
        """Comparative scoring against reference response"""
        
        if not request.reference_response:
            return await self._comprehensive_scoring(request)
        
        # Score both responses
        main_result = await self._comprehensive_scoring(request)
        
        ref_request = ScoringRequest(
            response_text=request.reference_response,
            criteria=request.criteria,
            scoring_mode="comprehensive",
            context=request.context
        )
        ref_result = await self._comprehensive_scoring(ref_request)
        
        # Compare scores
        comparison = {}
        for criterion in main_result["detailed_scores"]:
            main_score = main_result["detailed_scores"][criterion]
            ref_score = ref_result["detailed_scores"].get(criterion, 0.5)
            comparison[criterion] = {
                "main_score": main_score,
                "reference_score": ref_score,
                "difference": main_score - ref_score,
                "relative_performance": "better" if main_score > ref_score else "worse" if main_score < ref_score else "equal"
            }
        
        # Add comparison data
        main_result["comparison_analysis"] = comparison
        main_result["comparative_summary"] = {
            "overall_comparison": "better" if main_result["overall_score"] > ref_result["overall_score"] else "worse",
            "score_difference": main_result["overall_score"] - ref_result["overall_score"],
            "strengths": [k for k, v in comparison.items() if v["difference"] > 0.1],
            "weaknesses": [k for k, v in comparison.items() if v["difference"] < -0.1]
        }
        
        return main_result
    
    def _calculate_percentile(self, score: float) -> Optional[float]:
        """Calculate percentile rank against historical scores"""
        
        if not self.score_history:
            return None
        
        lower_scores = sum(1 for s in self.score_history if s < score)
        percentile = (lower_scores / len(self.score_history)) * 100
        
        return round(percentile, 1)
    
    def _generate_score_explanations(self, detailed_scores: Dict[str, float], 
                                   scoring_breakdown: Dict[str, Any]) -> List[str]:
        """Generate explanations for scores"""
        
        explanations = []
        
        # Overall assessment
        avg_score = sum(detailed_scores.values()) / len(detailed_scores) if detailed_scores else 0.5
        category = self._get_score_category(avg_score)
        explanations.append(f"Overall quality assessment: {category['description']}")
        
        # Best performing areas
        best_criteria = sorted(detailed_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        if best_criteria:
            best_names = [name.replace('_', ' ').title() for name, _ in best_criteria]
            explanations.append(f"Strongest areas: {', '.join(best_names)}")
        
        # Areas needing improvement
        weak_criteria = [name for name, score in detailed_scores.items() if score < 0.6]
        if weak_criteria:
            weak_names = [name.replace('_', ' ').title() for name in weak_criteria[:2]]
            explanations.append(f"Areas for improvement: {', '.join(weak_names)}")
        
        return explanations
    
    def _generate_recommendations(self, detailed_scores: Dict[str, float], 
                                overall_score: float) -> List[str]:
        """Generate improvement recommendations"""
        
        recommendations = []
        
        # Overall recommendations based on score
        if overall_score < 0.6:
            recommendations.append("Consider comprehensive revision for quality improvement")
        elif overall_score < 0.8:
            recommendations.append("Good foundation, focus on specific improvements")
        else:
            recommendations.append("High quality response, minor optimizations possible")
        
        # Specific recommendations
        for criterion, score in detailed_scores.items():
            if score < 0.6:
                if criterion == "content_quality":
                    recommendations.append("Add more depth and specific examples to content")
                elif criterion == "language_quality":
                    recommendations.append("Improve sentence structure and vocabulary variety")
                elif criterion == "structure_quality":
                    recommendations.append("Enhance organization with better paragraphing and flow")
                elif criterion == "relevance_score":
                    recommendations.append("Ensure stronger alignment with the original topic")
                elif criterion == "engagement_score":
                    recommendations.append("Make content more engaging and reader-friendly")
                elif criterion == "technical_score":
                    recommendations.append("Check formatting, length, and technical presentation")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _get_score_category(self, score: float) -> Dict[str, Any]:
        """Get score category information"""
        
        for category, config in self.score_ranges.items():
            if config["min"] <= score <= config["max"]:
                return {"category": category, "description": config["description"]}
        
        return {"category": "unclassified", "description": "Score outside normal range"}
    
    def _update_score_history(self, score: float):
        """Update score history for percentile calculations"""
        
        self.score_history.append(score)
        
        # Maintain maximum history size
        if len(self.score_history) > self.max_history_size:
            self.score_history = self.score_history[-self.max_history_size:]
    
    def _emergency_scoring(self, response_text: str) -> Dict[str, Any]:
        """Emergency fallback scoring"""
        
        basic_score = 0.5 if len(response_text) > 20 else 0.3
        
        return {
            "overall_score": basic_score,
            "detailed_scores": {"emergency_score": basic_score},
            "scoring_breakdown": {"method": "emergency_scoring"},
            "explanations": ["Emergency scoring - manual review recommended"],
            "recommendations": ["Conduct detailed analysis for accurate scoring"],
            "score_category": self._get_score_category(basic_score)
        }

# Initialize scorer
response_scorer = ResponseScorer()

@scorer_router.post("/score", response_model=ScoringResponse)
async def score_response(request: ScoringRequest):
    """Score response quality comprehensively"""
    
    start_time = time.time()
    
    try:
        # Generate scoring ID
        scoring_id = f"score_{int(time.time())}_{hash(request.response_text) % 1000}"
        
        # Perform scoring
        scoring_result = await response_scorer.score_response(request)
        
        processing_time = time.time() - start_time
        
        response = ScoringResponse(
            scoring_id=scoring_id,
            overall_score=scoring_result.get("overall_score", 0.5),
            detailed_scores=scoring_result.get("detailed_scores", {}),
            scoring_breakdown=scoring_result.get("scoring_breakdown", {}),
            percentile_rank=scoring_result.get("percentile_rank"),
            score_explanation=scoring_result.get("explanations", []),
            recommendations=scoring_result.get("recommendations", []),
            processing_time_seconds=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(
            f"📊 Response scored: {scoring_result.get('overall_score', 0):.2f} "
            f"({scoring_result.get('score_category', {}).get('category', 'unknown')})"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Response scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@scorer_router.get("/history")
async def get_scoring_history():
    """Get scoring history and statistics"""
    
    history = response_scorer.score_history
    
    if not history:
        return {
            "message": "No scoring history available",
            "total_scores": 0,
            "timestamp": datetime.now().isoformat()
        }
    
    return {
        "total_scores": len(history),
        "average_score": sum(history) / len(history),
        "highest_score": max(history),
        "lowest_score": min(history),
        "score_distribution": {
            "excellent": len([s for s in history if s >= 0.9]),
            "very_good": len([s for s in history if 0.8 <= s < 0.9]),
            "good": len([s for s in history if 0.7 <= s < 0.8]),
            "satisfactory": len([s for s in history if 0.6 <= s < 0.7]),
            "needs_improvement": len([s for s in history if 0.4 <= s < 0.6]),
            "poor": len([s for s in history if s < 0.4])
        },
        "recent_scores": history[-10:] if len(history) >= 10 else history,
        "timestamp": datetime.now().isoformat()
    }

@scorer_router.get("/criteria")
async def get_scoring_criteria():
    """Get scoring criteria and weights"""
    
    return {
        "scoring_criteria": response_scorer.scoring_criteria,
        "score_ranges": response_scorer.score_ranges,
        "default_mode": "comprehensive",
        "available_modes": ["comprehensive", "quick", "comparative"],
        "timestamp": datetime.now().isoformat()
    }

@scorer_router.post("/batch")
async def batch_score_responses(responses: List[str], scoring_mode: str = "quick"):
    """Score multiple responses in batch"""
    
    try:
        results = []
        
        for i, response_text in enumerate(responses[:10]):  # Limit to 10
            request = ScoringRequest(
                response_text=response_text,
                scoring_mode=scoring_mode
            )
            
            scoring_result = await response_scorer.score_response(request)
            
            results.append({
                "index": i,
                "overall_score": scoring_result.get("overall_score", 0.5),
                "category": scoring_result.get("score_category", {}).get("category", "unknown"),
                "top_strength": max(scoring_result.get("detailed_scores", {}).items(), 
                                  key=lambda x: x[1], default=("none", 0))[0],
                "main_weakness": min(scoring_result.get("detailed_scores", {}).items(), 
                                   key=lambda x: x[1], default=("none", 1))[0]
            })
        
        # Batch statistics
        scores = [r["overall_score"] for r in results]
        
        return {
            "batch_results": results,
            "batch_statistics": {
                "count": len(results),
                "average_score": sum(scores) / len(scores) if scores else 0,
                "highest_score": max(scores) if scores else 0,
                "lowest_score": min(scores) if scores else 0
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Batch scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@scorer_router.get("/history")
async def get_scoring_history():
    """Get scoring history and statistics"""
    
    history = response_scorer.score_history
    
    if not history:
        return {
            "message": "No scoring history available",
            "total_scores": 0,
            "timestamp": datetime.now().isoformat()
        }
    
    return {
        "total_scores": len(history),
        "average_score": sum(history) / len(history),
        "highest_score": max(history),
        "lowest_score": min(history),
        "score_distribution": {
            "excellent": len([s for s in history if s >= 0.9]),
            "very_good": len([s for s in history if 0.8 <= s < 0.9]),
            "good": len([s for s in history if 0.7 <= s < 0.8]),
            "satisfactory": len([s for s in history if 0.6 <= s < 0.7]),
            "needs_improvement": len([s for s in history if 0.4 <= s < 0.6]),
            "poor": len([s for s in history if s < 0.4])
        },
        "recent_scores": history[-10:] if len(history) >= 10 else history,
        "timestamp": datetime.now().isoformat()
    }