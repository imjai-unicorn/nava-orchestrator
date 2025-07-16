# backend/services/01-core/nava-logic-controller/app/core/result_synthesizer.py
"""
Result Synthesizer - Week 3 Component
Combines and synthesizes outputs from multiple AI models
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

class SynthesisStrategy(Enum):
    """Synthesis strategies for combining AI outputs"""
    CONSENSUS = "consensus"          # Find consensus between responses
    BEST_QUALITY = "best_quality"    # Select highest quality response
    WEIGHTED_MERGE = "weighted_merge" # Weighted combination
    SEQUENTIAL_REFINE = "sequential_refine" # Refine through sequence
    CONFIDENCE_BASED = "confidence_based" # Based on model confidence

@dataclass
class AIResponse:
    """Individual AI response container"""
    model: str
    content: str
    confidence: float
    processing_time: float
    metadata: Dict[str, Any]
    quality_score: Optional[float] = None
    reasoning: Optional[Dict[str, Any]] = None

@dataclass
class SynthesisResult:
    """Result of synthesis process"""
    final_response: str
    synthesis_strategy: SynthesisStrategy
    source_responses: List[AIResponse]
    confidence: float
    quality_metrics: Dict[str, Any]
    synthesis_metadata: Dict[str, Any]
    processing_time: float

class ResultSynthesizer:
    """Advanced result synthesis engine for multi-AI workflows"""
    
    def __init__(self):
        self.synthesis_strategies = {
            SynthesisStrategy.CONSENSUS: self._consensus_synthesis,
            SynthesisStrategy.BEST_QUALITY: self._best_quality_synthesis,
            SynthesisStrategy.WEIGHTED_MERGE: self._weighted_merge_synthesis,
            SynthesisStrategy.SEQUENTIAL_REFINE: self._sequential_refine_synthesis,
            SynthesisStrategy.CONFIDENCE_BASED: self._confidence_based_synthesis
        }
        
        self.quality_weights = {
            "accuracy": 0.25,
            "completeness": 0.20,
            "clarity": 0.20,
            "relevance": 0.20,
            "consistency": 0.15
        }
        
        self.model_reliability = {
            "gpt": 0.85,
            "claude": 0.88,
            "gemini": 0.82,
            "local": 0.75
        }
        
    async def synthesize_responses(
        self, 
        responses: List[AIResponse], 
        strategy: SynthesisStrategy = SynthesisStrategy.CONSENSUS,
        context: Optional[Dict[str, Any]] = None
    ) -> SynthesisResult:
        """Main synthesis function"""
        
        start_time = time.time()
        
        try:
            if not responses:
                raise ValueError("No responses provided for synthesis")
            
            if len(responses) == 1:
                return self._single_response_synthesis(responses[0], start_time)
            
            # Calculate quality scores for all responses
            for response in responses:
                if response.quality_score is None:
                    response.quality_score = self._calculate_quality_score(response)
            
            # Apply synthesis strategy
            synthesis_func = self.synthesis_strategies.get(strategy, self._consensus_synthesis)
            result = await synthesis_func(responses, context or {})
            
            # Calculate overall confidence and quality metrics
            result.confidence = self._calculate_synthesis_confidence(responses, result)
            result.quality_metrics = self._calculate_synthesis_quality(responses, result)
            result.processing_time = time.time() - start_time
            
            logger.info(
                f"✅ Synthesis complete: {strategy.value}, "
                f"confidence={result.confidence:.2f}, "
                f"time={result.processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Synthesis error: {e}")
            # Return fallback synthesis
            return self._fallback_synthesis(responses, start_time, str(e))
    
    async def _consensus_synthesis(self, responses: List[AIResponse], context: Dict[str, Any]) -> SynthesisResult:
        """Find consensus between multiple AI responses"""
        
        # Extract key points from each response
        key_points = []
        for response in responses:
            points = self._extract_key_points(response.content)
            key_points.append({
                "model": response.model,
                "points": points,
                "confidence": response.confidence,
                "quality": response.quality_score
            })
        
        # Find common themes and consensus points
        consensus_points = self._find_consensus_points(key_points)
        
        # Build consensus response
        consensus_response = self._build_consensus_response(consensus_points, responses)
        
        return SynthesisResult(
            final_response=consensus_response,
            synthesis_strategy=SynthesisStrategy.CONSENSUS,
            source_responses=responses,
            confidence=0.0,  # Will be calculated later
            quality_metrics={},  # Will be calculated later
            synthesis_metadata={
                "consensus_points": len(consensus_points),
                "agreement_level": self._calculate_agreement_level(key_points),
                "dominant_themes": consensus_points[:3] if consensus_points else []
            },
            processing_time=0.0  # Will be calculated later
        )
    
    async def _best_quality_synthesis(self, responses: List[AIResponse], context: Dict[str, Any]) -> SynthesisResult:
        """Select the highest quality response"""
        
        # Sort by quality score
        sorted_responses = sorted(responses, key=lambda r: r.quality_score or 0, reverse=True)
        best_response = sorted_responses[0]
        
        # Enhance with insights from other responses
        enhanced_response = self._enhance_with_alternatives(best_response, sorted_responses[1:])
        
        return SynthesisResult(
            final_response=enhanced_response,
            synthesis_strategy=SynthesisStrategy.BEST_QUALITY,
            source_responses=responses,
            confidence=0.0,  # Will be calculated later
            quality_metrics={},  # Will be calculated later
            synthesis_metadata={
                "selected_model": best_response.model,
                "quality_score": best_response.quality_score,
                "enhancement_applied": len(sorted_responses) > 1,
                "quality_ranking": [r.model for r in sorted_responses]
            },
            processing_time=0.0  # Will be calculated later
        )
    
    async def _weighted_merge_synthesis(self, responses: List[AIResponse], context: Dict[str, Any]) -> SynthesisResult:
        """Weighted combination of responses based on model reliability and quality"""
        
        # Calculate weights for each response
        weights = []
        for response in responses:
            model_weight = self.model_reliability.get(response.model, 0.7)
            quality_weight = response.quality_score or 0.5
            confidence_weight = response.confidence
            
            final_weight = (model_weight * 0.4 + quality_weight * 0.4 + confidence_weight * 0.2)
            weights.append(final_weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(responses)] * len(responses)
        
        # Create weighted merge
        merged_response = self._create_weighted_merge(responses, weights)
        
        return SynthesisResult(
            final_response=merged_response,
            synthesis_strategy=SynthesisStrategy.WEIGHTED_MERGE,
            source_responses=responses,
            confidence=0.0,  # Will be calculated later
            quality_metrics={},  # Will be calculated later
            synthesis_metadata={
                "weights": {resp.model: weight for resp, weight in zip(responses, weights)},
                "merge_method": "weighted_content_combination",
                "total_sources": len(responses)
            },
            processing_time=0.0  # Will be calculated later
        )
    
    async def _sequential_refine_synthesis(self, responses: List[AIResponse], context: Dict[str, Any]) -> SynthesisResult:
        """Sequentially refine responses building on each other"""
        
        if not responses:
            raise ValueError("No responses for sequential refinement")
        
        # Start with the first response
        refined_content = responses[0].content
        refinement_history = []
        
        # Sequentially refine with each subsequent response
        for i, response in enumerate(responses[1:], 1):
            refinement = self._refine_with_response(refined_content, response)
            refined_content = refinement["content"]
            refinement_history.append({
                "step": i,
                "model": response.model,
                "improvements": refinement["improvements"],
                "confidence": response.confidence
            })
        
        return SynthesisResult(
            final_response=refined_content,
            synthesis_strategy=SynthesisStrategy.SEQUENTIAL_REFINE,
            source_responses=responses,
            confidence=0.0,  # Will be calculated later
            quality_metrics={},  # Will be calculated later
            synthesis_metadata={
                "refinement_steps": len(refinement_history),
                "refinement_history": refinement_history,
                "base_model": responses[0].model,
                "final_model": responses[-1].model if responses else None
            },
            processing_time=0.0  # Will be calculated later
        )
    
    async def _confidence_based_synthesis(self, responses: List[AIResponse], context: Dict[str, Any]) -> SynthesisResult:
        """Synthesis based on confidence levels"""
        
        # Sort by confidence
        sorted_responses = sorted(responses, key=lambda r: r.confidence, reverse=True)
        
        # Use high-confidence responses as primary sources
        high_confidence = [r for r in sorted_responses if r.confidence >= 0.8]
        medium_confidence = [r for r in sorted_responses if 0.6 <= r.confidence < 0.8]
        
        if high_confidence:
            primary_responses = high_confidence
            secondary_responses = medium_confidence
        else:
            primary_responses = sorted_responses[:2]  # Top 2 if no high confidence
            secondary_responses = sorted_responses[2:]
        
        # Build confidence-based response
        confidence_response = self._build_confidence_response(primary_responses, secondary_responses)
        
        return SynthesisResult(
            final_response=confidence_response,
            synthesis_strategy=SynthesisStrategy.CONFIDENCE_BASED,
            source_responses=responses,
            confidence=0.0,  # Will be calculated later
            quality_metrics={},  # Will be calculated later
            synthesis_metadata={
                "high_confidence_count": len(high_confidence),
                "medium_confidence_count": len(medium_confidence),
                "primary_models": [r.model for r in primary_responses],
                "confidence_threshold": 0.8
            },
            processing_time=0.0  # Will be calculated later
        )
    
    def _single_response_synthesis(self, response: AIResponse, start_time: float) -> SynthesisResult:
        """Handle single response case"""
        
        return SynthesisResult(
            final_response=response.content,
            synthesis_strategy=SynthesisStrategy.BEST_QUALITY,
            source_responses=[response],
            confidence=response.confidence,
            quality_metrics={"single_response_quality": response.quality_score or 0.8},
            synthesis_metadata={
                "single_response": True,
                "model": response.model,
                "original_confidence": response.confidence
            },
            processing_time=time.time() - start_time
        )
    
    def _fallback_synthesis(self, responses: List[AIResponse], start_time: float, error: str) -> SynthesisResult:
        """Fallback synthesis when errors occur"""
        
        # Use the first response or create a basic fallback
        if responses:
            fallback_content = responses[0].content
            fallback_confidence = responses[0].confidence * 0.7  # Reduce confidence due to error
        else:
            fallback_content = "Synthesis processing encountered an error. Please try again."
            fallback_confidence = 0.1
        
        return SynthesisResult(
            final_response=fallback_content,
            synthesis_strategy=SynthesisStrategy.BEST_QUALITY,
            source_responses=responses,
            confidence=fallback_confidence,
            quality_metrics={"error_fallback": True},
            synthesis_metadata={
                "error": error,
                "fallback_used": True,
                "timestamp": datetime.now().isoformat()
            },
            processing_time=time.time() - start_time
        )
    
    def _calculate_quality_score(self, response: AIResponse) -> float:
        """Calculate quality score for a response"""
        
        content = response.content
        
        # Basic quality metrics
        metrics = {
            "length_score": min(len(content) / 500, 1.0),  # Optimal length around 500 chars
            "confidence_score": response.confidence,
            "model_reliability": self.model_reliability.get(response.model, 0.7),
            "response_time_score": max(0, 1.0 - (response.processing_time / 10.0)),  # Penalty for slow responses
            "content_coherence": self._assess_content_coherence(content)
        }
        
        # Weighted average
        weights = {"length_score": 0.15, "confidence_score": 0.25, "model_reliability": 0.20, 
                  "response_time_score": 0.15, "content_coherence": 0.25}
        
        quality_score = sum(metrics[key] * weights[key] for key in metrics)
        
        return min(max(quality_score, 0.0), 1.0)  # Clamp between 0 and 1
    
    def _assess_content_coherence(self, content: str) -> float:
        """Assess content coherence (basic implementation)"""
        
        if not content or len(content.strip()) < 10:
            return 0.0
        
        # Basic coherence metrics
        sentences = content.split('.')
        words = content.split()
        
        coherence_score = 0.7  # Base score
        
        # Length coherence
        if 50 <= len(words) <= 300:
            coherence_score += 0.1
        
        # Sentence structure
        if 3 <= len(sentences) <= 10:
            coherence_score += 0.1
        
        # Avoid repetition penalty
        if len(set(words)) / max(len(words), 1) > 0.7:
            coherence_score += 0.1
        
        return min(coherence_score, 1.0)
    
    def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points from content (basic implementation)"""
        
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        
        # Simple key point extraction
        key_points = []
        for sentence in sentences:
            if len(sentence) > 20 and any(word in sentence.lower() for word in 
                   ['important', 'key', 'main', 'primary', 'essential', 'crucial', 'significant']):
                key_points.append(sentence)
        
        # If no explicit key points, use first few sentences
        if not key_points:
            key_points = sentences[:3]
        
        return key_points[:5]  # Limit to 5 key points
    
    def _find_consensus_points(self, key_points_by_model: List[Dict[str, Any]]) -> List[str]:
        """Find consensus points across models"""
        
        all_points = []
        for model_data in key_points_by_model:
            all_points.extend(model_data["points"])
        
        # Simple consensus: points mentioned by multiple models
        consensus = []
        for point in all_points:
            similar_count = sum(1 for other_point in all_points 
                              if self._calculate_similarity(point, other_point) > 0.7)
            if similar_count > 1:  # Mentioned by at least 2 models
                consensus.append(point)
        
        return list(set(consensus))[:5]  # Remove duplicates and limit
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts (basic implementation)"""
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _build_consensus_response(self, consensus_points: List[str], responses: List[AIResponse]) -> str:
        """Build response from consensus points"""
        
        if not consensus_points:
            # Fallback to best response
            best_response = max(responses, key=lambda r: r.quality_score or 0)
            return best_response.content
        
        # Create structured response from consensus
        intro = "Based on comprehensive analysis, here are the key findings:"
        points_text = "\n".join(f"• {point}" for point in consensus_points)
        
        return f"{intro}\n\n{points_text}"
    
    def _calculate_agreement_level(self, key_points_by_model: List[Dict[str, Any]]) -> float:
        """Calculate overall agreement level between models"""
        
        if len(key_points_by_model) < 2:
            return 1.0
        
        total_similarity = 0.0
        comparisons = 0
        
        for i, model1 in enumerate(key_points_by_model):
            for j, model2 in enumerate(key_points_by_model[i+1:], i+1):
                for point1 in model1["points"]:
                    for point2 in model2["points"]:
                        total_similarity += self._calculate_similarity(point1, point2)
                        comparisons += 1
        
        return total_similarity / comparisons if comparisons > 0 else 0.0
    
    def _enhance_with_alternatives(self, best_response: AIResponse, alternatives: List[AIResponse]) -> str:
        """Enhance best response with insights from alternatives"""
        
        base_content = best_response.content
        
        if not alternatives:
            return base_content
        
        # Add additional insights
        additional_insights = []
        for alt in alternatives[:2]:  # Use top 2 alternatives
            alt_points = self._extract_key_points(alt.content)
            for point in alt_points:
                if not any(self._calculate_similarity(point, base_content) > 0.5 for _ in [None]):
                    additional_insights.append(point)
        
        if additional_insights:
            enhancement = f"\n\nAdditional considerations:\n" + "\n".join(f"• {insight}" for insight in additional_insights[:3])
            return base_content + enhancement
        
        return base_content
    
    def _create_weighted_merge(self, responses: List[AIResponse], weights: List[float]) -> str:
        """Create weighted merge of responses"""
        
        # For simplicity, use the highest weighted response as base
        max_weight_idx = weights.index(max(weights))
        base_response = responses[max_weight_idx]
        
        # Add weighted insights from other responses
        merged_content = base_response.content
        
        for i, (response, weight) in enumerate(zip(responses, weights)):
            if i != max_weight_idx and weight > 0.2:  # Include significant contributors
                key_points = self._extract_key_points(response.content)
                if key_points:
                    merged_content += f"\n\nFrom {response.model} analysis:\n• {key_points[0]}"
        
        return merged_content
    
    def _refine_with_response(self, base_content: str, response: AIResponse) -> Dict[str, Any]:
        """Refine base content with additional response"""
        
        improvements = []
        refined_content = base_content
        
        # Simple refinement: add insights that aren't already covered
        response_points = self._extract_key_points(response.content)
        for point in response_points:
            if not any(self._calculate_similarity(point, base_content) > 0.6 for _ in [None]):
                refined_content += f"\n\nAdditional insight: {point}"
                improvements.append(f"Added insight from {response.model}")
        
        return {
            "content": refined_content,
            "improvements": improvements
        }
    
    def _build_confidence_response(self, primary_responses: List[AIResponse], secondary_responses: List[AIResponse]) -> str:
        """Build response prioritizing high-confidence responses"""
        
        if not primary_responses:
            if secondary_responses:
                return secondary_responses[0].content
            return "Unable to generate confident response."
        
        # Use highest confidence response as base
        base_response = max(primary_responses, key=lambda r: r.confidence)
        response_content = base_response.content
        
        # Add supporting points from other high-confidence responses
        for response in primary_responses:
            if response != base_response:
                points = self._extract_key_points(response.content)
                if points:
                    response_content += f"\n\nSupporting analysis: {points[0]}"
        
        return response_content
    
    def _calculate_synthesis_confidence(self, responses: List[AIResponse], result: SynthesisResult) -> float:
        """Calculate overall confidence for synthesis result"""
        
        if not responses:
            return 0.0
        
        # Base confidence from source responses
        avg_confidence = sum(r.confidence for r in responses) / len(responses)
        
        # Adjust based on synthesis strategy
        strategy_multipliers = {
            SynthesisStrategy.CONSENSUS: 0.9,      # High confidence from consensus
            SynthesisStrategy.BEST_QUALITY: 0.85,  # Good confidence from quality selection
            SynthesisStrategy.WEIGHTED_MERGE: 0.8, # Moderate confidence from merging
            SynthesisStrategy.SEQUENTIAL_REFINE: 0.9, # High confidence from refinement
            SynthesisStrategy.CONFIDENCE_BASED: 0.95  # Highest confidence from confidence-based
        }
        
        strategy_multiplier = strategy_multipliers.get(result.synthesis_strategy, 0.8)
        
        # Adjust for number of sources (more sources = higher confidence)
        source_multiplier = min(1.0, 0.7 + (len(responses) * 0.1))
        
        final_confidence = avg_confidence * strategy_multiplier * source_multiplier
        
        return min(max(final_confidence, 0.0), 1.0)
    
    def _calculate_synthesis_quality(self, responses: List[AIResponse], result: SynthesisResult) -> Dict[str, Any]:
        """Calculate quality metrics for synthesis result"""
        
        return {
            "source_count": len(responses),
            "avg_source_quality": sum(r.quality_score or 0.8 for r in responses) / len(responses),
            "synthesis_strategy": result.synthesis_strategy.value,
            "content_length": len(result.final_response),
            "processing_efficiency": result.processing_time,
            "consensus_level": result.synthesis_metadata.get("agreement_level", 0.5),
            "overall_quality": min(sum(r.quality_score or 0.8 for r in responses) / len(responses) * 1.1, 1.0)
        }

# Global synthesizer instance
result_synthesizer = ResultSynthesizer()

# Helper functions for external use
async def synthesize_ai_responses(
    responses: List[Dict[str, Any]], 
    strategy: str = "consensus",
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Helper function to synthesize AI responses"""
    
    try:
        # Convert responses to AIResponse objects
        ai_responses = []
        for resp in responses:
            ai_response = AIResponse(
                model=resp.get("model", "unknown"),
                content=resp.get("content", ""),
                confidence=resp.get("confidence", 0.8),
                processing_time=resp.get("processing_time", 0.0),
                metadata=resp.get("metadata", {}),
                quality_score=resp.get("quality_score")
            )
            ai_responses.append(ai_response)
        
        # Convert strategy string to enum
        strategy_enum = SynthesisStrategy(strategy) if strategy in [s.value for s in SynthesisStrategy] else SynthesisStrategy.CONSENSUS
        
        # Perform synthesis
        result = await result_synthesizer.synthesize_responses(ai_responses, strategy_enum, context)
        
        return {
            "response": result.final_response,
            "confidence": result.confidence,
            "synthesis_strategy": result.synthesis_strategy.value,
            "quality_metrics": result.quality_metrics,
            "metadata": result.synthesis_metadata,
            "processing_time": result.processing_time,
            "source_count": len(result.source_responses)
        }
        
    except Exception as e:
        logger.error(f"❌ Synthesis helper error: {e}")
        return {
            "response": responses[0].get("content", "Synthesis error") if responses else "No responses to synthesize",
            "confidence": 0.5,
            "synthesis_strategy": "error_fallback",
            "error": str(e),
            "processing_time": 0.0,
            "source_count": len(responses)
        }