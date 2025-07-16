# backend/services/05-enhanced-intelligence/quality-service/app/improvement_suggester.py
"""
Improvement Suggester
AI-powered improvement recommendations
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

# Create router
improvement_router = APIRouter()

# Models
class ImprovementRequest(BaseModel):
    response_text: str = Field(..., min_length=1, max_length=50000)
    original_query: Optional[str] = Field(None)
    quality_scores: Optional[Dict[str, float]] = Field(None)
    target_quality: str = Field(default="good", description="poor, acceptable, good, excellent")
    focus_areas: Optional[List[str]] = Field(None)
    context: Optional[Dict[str, Any]] = Field(None)

class ImprovementResponse(BaseModel):
    improvement_id: str
    original_quality_level: str
    target_quality_level: str
    improvement_suggestions: List[Dict[str, Any]]
    rewritten_sections: Optional[Dict[str, str]]
    priority_improvements: List[str]
    estimated_impact: Dict[str, float]
    implementation_difficulty: str
    processing_time_seconds: float
    timestamp: str

class ImprovementSuggester:
    """AI-powered Improvement Suggestion Engine"""
    
    def __init__(self):
        self.improvement_strategies = {
            "content_enhancement": {
                "description": "Improve content depth and accuracy",
                "techniques": ["add_examples", "provide_details", "include_facts", "expand_explanations"],
                "difficulty": "medium"
            },
            "structure_optimization": {
                "description": "Enhance organization and flow",
                "techniques": ["add_headers", "reorganize_paragraphs", "improve_transitions", "create_lists"],
                "difficulty": "easy"
            },
            "language_refinement": {
                "description": "Improve language quality and clarity",
                "techniques": ["simplify_language", "vary_vocabulary", "improve_grammar", "enhance_readability"],
                "difficulty": "medium"
            },
            "engagement_boost": {
                "description": "Make content more engaging",
                "techniques": ["add_questions", "use_examples", "conversational_tone", "interactive_elements"],
                "difficulty": "easy"
            },
            "relevance_alignment": {
                "description": "Better align with user intent",
                "techniques": ["focus_on_query", "remove_tangents", "address_specific_needs", "contextual_relevance"],
                "difficulty": "hard"
            },
            "technical_polish": {
                "description": "Improve technical presentation",
                "techniques": ["format_properly", "check_length", "fix_punctuation", "consistent_style"],
                "difficulty": "easy"
            }
        }
        
        self.quality_targets = {
            "poor": 0.4,
            "acceptable": 0.6,
            "good": 0.8,
            "excellent": 0.95
        }
        
        self.improvement_templates = {
            "add_examples": "Consider adding a specific example: '{example_suggestion}'",
            "provide_details": "Expand this section with more detailed information about {topic}",
            "include_facts": "Support this claim with data or research findings",
            "expand_explanations": "Provide a more thorough explanation of {concept}",
            "add_headers": "Break this content into sections with clear headers",
            "reorganize_paragraphs": "Reorganize content for better logical flow",
            "improve_transitions": "Add transition sentences between paragraphs",
            "create_lists": "Convert this information into a bulleted or numbered list",
            "simplify_language": "Use simpler language: '{simplified_version}'",
            "vary_vocabulary": "Replace repeated words with synonyms",
            "improve_grammar": "Fix grammatical issues in: '{problematic_text}'",
            "enhance_readability": "Break long sentences into shorter ones",
            "add_questions": "Engage readers with questions like: '{suggested_question}'",
            "use_examples": "Illustrate this point with a real-world example",
            "conversational_tone": "Adopt a more conversational, approachable tone",
            "interactive_elements": "Add interactive elements to engage the reader",
            "focus_on_query": "Better address the original question: '{original_query}'",
            "remove_tangents": "Remove or minimize off-topic content",
            "address_specific_needs": "More directly address the user's specific needs",
            "contextual_relevance": "Ensure all content relates to the main topic",
            "format_properly": "Improve formatting and visual presentation",
            "check_length": "Adjust length - content is currently {current_status}",
            "fix_punctuation": "Correct punctuation and capitalization",
            "consistent_style": "Maintain consistent writing style throughout"
        }
    
    async def generate_improvements(self, request: ImprovementRequest) -> Dict[str, Any]:
        """Generate comprehensive improvement suggestions"""
        
        try:
            # Analyze current quality
            current_analysis = self._analyze_current_quality(
                request.response_text,
                request.quality_scores or {}
            )
            
            # Identify improvement opportunities
            opportunities = self._identify_opportunities(
                request.response_text,
                request.original_query,
                current_analysis,
                request.target_quality,
                request.focus_areas or []
            )
            
            # Generate specific suggestions
            suggestions = self._generate_specific_suggestions(
                request.response_text,
                opportunities,
                request.context or {}
            )
            
            # Prioritize improvements
            priority_improvements = self._prioritize_improvements(
                suggestions,
                request.target_quality,
                current_analysis
            )
            
            # Estimate impact
            impact_estimates = self._estimate_improvement_impact(
                suggestions,
                current_analysis
            )
            
            # Generate rewritten sections (if applicable)
            rewritten_sections = self._generate_rewritten_sections(
                request.response_text,
                suggestions[:3]  # Top 3 suggestions
            )
            
            return {
                "current_analysis": current_analysis,
                "opportunities": opportunities,
                "suggestions": suggestions,
                "priority_improvements": priority_improvements,
                "impact_estimates": impact_estimates,
                "rewritten_sections": rewritten_sections
            }
            
        except Exception as e:
            logger.error(f"❌ Improvement generation error: {e}")
            return self._emergency_suggestions(request.response_text)
    
    def _analyze_current_quality(self, response_text: str, quality_scores: Dict[str, float]) -> Dict[str, Any]:
        """Analyze current quality level and issues"""
        
        analysis = {
            "length": len(response_text),
            "word_count": len(response_text.split()),
            "paragraph_count": len([p for p in response_text.split('\n\n') if p.strip()]),
            "sentence_count": len([s for s in response_text.split('.') if s.strip()]),
            "issues": [],
            "strengths": []
        }
        
        # Analyze based on quality scores
        if quality_scores:
            for dimension, score in quality_scores.items():
                if score < 0.6:
                    analysis["issues"].append(f"Low {dimension.replace('_', ' ')} ({score:.2f})")
                elif score > 0.8:
                    analysis["strengths"].append(f"Strong {dimension.replace('_', ' ')} ({score:.2f})")
        
        # Basic text analysis
        words = response_text.split()
        
        # Length analysis
        if len(words) < 20:
            analysis["issues"].append("Response too short")
        elif len(words) > 500:
            analysis["issues"].append("Response may be too long")
        else:
            analysis["strengths"].append("Appropriate length")
        
        # Structure analysis
        if analysis["paragraph_count"] == 1 and len(words) > 100:
            analysis["issues"].append("Needs paragraph breaks")
        elif analysis["paragraph_count"] > 1:
            analysis["strengths"].append("Good paragraph structure")
        
        # Basic readability
        if analysis["sentence_count"] > 0:
            avg_words_per_sentence = len(words) / analysis["sentence_count"]
            if avg_words_per_sentence > 25:
                analysis["issues"].append("Sentences too long")
            elif 10 <= avg_words_per_sentence <= 20:
                analysis["strengths"].append("Good sentence length")
        
        return analysis
    
    def _identify_opportunities(self, response_text: str, original_query: Optional[str], 
                              current_analysis: Dict[str, Any], target_quality: str,
                              focus_areas: List[str]) -> List[Dict[str, Any]]:
        """Identify improvement opportunities"""
        
        opportunities = []
        target_score = self.quality_targets.get(target_quality, 0.8)
        
        # Strategy-based opportunities
        for strategy_name, strategy_config in self.improvement_strategies.items():
            if not focus_areas or strategy_name in focus_areas:
                opportunity = {
                    "strategy": strategy_name,
                    "description": strategy_config["description"],
                    "techniques": strategy_config["techniques"],
                    "difficulty": strategy_config["difficulty"],
                    "priority": self._calculate_opportunity_priority(
                        strategy_name, current_analysis, target_score
                    )
                }
                opportunities.append(opportunity)
        
        # Sort by priority
        opportunities.sort(key=lambda x: x["priority"], reverse=True)
        
        return opportunities
    
    def _calculate_opportunity_priority(self, strategy_name: str, 
                                      current_analysis: Dict[str, Any], 
                                      target_score: float) -> float:
        """Calculate priority for improvement opportunity"""
        
        priority = 0.5  # Base priority
        issues = current_analysis.get("issues", [])
        
        # Strategy-specific priority adjustments
        if strategy_name == "content_enhancement":
            if any("content" in issue.lower() for issue in issues):
                priority += 0.3
            if current_analysis.get("word_count", 0) < 50:
                priority += 0.2
        
        elif strategy_name == "structure_optimization":
            if any("structure" in issue.lower() or "paragraph" in issue.lower() for issue in issues):
                priority += 0.3
            if current_analysis.get("paragraph_count", 0) <= 1:
                priority += 0.2
        
        elif strategy_name == "language_refinement":
            if any("language" in issue.lower() or "sentence" in issue.lower() for issue in issues):
                priority += 0.3
        
        elif strategy_name == "technical_polish":
            if any("format" in issue.lower() or "length" in issue.lower() for issue in issues):
                priority += 0.4  # Easy fixes, high impact
        
        return min(1.0, priority)
    
    def _generate_specific_suggestions(self, response_text: str, 
                                     opportunities: List[Dict[str, Any]],
                                     context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specific improvement suggestions"""
        
        suggestions = []
        
        for opportunity in opportunities[:5]:  # Top 5 opportunities
            strategy = opportunity["strategy"]
            techniques = opportunity["techniques"]
            
            for technique in techniques[:2]:  # Top 2 techniques per strategy
                suggestion = self._create_specific_suggestion(
                    technique,
                    response_text,
                    context,
                    opportunity
                )
                if suggestion:
                    suggestions.append(suggestion)
        
        return suggestions
    
    def _create_specific_suggestion(self, technique: str, response_text: str,
                                  context: Dict[str, Any], opportunity: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a specific suggestion for a technique"""
        
        template = self.improvement_templates.get(technique, f"Apply {technique} technique")
        
        suggestion = {
            "technique": technique,
            "strategy": opportunity["strategy"],
            "difficulty": opportunity["difficulty"],
            "description": template,
            "specific_action": "",
            "expected_benefit": "",
            "implementation_notes": []
        }
        
        # Customize based on technique
        if technique == "add_examples":
            suggestion["specific_action"] = "Add a concrete example to illustrate the main point"
            suggestion["expected_benefit"] = "Improves understanding and engagement"
            suggestion["description"] = template.format(example_suggestion="[specific example based on content]")
        
        elif technique == "provide_details":
            suggestion["specific_action"] = "Expand key concepts with more detailed explanations"
            suggestion["expected_benefit"] = "Increases content depth and value"
            # Find potential topics to expand
            words = response_text.split()
            if len(words) > 10:
                key_terms = [w for w in words if len(w) > 6][:3]
                if key_terms:
                    suggestion["description"] = template.format(topic=key_terms[0])
        
        elif technique == "add_headers":
            if len(response_text.split('\n\n')) > 1:
                suggestion["specific_action"] = "Add clear section headers to organize content"
                suggestion["expected_benefit"] = "Improves readability and navigation"
            else:
                return None  # Not applicable for single paragraph
        
        elif technique == "simplify_language":
            # Find complex sentences
            sentences = [s.strip() for s in response_text.split('.') if s.strip()]
            long_sentences = [s for s in sentences if len(s.split()) > 20]
            if long_sentences:
                suggestion["specific_action"] = "Break down complex sentences into simpler ones"
                suggestion["expected_benefit"] = "Improves readability and comprehension"
                suggestion["description"] = template.format(simplified_version="[simplified version of complex sentence]")
            else:
                return None
        
        elif technique == "check_length":
            word_count = len(response_text.split())
            if word_count < 50:
                current_status = "too short"
                suggestion["specific_action"] = "Expand content with more information"
                suggestion["expected_benefit"] = "Provides more comprehensive coverage"
            elif word_count > 500:
                current_status = "too long"
                suggestion["specific_action"] = "Condense content to focus on key points"
                suggestion["expected_benefit"] = "Improves focus and readability"
            else:
                return None  # Length is appropriate
            
            suggestion["description"] = template.format(current_status=current_status)
        
        elif technique == "focus_on_query":
            original_query = context.get("original_query", "")
            if original_query:
                suggestion["specific_action"] = "Ensure response directly addresses the original question"
                suggestion["expected_benefit"] = "Improves relevance and user satisfaction"
                suggestion["description"] = template.format(original_query=original_query)
            else:
                return None
        
        else:
            # Generic suggestion
            suggestion["specific_action"] = f"Apply {technique.replace('_', ' ')} improvements"
            suggestion["expected_benefit"] = "Enhances overall quality"
        
        return suggestion
    
    def _prioritize_improvements(self, suggestions: List[Dict[str, Any]], 
                               target_quality: str, current_analysis: Dict[str, Any]) -> List[str]:
        """Prioritize improvements based on impact and difficulty"""
        
        # Score each suggestion
        scored_suggestions = []
        
        for suggestion in suggestions:
            score = 0.5  # Base score
            
            # Difficulty adjustment (easier = higher priority)
            if suggestion["difficulty"] == "easy":
                score += 0.3
            elif suggestion["difficulty"] == "medium":
                score += 0.1
            # hard stays at base
            
            # Issue-based priority
            issues = current_analysis.get("issues", [])
            strategy = suggestion["strategy"]
            
            if strategy == "technical_polish" and any("format" in issue.lower() for issue in issues):
                score += 0.4  # High impact, easy fix
            elif strategy == "structure_optimization" and any("paragraph" in issue.lower() for issue in issues):
                score += 0.3
            elif strategy == "content_enhancement" and any("short" in issue.lower() for issue in issues):
                score += 0.3
            
            scored_suggestions.append((suggestion["technique"], score))
        
        # Sort by score and return top techniques
        scored_suggestions.sort(key=lambda x: x[1], reverse=True)
        
        return [technique for technique, _ in scored_suggestions[:5]]
    
    def _estimate_improvement_impact(self, suggestions: List[Dict[str, Any]], 
                                   current_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Estimate the impact of implementing suggestions"""
        
        impact_estimates = {}
        
        # Base impacts for different improvement types
        base_impacts = {
            "content_enhancement": 0.15,
            "structure_optimization": 0.12,
            "language_refinement": 0.10,
            "engagement_boost": 0.08,
            "relevance_alignment": 0.20,
            "technical_polish": 0.05
        }
        
        strategy_impacts = {}
        
        for suggestion in suggestions:
            strategy = suggestion["strategy"]
            if strategy not in strategy_impacts:
                strategy_impacts[strategy] = 0
            
            base_impact = base_impacts.get(strategy, 0.05)
            
            # Adjust based on current issues
            issues = current_analysis.get("issues", [])
            if any(strategy.split("_")[0] in issue.lower() for issue in issues):
                base_impact *= 1.5  # Higher impact if addressing current issue
            
            strategy_impacts[strategy] = max(strategy_impacts[strategy], base_impact)
        
        # Calculate cumulative impact (with diminishing returns)
        total_impact = 0
        for strategy, impact in strategy_impacts.items():
            total_impact += impact * (0.8 ** len([s for s in strategy_impacts if s != strategy]))
        
        impact_estimates["overall_improvement"] = min(0.5, total_impact)  # Cap at 50% improvement
        impact_estimates["strategy_breakdown"] = strategy_impacts
        
        return impact_estimates
    
    def _generate_rewritten_sections(self, response_text: str, 
                                   top_suggestions: List[Dict[str, Any]]) -> Dict[str, str]:
        """Generate example rewritten sections"""
        
        rewritten_sections = {}
        
        for suggestion in top_suggestions:
            technique = suggestion["technique"]
            
            if technique == "add_headers" and len(response_text.split('\n\n')) > 1:
                # Generate example with headers
                paragraphs = [p.strip() for p in response_text.split('\n\n') if p.strip()]
                if len(paragraphs) >= 2:
                    rewritten = "## Main Point\n\n" + paragraphs[0]
                    if len(paragraphs) > 1:
                        rewritten += "\n\n## Additional Information\n\n" + paragraphs[1]
                    rewritten_sections["with_headers"] = rewritten
            
            elif technique == "simplify_language":
                # Find first long sentence and provide simplified version
                sentences = [s.strip() for s in response_text.split('.') if s.strip()]
                long_sentences = [s for s in sentences if len(s.split()) > 20]
                if long_sentences:
                    original_sentence = long_sentences[0]
                    # Simple simplification: break at conjunctions
                    simplified = original_sentence.replace(", and ", ". ").replace(", but ", ". However, ")
                    rewritten_sections["simplified_sentence"] = {
                        "original": original_sentence,
                        "simplified": simplified
                    }
            
            elif technique == "add_examples":
                # Add example placeholder
                first_paragraph = response_text.split('\n\n')[0] if '\n\n' in response_text else response_text[:200]
                with_example = first_paragraph + "\n\nFor example, [insert relevant example here]."
                rewritten_sections["with_example"] = with_example
        
        return rewritten_sections
    
    def _emergency_suggestions(self, response_text: str) -> Dict[str, Any]:
        """Emergency fallback suggestions"""
        
        basic_suggestions = [
            {
                "technique": "general_review",
                "strategy": "overall_improvement",
                "difficulty": "medium",
                "description": "Conduct comprehensive review and revision",
                "specific_action": "Review content for accuracy, clarity, and completeness",
                "expected_benefit": "General quality improvement"
            }
        ]
        
        return {
            "current_analysis": {"issues": ["Analysis unavailable"], "strengths": []},
            "opportunities": [{"strategy": "general_improvement", "priority": 0.5}],
            "suggestions": basic_suggestions,
            "priority_improvements": ["general_review"],
            "impact_estimates": {"overall_improvement": 0.2},
            "rewritten_sections": {}
        }

# Initialize suggester
improvement_suggester = ImprovementSuggester()

@improvement_router.post("/suggest", response_model=ImprovementResponse)
async def suggest_improvements(request: ImprovementRequest):
    """Generate comprehensive improvement suggestions"""
    
    start_time = time.time()
    
    try:
        # Generate improvement ID
        improvement_id = f"imp_{int(time.time())}_{hash(request.response_text) % 1000}"
        
        # Generate improvements
        improvements = await improvement_suggester.generate_improvements(request)
        
        # Determine quality levels
        current_quality = "acceptable"  # Default, could be derived from scores
        if request.quality_scores:
            avg_score = sum(request.quality_scores.values()) / len(request.quality_scores)
            if avg_score >= 0.9:
                current_quality = "excellent"
            elif avg_score >= 0.8:
                current_quality = "good"
            elif avg_score >= 0.6:
                current_quality = "acceptable"
            else:
                current_quality = "poor"
        
        # Determine implementation difficulty
        difficulties = [s.get("difficulty", "medium") for s in improvements.get("suggestions", [])]
        if all(d == "easy" for d in difficulties):
            impl_difficulty = "easy"
        elif any(d == "hard" for d in difficulties):
            impl_difficulty = "hard"
        else:
            impl_difficulty = "medium"
        
        processing_time = time.time() - start_time
        
        response = ImprovementResponse(
            improvement_id=improvement_id,
            original_quality_level=current_quality,
            target_quality_level=request.target_quality,
            improvement_suggestions=improvements.get("suggestions", []),
            rewritten_sections=improvements.get("rewritten_sections"),
            priority_improvements=improvements.get("priority_improvements", []),
            estimated_impact=improvements.get("impact_estimates", {}),
            implementation_difficulty=impl_difficulty,
            processing_time_seconds=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(
            f"💡 Improvements suggested: {len(improvements.get('suggestions', []))} suggestions "
            f"for {current_quality} → {request.target_quality}"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Improvement suggestion error: {e}")
        raise HTTPException(status_code=500, detail=f"Improvement suggestion failed: {str(e)}")

@improvement_router.get("/strategies")
async def get_improvement_strategies():
    """Get available improvement strategies"""
    
    return {
        "strategies": improvement_suggester.improvement_strategies,
        "quality_targets": improvement_suggester.quality_targets,
        "available_techniques": list(improvement_suggester.improvement_templates.keys()),
        "timestamp": datetime.now().isoformat()
    }

@improvement_router.post("/quick")
async def quick_improvement_tips(response_text: str, target_quality: str = "good"):
    """Get quick improvement tips"""
    
    try:
        request = ImprovementRequest(
            response_text=response_text,
            target_quality=target_quality,
            focus_areas=["technical_polish", "structure_optimization"]  # Quick wins
        )
        
        improvements = await improvement_suggester.generate_improvements(request)
        
        # Get top 3 quick wins
        quick_tips = []
        for suggestion in improvements.get("suggestions", [])[:3]:
            if suggestion.get("difficulty") in ["easy", "medium"]:
                quick_tips.append({
                    "tip": suggestion.get("description", ""),
                    "action": suggestion.get("specific_action", ""),
                    "benefit": suggestion.get("expected_benefit", ""),
                    "difficulty": suggestion.get("difficulty", "medium")
                })
        
        return {
            "quick_tips": quick_tips,
            "estimated_time_to_implement": "5-15 minutes",
            "expected_improvement": f"Upgrade from current level to {target_quality}",
            "priority_order": improvements.get("priority_improvements", [])[:3],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Quick improvement tips error: {e}")
        raise HTTPException(status_code=500, detail=str(e))