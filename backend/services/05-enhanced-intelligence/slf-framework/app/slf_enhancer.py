# backend/services/05-enhanced-intelligence/slf-framework/app/slf_enhancer.py
"""
SLF Enhancer Core
Systematic Language Framework for enhanced AI reasoning
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

# Create router
slf_router = APIRouter()

# Models
class SLFRequest(BaseModel):
    original_prompt: str = Field(..., min_length=1, max_length=10000)
    model_target: str = Field(default="gpt", description="gpt, claude, gemini")
    reasoning_type: str = Field(default="systematic", description="systematic, creative, analytical, strategic")
    enhancement_level: str = Field(default="moderate", description="minimal, moderate, comprehensive")
    context: Optional[Dict[str, Any]] = Field(None)
    enterprise_mode: bool = Field(default=False)

class SLFResponse(BaseModel):
    slf_id: str
    original_prompt: str
    enhanced_prompt: str
    reasoning_framework: str
    enhancement_techniques: List[str]
    model_optimizations: Dict[str, Any]
    expected_improvements: Dict[str, float]
    enterprise_compliance: Dict[str, bool]
    processing_time_seconds: float
    timestamp: str

class SLFEnhancer:
    """Systematic Language Framework Enhancement Engine"""
    
    def __init__(self):
        self.reasoning_frameworks = {
            "systematic": {
                "description": "Structured analytical reasoning",
                "template": "Let's approach this systematically:\n\n1. First, let me understand what you're asking: {query_analysis}\n\n2. Key considerations: {key_factors}\n\n3. Systematic analysis: {analysis_structure}\n\n4. Conclusion with reasoning: {conclusion_framework}",
                "techniques": ["structured_thinking", "step_by_step", "logical_progression", "evidence_based"],
                "improvement_areas": ["accuracy", "completeness", "logical_flow"]
            },
            "creative": {
                "description": "Enhanced creative and innovative thinking",
                "template": "Let's explore this creatively:\n\n🎨 Creative angle: {creative_perspective}\n\n💡 Innovative approaches: {innovation_factors}\n\n🌟 Unique insights: {unique_elements}\n\n✨ Creative synthesis: {creative_conclusion}",
                "techniques": ["divergent_thinking", "creative_synthesis", "innovative_perspective", "artistic_expression"],
                "improvement_areas": ["creativity", "originality", "engagement", "inspiration"]
            },
            "analytical": {
                "description": "Deep analytical and critical thinking",
                "template": "Let's analyze this thoroughly:\n\n📊 Data analysis: {data_perspective}\n\n🔍 Critical examination: {critical_factors}\n\n⚖️ Comparative analysis: {comparison_elements}\n\n📈 Analytical conclusion: {analytical_synthesis}",
                "techniques": ["critical_thinking", "data_analysis", "comparative_reasoning", "evidence_evaluation"],
                "improvement_areas": ["depth", "accuracy", "critical_thinking", "evidence_quality"]
            },
            "strategic": {
                "description": "Strategic and high-level thinking",
                "template": "Let's think strategically:\n\n🎯 Strategic objectives: {strategic_goals}\n\n🗺️ Landscape analysis: {environmental_factors}\n\n⚡ Strategic options: {strategic_alternatives}\n\n🏆 Strategic recommendation: {strategic_conclusion}",
                "techniques": ["strategic_thinking", "systems_perspective", "long_term_planning", "stakeholder_analysis"],
                "improvement_areas": ["strategic_value", "long_term_thinking", "stakeholder_awareness", "systems_thinking"]
            }
        }
        
        self.model_optimizations = {
            "gpt": {
                "prompt_style": "conversational_structured",
                "reasoning_emphasis": "step_by_step_explanation",
                "output_format": "structured_with_examples",
                "temperature_suggestion": 0.7,
                "max_tokens_suggestion": 2048,
                "techniques": ["clear_instructions", "examples", "structured_output", "contextual_hints"]
            },
            "claude": {
                "prompt_style": "detailed_analytical",
                "reasoning_emphasis": "comprehensive_analysis",
                "output_format": "detailed_structured",
                "context_utilization": "maximum",
                "techniques": ["detailed_context", "analytical_framework", "comprehensive_coverage", "reasoning_chains"]
            },
            "gemini": {
                "prompt_style": "factual_systematic",
                "reasoning_emphasis": "evidence_based",
                "output_format": "factual_structured",
                "search_integration": "enabled",
                "techniques": ["fact_checking", "source_integration", "systematic_verification", "multimodal_context"]
            }
        }
        
        self.enhancement_techniques = {
            "structured_thinking": "Organize thoughts in clear, logical structure",
            "step_by_step": "Break down complex problems into manageable steps",
            "logical_progression": "Ensure each point follows logically from the previous",
            "evidence_based": "Support claims with evidence and reasoning",
            "contextual_awareness": "Consider broader context and implications",
            "stakeholder_perspective": "Consider multiple viewpoints and stakeholders",
            "critical_evaluation": "Apply critical thinking to evaluate options",
            "synthesis": "Combine multiple elements into coherent conclusion"
        }
        
        self.enterprise_requirements = {
            "professional_tone": "Maintain professional, business-appropriate language",
            "fact_checking": "Ensure accuracy and verifiability of claims",
            "bias_awareness": "Acknowledge and address potential biases",
            "compliance_adherence": "Follow enterprise policies and guidelines",
            "stakeholder_consideration": "Consider impact on various stakeholders",
            "risk_awareness": "Identify and address potential risks",
            "actionable_output": "Provide clear, actionable recommendations"
        }
    
    async def enhance_prompt(self, request: SLFRequest) -> Dict[str, Any]:
        """Enhance prompt using SLF framework"""
        
        try:
            # Select reasoning framework
            framework = self.reasoning_frameworks.get(
                request.reasoning_type, 
                self.reasoning_frameworks["systematic"]
            )
            
            # Analyze original prompt
            prompt_analysis = self._analyze_prompt(request.original_prompt, request.context or {})
            
            # Generate enhanced prompt
            enhanced_prompt = self._generate_enhanced_prompt(
                request.original_prompt,
                framework,
                request.model_target,
                request.enhancement_level,
                prompt_analysis,
                request.enterprise_mode
            )
            
            # Apply model-specific optimizations
            model_optimizations = self._apply_model_optimizations(
                enhanced_prompt,
                request.model_target,
                framework
            )
            
            # Calculate expected improvements
            expected_improvements = self._calculate_expected_improvements(
                prompt_analysis,
                framework,
                request.enhancement_level
            )
            
            # Check enterprise compliance
            enterprise_compliance = self._check_enterprise_compliance(
                enhanced_prompt,
                request.enterprise_mode
            )
            
            return {
                "enhanced_prompt": enhanced_prompt,
                "framework": framework,
                "prompt_analysis": prompt_analysis,
                "model_optimizations": model_optimizations,
                "expected_improvements": expected_improvements,
                "enterprise_compliance": enterprise_compliance,
                "techniques_used": self._identify_techniques_used(framework, request.enhancement_level)
            }
            
        except Exception as e:
            logger.error(f"❌ SLF enhancement error: {e}")
            return self._emergency_enhancement(request.original_prompt)
    
    def _analyze_prompt(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the original prompt structure and intent"""
        
        analysis = {
            "prompt_type": "unknown",
            "complexity": "medium",
            "intent": "informational",
            "structure_quality": "basic",
            "improvement_opportunities": []
        }
        
        prompt_lower = prompt.lower()
        
        # Determine prompt type
        if any(word in prompt_lower for word in ["analyze", "compare", "evaluate"]):
            analysis["prompt_type"] = "analytical"
        elif any(word in prompt_lower for word in ["create", "write", "design", "generate"]):
            analysis["prompt_type"] = "creative"
        elif any(word in prompt_lower for word in ["plan", "strategy", "approach", "recommend"]):
            analysis["prompt_type"] = "strategic"
        elif any(word in prompt_lower for word in ["explain", "how", "what", "why"]):
            analysis["prompt_type"] = "explanatory"
        else:
            analysis["prompt_type"] = "general"
        
        # Assess complexity
        word_count = len(prompt.split())
        if word_count > 50:
            analysis["complexity"] = "high"
        elif word_count > 20:
            analysis["complexity"] = "medium" 
        else:
            analysis["complexity"] = "low"
        
        # Determine intent
        if prompt_lower.startswith(("how", "what", "why", "when", "where")):
            analysis["intent"] = "informational"
        elif any(word in prompt_lower for word in ["should", "recommend", "suggest"]):
            analysis["intent"] = "advisory"
        elif any(word in prompt_lower for word in ["create", "make", "build"]):
            analysis["intent"] = "creative"
        else:
            analysis["intent"] = "general"
        
        # Structure quality assessment
        has_clear_question = "?" in prompt
        has_context = len(prompt.split()) > 10
        has_specific_requirements = any(word in prompt_lower for word in ["specific", "detailed", "comprehensive"])
        
        structure_score = sum([has_clear_question, has_context, has_specific_requirements])
        if structure_score >= 2:
            analysis["structure_quality"] = "good"
        elif structure_score == 1:
            analysis["structure_quality"] = "basic"
        else:
            analysis["structure_quality"] = "poor"
        
        # Identify improvement opportunities
        if not has_clear_question and analysis["intent"] == "informational":
            analysis["improvement_opportunities"].append("clarify_question")
        
        if word_count < 10:
            analysis["improvement_opportunities"].append("add_context")
        
        if "please" not in prompt_lower and "help" not in prompt_lower:
            analysis["improvement_opportunities"].append("improve_tone")
        
        if not has_specific_requirements and analysis["complexity"] == "high":
            analysis["improvement_opportunities"].append("specify_requirements")
        
        return analysis
    
    def _generate_enhanced_prompt(self, original_prompt: str, framework: Dict[str, Any], 
                                 model_target: str, enhancement_level: str, 
                                 prompt_analysis: Dict[str, Any], enterprise_mode: bool) -> str:
        """Generate enhanced prompt using SLF framework"""
        
        # Base enhancement
        enhanced = original_prompt
        
        # Add framework-specific enhancement
        if enhancement_level in ["moderate", "comprehensive"]:
            framework_template = framework["template"]
            
            # Extract key components for template
            query_analysis = f"Understanding the core question: {original_prompt[:100]}..."
            key_factors = "relevant factors, constraints, and context"
            
            if framework["description"] == "Structured analytical reasoning":
                analysis_structure = "systematic breakdown and logical analysis"
                conclusion_framework = "evidence-based conclusion with clear reasoning"
                enhanced = framework_template.format(
                    query_analysis=query_analysis,
                    key_factors=key_factors,
                    analysis_structure=analysis_structure,
                    conclusion_framework=conclusion_framework
                )
            
            elif framework["description"] == "Enhanced creative and innovative thinking":
                creative_perspective = "innovative viewpoints and fresh angles"
                innovation_factors = "creative possibilities and novel approaches"
                unique_elements = "distinctive insights and original thinking"
                creative_conclusion = "creative synthesis with actionable outcomes"
                enhanced = framework_template.format(
                    creative_perspective=creative_perspective,
                    innovation_factors=innovation_factors,
                    unique_elements=unique_elements,
                    creative_conclusion=creative_conclusion
                )
        
        # Add model-specific optimizations
        model_config = self.model_optimizations.get(model_target, {})
        
        if model_target == "claude" and enhancement_level == "comprehensive":
            enhanced = f"Please provide a comprehensive analysis of the following:\n\n{enhanced}\n\nPlease structure your response with clear reasoning chains and detailed explanations."
        
        elif model_target == "gpt" and enhancement_level in ["moderate", "comprehensive"]:
            enhanced = f"Let's work through this step by step:\n\n{enhanced}\n\nPlease provide clear explanations and practical examples where appropriate."
        
        elif model_target == "gemini":
            enhanced = f"Please analyze this systematically with factual accuracy:\n\n{enhanced}\n\nEnsure all claims are well-supported and consider multiple perspectives."
        
        # Add enterprise requirements
        if enterprise_mode:
            enterprise_addition = "\n\nEnterprise requirements:\n- Maintain professional tone\n- Provide evidence-based analysis\n- Consider stakeholder impacts\n- Include actionable recommendations"
            enhanced += enterprise_addition
        
        # Add specific improvements based on analysis
        improvements = prompt_analysis.get("improvement_opportunities", [])
        
        if "clarify_question" in improvements:
            enhanced = f"Specific question: {enhanced}\n\nPlease address this question directly and comprehensively."
        
        if "add_context" in improvements:
            enhanced = f"Context and background: Please consider relevant context when addressing: {enhanced}"
        
        return enhanced
    
    def _apply_model_optimizations(self, enhanced_prompt: str, model_target: str, 
                                  framework: Dict[str, Any]) -> Dict[str, Any]:
        """Apply model-specific optimizations"""
        
        model_config = self.model_optimizations.get(model_target, {})
        
        optimizations = {
            "model": model_target,
            "recommended_parameters": {},
            "prompt_style": model_config.get("prompt_style", "structured"),
            "specific_techniques": model_config.get("techniques", []),
            "optimization_notes": []
        }
        
        # Model-specific parameter recommendations
        if model_target == "gpt":
            optimizations["recommended_parameters"] = {
                "temperature": model_config.get("temperature_suggestion", 0.7),
                "max_tokens": model_config.get("max_tokens_suggestion", 2048),
                "top_p": 0.95
            }
            optimizations["optimization_notes"].append("Use moderate temperature for balanced creativity and accuracy")
        
        elif model_target == "claude":
            optimizations["recommended_parameters"] = {
                "context_window": "utilize_maximum",
                "reasoning_depth": "comprehensive"
            }
            optimizations["optimization_notes"].append("Leverage Claude's strong analytical capabilities")
        
        elif model_target == "gemini":
            optimizations["recommended_parameters"] = {
                "search_integration": "enabled",
                "fact_checking": "strict"
            }
            optimizations["optimization_notes"].append("Utilize Gemini's search and fact-checking capabilities")
        
        return optimizations
    
    def _calculate_expected_improvements(self, prompt_analysis: Dict[str, Any], 
                                       framework: Dict[str, Any], 
                                       enhancement_level: str) -> Dict[str, float]:
        """Calculate expected improvements from SLF enhancement"""
        
        base_improvements = {
            "reasoning_quality": 0.2,
            "response_accuracy": 0.15,
            "logical_structure": 0.25,
            "completeness": 0.20,
            "professional_quality": 0.10
        }
        
        # Enhancement level multipliers
        level_multipliers = {
            "minimal": 0.5,
            "moderate": 1.0,
            "comprehensive": 1.5
        }
        
        multiplier = level_multipliers.get(enhancement_level, 1.0)
        
        # Framework-specific bonuses
        framework_bonuses = {
            "systematic": {"reasoning_quality": 0.1, "logical_structure": 0.15},
            "analytical": {"response_accuracy": 0.1, "completeness": 0.1},
            "creative": {"professional_quality": 0.05},
            "strategic": {"completeness": 0.1, "professional_quality": 0.1}
        }
        
        framework_type = framework.get("description", "").split()[0].lower()
        bonuses = framework_bonuses.get(framework_type, {})
        
        # Calculate final improvements
        improvements = {}
        for metric, base_value in base_improvements.items():
            improvement = base_value * multiplier
            improvement += bonuses.get(metric, 0)
            improvements[metric] = min(0.5, improvement)  # Cap at 50% improvement
        
        return improvements
    
    def _check_enterprise_compliance(self, enhanced_prompt: str, enterprise_mode: bool) -> Dict[str, bool]:
        """Check enterprise compliance requirements"""
        
        compliance = {}
        
        if not enterprise_mode:
            compliance["enterprise_mode"] = False
            return compliance
        
        prompt_lower = enhanced_prompt.lower()
        
        # Professional tone check
        compliance["professional_tone"] = not any(word in prompt_lower for word in ["casual", "informal", "slang"])
        
        # Fact-checking emphasis
        compliance["fact_checking_emphasis"] = any(phrase in prompt_lower for phrase in 
                                                  ["evidence", "facts", "verify", "accurate"])
        
        # Bias awareness
        compliance["bias_awareness"] = any(phrase in prompt_lower for phrase in 
                                         ["perspective", "viewpoint", "consider", "stakeholder"])
        
        # Actionable output
        compliance["actionable_output"] = any(phrase in prompt_lower for phrase in 
                                            ["recommendation", "action", "implement", "practical"])
        
        # Risk consideration
        compliance["risk_awareness"] = any(phrase in prompt_lower for phrase in 
                                         ["risk", "consideration", "impact", "consequence"])
        
        return compliance
    
    def _identify_techniques_used(self, framework: Dict[str, Any], enhancement_level: str) -> List[str]:
        """Identify which enhancement techniques were used"""
        
        techniques = framework.get("techniques", [])
        
        if enhancement_level == "comprehensive":
            return techniques
        elif enhancement_level == "moderate":
            return techniques[:3]  # First 3 techniques
        else:  # minimal
            return techniques[:1]  # First technique only
    
    def _emergency_enhancement(self, original_prompt: str) -> Dict[str, Any]:
        """Emergency fallback enhancement"""
        
        basic_enhancement = f"Please provide a thorough and well-structured response to: {original_prompt}"
        
        return {
            "enhanced_prompt": basic_enhancement,
            "framework": {"description": "Basic enhancement"},
            "prompt_analysis": {"prompt_type": "general"},
            "model_optimizations": {"model": "gpt", "recommended_parameters": {}},
            "expected_improvements": {"reasoning_quality": 0.1},
            "enterprise_compliance": {"emergency_mode": True},
            "techniques_used": ["basic_enhancement"]
        }

# Initialize SLF enhancer
slf_enhancer = SLFEnhancer()

@slf_router.post("/enhance", response_model=SLFResponse)
async def enhance_prompt(request: SLFRequest):
    """Enhance prompt using SLF framework"""
    
    start_time = time.time()
    
    try:
        # Generate SLF ID
        slf_id = f"slf_{int(time.time())}_{hash(request.original_prompt) % 1000}"
        
        # Perform enhancement
        enhancement = await slf_enhancer.enhance_prompt(request)
        
        processing_time = time.time() - start_time
        
        response = SLFResponse(
            slf_id=slf_id,
            original_prompt=request.original_prompt,
            enhanced_prompt=enhancement.get("enhanced_prompt", request.original_prompt),
            reasoning_framework=enhancement.get("framework", {}).get("description", "basic"),
            enhancement_techniques=enhancement.get("techniques_used", []),
            model_optimizations=enhancement.get("model_optimizations", {}),
            expected_improvements=enhancement.get("expected_improvements", {}),
            enterprise_compliance=enhancement.get("enterprise_compliance", {}),
            processing_time_seconds=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
        improvement_avg = sum(enhancement.get("expected_improvements", {}).values()) / max(1, len(enhancement.get("expected_improvements", {})))
        
        logger.info(
            f"🧠 SLF Enhancement: {request.reasoning_type} framework, "
            f"{improvement_avg:.1%} avg improvement expected"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"❌ SLF enhancement error: {e}")
        raise HTTPException(status_code=500, detail=f"SLF enhancement failed: {str(e)}")

@slf_router.get("/frameworks")
async def get_reasoning_frameworks():
    """Get available reasoning frameworks"""
    
    return {
        "frameworks": slf_enhancer.reasoning_frameworks,
        "model_optimizations": slf_enhancer.model_optimizations,
        "enhancement_techniques": slf_enhancer.enhancement_techniques,
        "enterprise_requirements": slf_enhancer.enterprise_requirements,
        "timestamp": datetime.now().isoformat()
    }

@slf_router.post("/quick")
async def quick_enhance(prompt: str, model: str = "gpt", reasoning_type: str = "systematic"):
    """Quick prompt enhancement"""
    
    try:
        request = SLFRequest(
            original_prompt=prompt,
            model_target=model,
            reasoning_type=reasoning_type,
            enhancement_level="moderate"
        )
        
        enhancement = await slf_enhancer.enhance_prompt(request)
        
        return {
            "original": prompt,
            "enhanced": enhancement.get("enhanced_prompt", prompt),
            "framework_used": enhancement.get("framework", {}).get("description", "basic"),
            "expected_improvement": f"{sum(enhancement.get('expected_improvements', {}).values()) * 100:.0f}%",
            "techniques": enhancement.get("techniques_used", []),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Quick enhancement error: {e}")
        raise HTTPException(status_code=500, detail=str(e))