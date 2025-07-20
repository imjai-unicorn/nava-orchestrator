# backend/services/03-external-ai/claude-client/app/reasoning_enhancer.py
"""
Claude Reasoning Enhancer
Advanced reasoning enhancement for Claude models
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import json

logger = logging.getLogger(__name__)

class ReasoningType(str, Enum):
    """Types of reasoning for enhancement"""
    ANALYTICAL = "analytical"          # Systematic analysis and breakdown
    LOGICAL = "logical"               # Step-by-step logical reasoning
    CREATIVE = "creative"             # Creative problem-solving
    CRITICAL = "critical"             # Critical thinking and evaluation
    CAUSAL = "causal"                # Cause-and-effect reasoning
    COMPARATIVE = "comparative"       # Comparison and contrast
    DEDUCTIVE = "deductive"          # Deductive reasoning
    INDUCTIVE = "inductive"          # Inductive reasoning
    ABDUCTIVE = "abductive"          # Best explanation reasoning
    SYSTEMATIC = "systematic"        # Systematic methodology

class ReasoningFramework(str, Enum):
    """Reasoning frameworks to apply"""
    FIRST_PRINCIPLES = "first_principles"    # Break down to fundamental truths
    SOCRATIC_METHOD = "socratic_method"      # Question-based exploration
    SYSTEMS_THINKING = "systems_thinking"    # Holistic system analysis
    DECISION_TREES = "decision_trees"        # Branching decision analysis
    SWOT_ANALYSIS = "swot_analysis"          # Strengths, Weaknesses, Opportunities, Threats
    ROOT_CAUSE = "root_cause"                # Root cause analysis
    PROS_CONS = "pros_cons"                  # Pros and cons analysis
    HYPOTHESIS_TESTING = "hypothesis_testing" # Scientific method approach
    DESIGN_THINKING = "design_thinking"       # Human-centered problem solving
    LEAN_THINKING = "lean_thinking"          # Eliminate waste, optimize value

@dataclass
class ReasoningEnhancementResult:
    """Result of reasoning enhancement"""
    original_prompt: str
    enhanced_prompt: str
    reasoning_type: ReasoningType
    frameworks_applied: List[ReasoningFramework]
    enhancement_score: float  # 0-1 scale
    reasoning_depth: str  # shallow, medium, deep
    estimated_response_improvement: Dict[str, float]
    guidance_added: List[str]

class ClaudeReasoningEnhancer:
    """Advanced reasoning enhancer for Claude models"""
    
    def __init__(self):
        self.reasoning_templates = self._load_reasoning_templates()
        self.framework_patterns = self._load_framework_patterns()
        logger.info("🧠 Claude Reasoning Enhancer initialized")
    
    def enhance_reasoning(self, 
                         prompt: str,
                         reasoning_type: ReasoningType,
                         target_frameworks: Optional[List[ReasoningFramework]] = None,
                         reasoning_depth: str = "medium",
                         domain_context: Optional[Dict[str, Any]] = None) -> ReasoningEnhancementResult:
        """
        Enhance a prompt with advanced reasoning capabilities
        
        Args:
            prompt: Original prompt to enhance
            reasoning_type: Type of reasoning to focus on
            target_frameworks: Specific frameworks to apply
            reasoning_depth: Desired depth (shallow, medium, deep)
            domain_context: Domain-specific context
            
        Returns:
            ReasoningEnhancementResult with enhanced prompt and metadata
        """
        logger.debug(f"Enhancing {reasoning_type} reasoning for prompt: {prompt[:100]}...")
        
        # Analyze prompt for reasoning requirements
        analysis = self._analyze_reasoning_requirements(prompt, reasoning_type)
        
        # Select appropriate frameworks
        frameworks = target_frameworks or self._select_reasoning_frameworks(
            reasoning_type, analysis, reasoning_depth
        )
        
        # Apply reasoning enhancements
        enhanced_prompt = self._apply_reasoning_enhancements(
            prompt, reasoning_type, frameworks, reasoning_depth, domain_context, analysis
        )
        
        # Calculate enhancement metrics
        enhancement_score = self._calculate_enhancement_score(prompt, enhanced_prompt, reasoning_type)
        estimated_improvement = self._estimate_response_improvement(enhanced_prompt, reasoning_type)
        guidance_added = self._extract_guidance_elements(enhanced_prompt, prompt)
        
        result = ReasoningEnhancementResult(
            original_prompt=prompt,
            enhanced_prompt=enhanced_prompt,
            reasoning_type=reasoning_type,
            frameworks_applied=frameworks,
            enhancement_score=enhancement_score,
            reasoning_depth=reasoning_depth,
            estimated_response_improvement=estimated_improvement,
            guidance_added=guidance_added
        )
        
        logger.info(f"✅ Reasoning enhanced with {len(frameworks)} frameworks, score: {enhancement_score:.2f}")
        
        return result
    
    def _analyze_reasoning_requirements(self, prompt: str, reasoning_type: ReasoningType) -> Dict[str, Any]:
        """Analyze what reasoning enhancements are needed"""
        analysis = {
            "complexity_level": self._assess_complexity(prompt),
            "has_clear_problem": self._has_clear_problem_statement(prompt),
            "requires_analysis": self._requires_analysis(prompt),
            "requires_evaluation": self._requires_evaluation(prompt),
            "requires_comparison": self._requires_comparison(prompt),
            "requires_causation": self._requires_causation_analysis(prompt),
            "has_constraints": self._has_constraints(prompt),
            "domain_specific": self._is_domain_specific(prompt),
            "multi_step_problem": self._is_multi_step_problem(prompt),
            "ambiguity_level": self._assess_ambiguity(prompt),
            "reasoning_gaps": self._identify_reasoning_gaps(prompt, reasoning_type)
        }
        
        return analysis
    
    def _select_reasoning_frameworks(self, 
                                   reasoning_type: ReasoningType,
                                   analysis: Dict[str, Any],
                                   depth: str) -> List[ReasoningFramework]:
        """Select appropriate reasoning frameworks"""
        frameworks = []
        
        # Base frameworks by reasoning type
        type_frameworks = {
            ReasoningType.ANALYTICAL: [ReasoningFramework.FIRST_PRINCIPLES, ReasoningFramework.SYSTEMS_THINKING],
            ReasoningType.LOGICAL: [ReasoningFramework.DECISION_TREES, ReasoningFramework.HYPOTHESIS_TESTING],
            ReasoningType.CREATIVE: [ReasoningFramework.DESIGN_THINKING, ReasoningFramework.FIRST_PRINCIPLES],
            ReasoningType.CRITICAL: [ReasoningFramework.SOCRATIC_METHOD, ReasoningFramework.PROS_CONS],
            ReasoningType.CAUSAL: [ReasoningFramework.ROOT_CAUSE, ReasoningFramework.SYSTEMS_THINKING],
            ReasoningType.COMPARATIVE: [ReasoningFramework.SWOT_ANALYSIS, ReasoningFramework.PROS_CONS],
            ReasoningType.SYSTEMATIC: [ReasoningFramework.SYSTEMS_THINKING, ReasoningFramework.LEAN_THINKING]
        }
        
        # Start with type-specific frameworks
        frameworks.extend(type_frameworks.get(reasoning_type, [ReasoningFramework.FIRST_PRINCIPLES]))
        
        # Add frameworks based on analysis
        if analysis["requires_evaluation"]:
            frameworks.append(ReasoningFramework.PROS_CONS)
        
        if analysis["requires_comparison"]:
            frameworks.append(ReasoningFramework.SWOT_ANALYSIS)
        
        if analysis["requires_causation"]:
            frameworks.append(ReasoningFramework.ROOT_CAUSE)
        
        if analysis["multi_step_problem"]:
            frameworks.append(ReasoningFramework.DECISION_TREES)
        
        if analysis["complexity_level"] == "high":
            frameworks.append(ReasoningFramework.SYSTEMS_THINKING)
        
        # Adjust for depth
        if depth == "shallow":
            frameworks = frameworks[:1]
        elif depth == "medium":
            frameworks = frameworks[:2]
        # deep uses all selected frameworks
        
        # Remove duplicates while preserving order
        seen = set()
        unique_frameworks = []
        for fw in frameworks:
            if fw not in seen:
                seen.add(fw)
                unique_frameworks.append(fw)
        
        return unique_frameworks[:3]  # Max 3 frameworks to avoid over-complexity
    
    def _apply_reasoning_enhancements(self, 
                                    prompt: str,
                                    reasoning_type: ReasoningType,
                                    frameworks: List[ReasoningFramework],
                                    depth: str,
                                    domain_context: Optional[Dict[str, Any]],
                                    analysis: Dict[str, Any]) -> str:
        """Apply reasoning enhancements to the prompt"""
        
        # Start with base reasoning enhancement
        enhanced = self._add_base_reasoning_structure(prompt, reasoning_type, depth)
        
        # Apply each framework
        for framework in frameworks:
            enhanced = self._apply_framework(enhanced, framework, reasoning_type, depth)
        
        # Add domain-specific reasoning if applicable
        if domain_context:
            enhanced = self._add_domain_reasoning(enhanced, domain_context, reasoning_type)
        
        # Add metacognitive elements
        enhanced = self._add_metacognitive_guidance(enhanced, reasoning_type, analysis)
        
        return enhanced
    
    def _add_base_reasoning_structure(self, prompt: str, reasoning_type: ReasoningType, depth: str) -> str:
        """Add base reasoning structure to prompt"""
        
        structures = {
            ReasoningType.ANALYTICAL: {
                "shallow": "Please analyze this systematically by breaking it down into key components.",
                "medium": """
Please provide a systematic analysis using this structure:
1. Break down the problem into key components
2. Analyze each component thoroughly
3. Identify relationships and patterns
4. Synthesize insights and conclusions""",
                "deep": """
Please conduct a comprehensive analytical examination:

Phase 1: Decomposition
- Break down into fundamental elements
- Identify all relevant factors and variables
- Map relationships and dependencies

Phase 2: Analysis
- Examine each element in detail
- Consider multiple perspectives and viewpoints
- Apply relevant analytical frameworks

Phase 3: Synthesis
- Integrate findings across all elements
- Identify patterns, trends, and insights
- Draw evidence-based conclusions"""
            },
            
            ReasoningType.CRITICAL: {
                "shallow": "Please think critically about this, questioning assumptions and evaluating evidence.",
                "medium": """
Please apply critical thinking using this approach:
1. Identify and question underlying assumptions
2. Evaluate the quality and reliability of evidence
3. Consider alternative perspectives and counterarguments
4. Assess the strength of logical connections
5. Draw balanced, evidence-based conclusions""",
                "deep": """
Please engage in deep critical analysis:

Critical Examination Framework:
1. Assumption Analysis
   - What assumptions are being made?
   - Are these assumptions valid and well-founded?
   - What happens if we challenge these assumptions?

2. Evidence Evaluation
   - What evidence supports different viewpoints?
   - How reliable and credible are the sources?
   - What evidence might be missing or overlooked?

3. Logical Analysis
   - Are the logical connections sound?
   - Are there any logical fallacies present?
   - Do the conclusions follow from the premises?

4. Alternative Perspectives
   - What other viewpoints exist?
   - How might different stakeholders see this differently?
   - What are the strongest counterarguments?

5. Meta-Analysis
   - What are the implications of different conclusions?
   - What additional questions arise from this analysis?
   - How confident can we be in our assessment?"""
            },
            
            ReasoningType.CREATIVE: {
                "shallow": "Please think creatively and explore innovative approaches to this challenge.",
                "medium": """
Please approach this creatively using divergent and convergent thinking:
1. Generate multiple creative possibilities (divergent thinking)
2. Explore unconventional approaches and perspectives
3. Build on and combine ideas in novel ways
4. Evaluate and refine the most promising concepts (convergent thinking)""",
                "deep": """
Please engage in comprehensive creative reasoning:

Creative Exploration Process:
1. Divergent Ideation
   - Generate numerous possibilities without initial judgment
   - Explore unconventional and "outside the box" approaches
   - Use analogies and metaphors from different domains
   - Challenge conventional wisdom and standard approaches

2. Perspective Shifting
   - How would different types of people approach this?
   - What if constraints were removed or changed?
   - How might this look from completely different contexts?

3. Creative Synthesis
   - Combine disparate ideas in novel ways
   - Look for unexpected connections and relationships
   - Build on ideas to create new possibilities

4. Creative Evaluation
   - Which ideas have the most potential?
   - How could promising concepts be developed further?
   - What creative solutions best address the core challenge?"""
            }
        }
        
        structure = structures.get(reasoning_type, {}).get(depth, "Please reason through this step by step.")
        
        return f"{structure}\n\nOriginal request: {prompt}"
    
    def _apply_framework(self, 
                        prompt: str, 
                        framework: ReasoningFramework,
                        reasoning_type: ReasoningType,
                        depth: str) -> str:
        """Apply specific reasoning framework"""
        
        framework_templates = {
            ReasoningFramework.FIRST_PRINCIPLES: """
Additionally, please apply first principles thinking:
- Break this down to the most fundamental truths and basic elements
- Question each assumption until you reach bedrock facts
- Rebuild understanding from these fundamental components
- Avoid reasoning by analogy or convention""",
            
            ReasoningFramework.SOCRATIC_METHOD: """
Please explore this using the Socratic method:
- What key questions need to be asked?
- What do we think we know, and how do we know it?
- What assumptions are we making?
- What questions lead to deeper understanding?
- How do our answers reveal new questions?""",
            
            ReasoningFramework.SYSTEMS_THINKING: """
Please apply systems thinking:
- How do the different elements interact and influence each other?
- What are the feedback loops and interconnections?
- How might changes in one area affect the whole system?
- What are the leverage points for maximum impact?
- How does this fit within larger systems and contexts?""",
            
            ReasoningFramework.ROOT_CAUSE: """
Please conduct root cause analysis:
- What is the surface-level problem or symptom?
- What factors contribute to this problem?
- For each factor, ask "why does this happen?" repeatedly
- Trace back through the causal chain to fundamental causes
- Distinguish between immediate causes, contributing factors, and root causes""",
            
            ReasoningFramework.PROS_CONS: """
Please conduct a thorough pros and cons analysis:
- What are the strongest arguments in favor?
- What are the most significant drawbacks or concerns?
- How do the pros and cons compare in importance and likelihood?
- Are there any overlooked advantages or disadvantages?
- What does the balance suggest about the best course of action?""",
            
            ReasoningFramework.DECISION_TREES: """
Please map out the decision structure:
- What are the key decision points?
- What options are available at each decision point?
- What are the likely outcomes of each path?
- How do different choices lead to different end states?
- Which pathways offer the best risk/reward profiles?""",
            
            ReasoningFramework.SWOT_ANALYSIS: """
Please conduct a SWOT analysis:
- Strengths: What internal advantages and positive factors exist?
- Weaknesses: What internal limitations or areas for improvement exist?
- Opportunities: What external opportunities can be leveraged?
- Threats: What external challenges or risks need to be addressed?
- How can strengths be leveraged and weaknesses addressed?""",
            
            ReasoningFramework.HYPOTHESIS_TESTING: """
Please apply scientific reasoning:
- What hypotheses can be formed about this situation?
- What evidence would support or refute each hypothesis?
- How can these hypotheses be tested or validated?
- What are the implications if each hypothesis is correct?
- Based on available evidence, which hypotheses are most likely?"""
        }
        
        framework_text = framework_templates.get(framework, "")
        if framework_text:
            return f"{prompt}\n{framework_text}"
        
        return prompt
    
    def _add_domain_reasoning(self, 
                            prompt: str, 
                            domain_context: Dict[str, Any],
                            reasoning_type: ReasoningType) -> str:
        """Add domain-specific reasoning guidance"""
        
        domain = domain_context.get("domain", "")
        expertise_level = domain_context.get("expertise_level", "intermediate")
        
        domain_guidance = {
            "business": "Consider business implications, stakeholder impacts, resource constraints, and ROI.",
            "technical": "Focus on technical feasibility, scalability, maintainability, and best practices.",
            "scientific": "Apply scientific method, consider experimental design, and evaluate evidence quality.",
            "legal": "Consider legal precedents, regulatory compliance, and risk mitigation.",
            "medical": "Focus on patient safety, evidence-based practices, and clinical protocols.",
            "educational": "Consider learning objectives, pedagogical approaches, and student outcomes.",
            "creative": "Explore aesthetic principles, cultural context, and innovative expression."
        }
        
        if domain in domain_guidance:
            domain_text = f"\nDomain-specific considerations ({domain}): {domain_guidance[domain]}"
            prompt += domain_text
        
        return prompt
    
    def _add_metacognitive_guidance(self, 
                                  prompt: str,
                                  reasoning_type: ReasoningType,
                                  analysis: Dict[str, Any]) -> str:
        """Add metacognitive reasoning guidance"""
        
        metacognitive_elements = []
        
        # Add confidence assessment
        metacognitive_elements.append("Please assess your confidence in your reasoning and conclusions.")
        
        # Add limitation awareness
        if analysis["ambiguity_level"] == "high":
            metacognitive_elements.append("Please acknowledge areas of uncertainty and ambiguity.")
        
        # Add alternative consideration
        metacognitive_elements.append("Please consider what alternative conclusions might be possible.")
        
        # Add reasoning quality check
        metacognitive_elements.append("Please reflect on the quality and completeness of your reasoning process.")
        
        if metacognitive_elements:
            metacognitive_text = "\n\nMetacognitive guidance:\n" + "\n".join(f"- {element}" for element in metacognitive_elements)
            prompt += metacognitive_text
        
        return prompt
    
    def _assess_complexity(self, prompt: str) -> str:
        """Assess the complexity level of the prompt"""
        complexity_indicators = {
            "high": ["complex", "complicated", "multiple", "various", "comprehensive", "detailed"],
            "medium": ["analyze", "compare", "evaluate", "consider", "examine"],
            "low": ["what", "how", "when", "where", "simple", "basic"]
        }
        
        scores = {"high": 0, "medium": 0, "low": 0}
        prompt_lower = prompt.lower()
        
        for level, indicators in complexity_indicators.items():
            scores[level] = sum(1 for indicator in indicators if indicator in prompt_lower)
        
        return max(scores.keys(), key=lambda x: scores[x])
    
    def _has_clear_problem_statement(self, prompt: str) -> bool:
        """Check if prompt has a clear problem statement"""
        problem_indicators = ["problem", "issue", "challenge", "difficulty", "question", "how to"]
        return any(indicator in prompt.lower() for indicator in problem_indicators)
    
    def _requires_analysis(self, prompt: str) -> bool:
        """Check if prompt requires analytical reasoning"""
        analysis_indicators = ["analyze", "examine", "investigate", "study", "research", "explore"]
        return any(indicator in prompt.lower() for indicator in analysis_indicators)
    
    def _requires_evaluation(self, prompt: str) -> bool:
        """Check if prompt requires evaluative reasoning"""
        evaluation_indicators = ["evaluate", "assess", "judge", "critique", "review", "rate", "compare"]
        return any(indicator in prompt.lower() for indicator in evaluation_indicators)
    
    def _requires_comparison(self, prompt: str) -> bool:
        """Check if prompt requires comparative reasoning"""
        comparison_indicators = ["compare", "contrast", "versus", "vs", "difference", "similarity", "between"]
        return any(indicator in prompt.lower() for indicator in comparison_indicators)
    
    def _requires_causation_analysis(self, prompt: str) -> bool:
        """Check if prompt requires causal reasoning"""
        causation_indicators = ["why", "cause", "reason", "because", "due to", "result", "effect", "impact"]
        return any(indicator in prompt.lower() for indicator in causation_indicators)
    
    def _has_constraints(self, prompt: str) -> bool:
        """Check if prompt mentions constraints"""
        constraint_indicators = ["must", "should", "cannot", "limit", "restrict", "within", "budget", "time"]
        return any(indicator in prompt.lower() for indicator in constraint_indicators)
    
    def _is_domain_specific(self, prompt: str) -> bool:
        """Check if prompt is domain-specific"""
        # Simple heuristic based on technical terms
        technical_indicators = ["technical", "business", "medical", "legal", "scientific", "engineering"]
        return any(indicator in prompt.lower() for indicator in technical_indicators)
    
    def _is_multi_step_problem(self, prompt: str) -> bool:
        """Check if prompt requires multi-step reasoning"""
        multi_step_indicators = ["steps", "process", "procedure", "method", "approach", "strategy", "plan"]
        return any(indicator in prompt.lower() for indicator in multi_step_indicators)
    
    def _assess_ambiguity(self, prompt: str) -> str:
        """Assess the ambiguity level of the prompt"""
        # Simple heuristic based on specificity
        specific_indicators = ["specific", "exactly", "precisely", "