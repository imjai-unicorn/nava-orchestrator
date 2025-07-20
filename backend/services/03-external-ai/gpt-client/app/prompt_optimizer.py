# backend/services/03-external-ai/gpt-client/app/prompt_optimizer.py
"""
GPT Prompt Optimizer
Advanced prompt optimization for GPT models
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import json

logger = logging.getLogger(__name__)

class PromptType(str, Enum):
    """Types of prompts for optimization"""
    CONVERSATION = "conversation"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    SUMMARIZATION = "summarization"
    INSTRUCTION = "instruction"
    ROLEPLAY = "roleplay"
    REASONING = "reasoning"

class OptimizationStrategy(str, Enum):
    """Prompt optimization strategies"""
    CLARITY = "clarity"              # Improve clarity and specificity
    CONTEXT = "context"              # Add relevant context
    STRUCTURE = "structure"          # Improve prompt structure
    EXAMPLES = "examples"            # Add examples and demonstrations
    CONSTRAINTS = "constraints"      # Add constraints and guidelines
    PERSONA = "persona"              # Define AI persona/role
    CHAIN_OF_THOUGHT = "chain_of_thought"  # Enable step-by-step reasoning
    FEW_SHOT = "few_shot"           # Add few-shot examples

@dataclass
class PromptOptimizationResult:
    """Result of prompt optimization"""
    original_prompt: str
    optimized_prompt: str
    strategies_applied: List[OptimizationStrategy]
    improvement_score: float  # 0-1 scale
    reasoning: str
    estimated_token_increase: int
    performance_prediction: Dict[str, float]

class GPTPromptOptimizer:
    """Advanced prompt optimizer for GPT models"""
    
    def __init__(self):
        self.optimization_patterns = self._load_optimization_patterns()
        self.performance_metrics = {}
        logger.info("🎯 GPT Prompt Optimizer initialized")
    
    def optimize_prompt(self, 
                       prompt: str, 
                       prompt_type: PromptType,
                       user_context: Optional[Dict[str, Any]] = None,
                       performance_target: Optional[Dict[str, float]] = None) -> PromptOptimizationResult:
        """
        Optimize a prompt for better GPT performance
        
        Args:
            prompt: Original prompt to optimize
            prompt_type: Type of prompt for targeted optimization
            user_context: Additional context about user/task
            performance_target: Target performance metrics
            
        Returns:
            PromptOptimizationResult with optimized prompt and metadata
        """
        logger.debug(f"Optimizing {prompt_type} prompt: {prompt[:100]}...")
        
        # Analyze current prompt
        analysis = self._analyze_prompt(prompt, prompt_type)
        
        # Determine optimization strategies
        strategies = self._select_optimization_strategies(analysis, prompt_type, performance_target)
        
        # Apply optimizations
        optimized_prompt = prompt
        applied_strategies = []
        
        for strategy in strategies:
            optimized_prompt, applied = self._apply_optimization_strategy(
                optimized_prompt, strategy, prompt_type, user_context, analysis
            )
            if applied:
                applied_strategies.append(strategy)
        
        # Calculate improvement metrics
        improvement_score = self._calculate_improvement_score(prompt, optimized_prompt, analysis)
        performance_prediction = self._predict_performance(optimized_prompt, prompt_type)
        token_increase = self._estimate_token_increase(prompt, optimized_prompt)
        
        result = PromptOptimizationResult(
            original_prompt=prompt,
            optimized_prompt=optimized_prompt,
            strategies_applied=applied_strategies,
            improvement_score=improvement_score,
            reasoning=self._generate_optimization_reasoning(applied_strategies, analysis),
            estimated_token_increase=token_increase,
            performance_prediction=performance_prediction
        )
        
        logger.info(f"✅ Prompt optimized with {len(applied_strategies)} strategies, improvement: {improvement_score:.2f}")
        
        return result
    
    def _analyze_prompt(self, prompt: str, prompt_type: PromptType) -> Dict[str, Any]:
        """Analyze prompt characteristics"""
        analysis = {
            "length": len(prompt),
            "token_estimate": len(prompt.split()) * 1.3,  # Rough token estimate
            "has_examples": "example" in prompt.lower() or "for instance" in prompt.lower(),
            "has_constraints": any(word in prompt.lower() for word in ["must", "should", "don't", "avoid"]),
            "has_role_definition": any(word in prompt.lower() for word in ["you are", "act as", "role"]),
            "has_step_by_step": any(phrase in prompt.lower() for phrase in ["step by step", "first", "then", "finally"]),
            "specificity_score": self._calculate_specificity_score(prompt),
            "clarity_score": self._calculate_clarity_score(prompt),
            "structure_score": self._calculate_structure_score(prompt),
            "question_count": prompt.count("?"),
            "instruction_count": len([s for s in prompt.split(".") if any(word in s.lower() for word in ["please", "can you", "provide", "explain"])]),
        }
        
        return analysis
    
    def _select_optimization_strategies(self, 
                                     analysis: Dict[str, Any], 
                                     prompt_type: PromptType,
                                     performance_target: Optional[Dict[str, float]]) -> List[OptimizationStrategy]:
        """Select appropriate optimization strategies"""
        strategies = []
        
        # Based on prompt analysis
        if analysis["clarity_score"] < 0.7:
            strategies.append(OptimizationStrategy.CLARITY)
        
        if analysis["structure_score"] < 0.6:
            strategies.append(OptimizationStrategy.STRUCTURE)
        
        if not analysis["has_examples"] and prompt_type in [PromptType.INSTRUCTION, PromptType.TECHNICAL]:
            strategies.append(OptimizationStrategy.EXAMPLES)
        
        if not analysis["has_role_definition"] and prompt_type in [PromptType.ROLEPLAY, PromptType.CREATIVE]:
            strategies.append(OptimizationStrategy.PERSONA)
        
        if not analysis["has_step_by_step"] and prompt_type in [PromptType.REASONING, PromptType.ANALYSIS]:
            strategies.append(OptimizationStrategy.CHAIN_OF_THOUGHT)
        
        # Based on prompt type
        type_specific_strategies = {
            PromptType.CONVERSATION: [OptimizationStrategy.PERSONA, OptimizationStrategy.CONTEXT],
            PromptType.ANALYSIS: [OptimizationStrategy.CHAIN_OF_THOUGHT, OptimizationStrategy.STRUCTURE],
            PromptType.CREATIVE: [OptimizationStrategy.PERSONA, OptimizationStrategy.EXAMPLES],
            PromptType.TECHNICAL: [OptimizationStrategy.EXAMPLES, OptimizationStrategy.CONSTRAINTS],
            PromptType.SUMMARIZATION: [OptimizationStrategy.CONSTRAINTS, OptimizationStrategy.STRUCTURE],
            PromptType.INSTRUCTION: [OptimizationStrategy.EXAMPLES, OptimizationStrategy.CHAIN_OF_THOUGHT],
            PromptType.REASONING: [OptimizationStrategy.CHAIN_OF_THOUGHT, OptimizationStrategy.STRUCTURE]
        }
        
        for strategy in type_specific_strategies.get(prompt_type, []):
            if strategy not in strategies:
                strategies.append(strategy)
        
        return strategies[:4]  # Limit to 4 strategies to avoid over-optimization
    
    def _apply_optimization_strategy(self, 
                                   prompt: str, 
                                   strategy: OptimizationStrategy,
                                   prompt_type: PromptType,
                                   user_context: Optional[Dict[str, Any]],
                                   analysis: Dict[str, Any]) -> Tuple[str, bool]:
        """Apply specific optimization strategy"""
        
        try:
            if strategy == OptimizationStrategy.CLARITY:
                return self._apply_clarity_optimization(prompt, analysis), True
            
            elif strategy == OptimizationStrategy.STRUCTURE:
                return self._apply_structure_optimization(prompt, prompt_type), True
            
            elif strategy == OptimizationStrategy.EXAMPLES:
                return self._apply_examples_optimization(prompt, prompt_type), True
            
            elif strategy == OptimizationStrategy.PERSONA:
                return self._apply_persona_optimization(prompt, prompt_type), True
            
            elif strategy == OptimizationStrategy.CHAIN_OF_THOUGHT:
                return self._apply_chain_of_thought_optimization(prompt), True
            
            elif strategy == OptimizationStrategy.CONSTRAINTS:
                return self._apply_constraints_optimization(prompt, prompt_type), True
            
            elif strategy == OptimizationStrategy.CONTEXT:
                return self._apply_context_optimization(prompt, user_context), True
            
            elif strategy == OptimizationStrategy.FEW_SHOT:
                return self._apply_few_shot_optimization(prompt, prompt_type), True
            
            else:
                return prompt, False
                
        except Exception as e:
            logger.error(f"❌ Failed to apply {strategy} optimization: {e}")
            return prompt, False
    
    def _apply_clarity_optimization(self, prompt: str, analysis: Dict[str, Any]) -> str:
        """Improve prompt clarity and specificity"""
        optimized = prompt
        
        # Add clarity instructions
        if analysis["specificity_score"] < 0.6:
            clarity_instruction = "\n\nPlease provide a specific, detailed, and well-structured response."
            optimized += clarity_instruction
        
        # Replace vague terms with more specific ones
        vague_replacements = {
            "things": "specific items/concepts",
            "stuff": "relevant information",
            "good": "effective and well-designed",
            "bad": "problematic or ineffective",
            "nice": "well-executed and appropriate"
        }
        
        for vague, specific in vague_replacements.items():
            optimized = optimized.replace(vague, specific)
        
        return optimized
    
    def _apply_structure_optimization(self, prompt: str, prompt_type: PromptType) -> str:
        """Improve prompt structure and organization"""
        
        # Add structure based on prompt type
        if prompt_type == PromptType.ANALYSIS:
            structure_template = """
Please analyze the following systematically:

{original_prompt}

Structure your response as follows:
1. Key Observations
2. Analysis and Insights  
3. Implications and Conclusions
"""
        elif prompt_type == PromptType.INSTRUCTION:
            structure_template = """
Task: {original_prompt}

Please provide:
1. Clear step-by-step instructions
2. Expected outcomes for each step
3. Potential challenges and solutions
"""
        else:
            # Generic structure improvement
            structure_template = """
Request: {original_prompt}

Please provide a well-organized response with clear sections and logical flow.
"""
        
        return structure_template.format(original_prompt=prompt.strip())
    
    def _apply_examples_optimization(self, prompt: str, prompt_type: PromptType) -> str:
        """Add relevant examples to the prompt"""
        
        examples_by_type = {
            PromptType.TECHNICAL: """
For example, when explaining technical concepts, include:
- Concrete code examples or technical specifications
- Real-world use cases and applications
- Step-by-step implementation details
""",
            PromptType.CREATIVE: """
For example, when creating content:
- Use vivid, descriptive language
- Include specific details and examples
- Draw from relevant cultural or historical references
""",
            PromptType.INSTRUCTION: """
Example format:
Step 1: [Action] - [Expected Outcome]
Step 2: [Action] - [Expected Outcome]
Step 3: [Action] - [Expected Outcome]
"""
        }
        
        example_text = examples_by_type.get(prompt_type, "")
        if example_text:
            return f"{prompt}\n\n{example_text}"
        
        return prompt
    
    def _apply_persona_optimization(self, prompt: str, prompt_type: PromptType) -> str:
        """Add appropriate persona/role definition"""
        
        personas_by_type = {
            PromptType.TECHNICAL: "You are an experienced technical expert with deep knowledge in your field.",
            PromptType.CREATIVE: "You are a creative professional with expertise in innovative thinking and artistic expression.",
            PromptType.ANALYSIS: "You are an analytical expert skilled at breaking down complex problems and identifying key insights.",
            PromptType.CONVERSATION: "You are a knowledgeable and helpful assistant, focused on providing clear and useful information.",
            PromptType.INSTRUCTION: "You are a patient and thorough instructor, skilled at breaking down complex tasks into manageable steps."
        }
        
        persona = personas_by_type.get(prompt_type, "You are a knowledgeable assistant.")
        return f"{persona}\n\n{prompt}"
    
    def _apply_chain_of_thought_optimization(self, prompt: str) -> str:
        """Add chain-of-thought reasoning instruction"""
        
        cot_instruction = """
Please think through this step-by-step:
1. First, analyze the key components of the request
2. Then, work through your reasoning process
3. Finally, provide your conclusion or response

"""
        
        return f"{cot_instruction}{prompt}"
    
    def _apply_constraints_optimization(self, prompt: str, prompt_type: PromptType) -> str:
        """Add appropriate constraints and guidelines"""
        
        constraints_by_type = {
            PromptType.TECHNICAL: "\n\nConstraints: Ensure technical accuracy, provide working examples, and explain any assumptions.",
            PromptType.CREATIVE: "\n\nConstraints: Be original and engaging while maintaining appropriate tone and content.",
            PromptType.SUMMARIZATION: "\n\nConstraints: Be concise but comprehensive, maintain key information, and use clear structure.",
        }
        
        constraints = constraints_by_type.get(prompt_type, "")
        return f"{prompt}{constraints}"
    
    def _apply_context_optimization(self, prompt: str, user_context: Optional[Dict[str, Any]]) -> str:
        """Add relevant context information"""
        
        if not user_context:
            return prompt
        
        context_parts = []
        
        if user_context.get("user_role"):
            context_parts.append(f"User role: {user_context['user_role']}")
        
        if user_context.get("domain"):
            context_parts.append(f"Domain: {user_context['domain']}")
        
        if user_context.get("experience_level"):
            context_parts.append(f"Experience level: {user_context['experience_level']}")
        
        if context_parts:
            context_text = f"Context: {', '.join(context_parts)}\n\n"
            return f"{context_text}{prompt}"
        
        return prompt
    
    def _apply_few_shot_optimization(self, prompt: str, prompt_type: PromptType) -> str:
        """Add few-shot examples"""
        
        # This would typically include domain-specific examples
        # For now, add a general few-shot structure
        few_shot_template = f"""
Here are examples of the type of response expected:

Example 1: [Relevant example would go here]
Example 2: [Another relevant example would go here]

Now, please respond to: {prompt}
"""
        
        return few_shot_template
    
    def _calculate_specificity_score(self, prompt: str) -> float:
        """Calculate how specific the prompt is"""
        specific_indicators = ["specific", "exactly", "precisely", "particular", "detailed"]
        vague_indicators = ["some", "things", "stuff", "good", "bad", "nice"]
        
        specific_count = sum(1 for indicator in specific_indicators if indicator in prompt.lower())
        vague_count = sum(1 for indicator in vague_indicators if indicator in prompt.lower())
        
        total_words = len(prompt.split())
        if total_words == 0:
            return 0.0
        
        specificity = (specific_count - vague_count) / total_words * 10
        return max(0.0, min(1.0, specificity + 0.5))  # Normalize to 0-1 with baseline
    
    def _calculate_clarity_score(self, prompt: str) -> float:
        """Calculate prompt clarity score"""
        clarity_factors = {
            "sentence_length": 0.3,  # Shorter sentences are clearer
            "question_clarity": 0.3,  # Clear questions
            "instruction_clarity": 0.4  # Clear instructions
        }
        
        # Average sentence length
        sentences = prompt.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        sentence_score = max(0, 1 - (avg_sentence_length - 15) / 15)  # Optimal around 15 words
        
        # Question clarity (presence of question words)
        question_words = ["what", "how", "why", "when", "where", "which", "who"]
        question_score = min(1.0, sum(1 for word in question_words if word in prompt.lower()) / 3)
        
        # Instruction clarity (presence of action words)
        action_words = ["explain", "describe", "analyze", "create", "provide", "list", "compare"]
        instruction_score = min(1.0, sum(1 for word in action_words if word in prompt.lower()) / 2)
        
        total_score = (
            sentence_score * clarity_factors["sentence_length"] +
            question_score * clarity_factors["question_clarity"] +
            instruction_score * clarity_factors["instruction_clarity"]
        )
        
        return total_score
    
    def _calculate_structure_score(self, prompt: str) -> float:
        """Calculate prompt structure score"""
        structure_indicators = {
            "has_clear_sections": any(indicator in prompt for indicator in ["1.", "2.", "•", "-", "Step"]),
            "has_clear_request": any(word in prompt.lower() for word in ["please", "can you", "provide"]),
            "logical_flow": len(prompt.split('\n')) > 1,  # Multi-line suggests structure
        }
        
        score = sum(structure_indicators.values()) / len(structure_indicators)
        return score
    
    def _calculate_improvement_score(self, original: str, optimized: str, analysis: Dict[str, Any]) -> float:
        """Calculate expected improvement from optimization"""
        
        # Length improvement (but not too much)
        length_ratio = len(optimized) / len(original) if len(original) > 0 else 1
        length_score = 1.0 if 1.2 <= length_ratio <= 2.0 else 0.5
        
        # Structure improvement
        original_structure = self._calculate_structure_score(original)
        optimized_structure = self._calculate_structure_score(optimized)
        structure_improvement = max(0, optimized_structure - original_structure)
        
        # Clarity improvement
        original_clarity = self._calculate_clarity_score(original)
        optimized_clarity = self._calculate_clarity_score(optimized)
        clarity_improvement = max(0, optimized_clarity - original_clarity)
        
        # Combined score
        improvement_score = (
            length_score * 0.2 +
            structure_improvement * 0.4 +
            clarity_improvement * 0.4
        )
        
        return min(1.0, improvement_score)
    
    def _predict_performance(self, prompt: str, prompt_type: PromptType) -> Dict[str, float]:
        """Predict performance metrics for optimized prompt"""
        
        # This would typically be based on historical data
        # For now, provide estimates based on prompt characteristics
        
        analysis = self._analyze_prompt(prompt, prompt_type)
        
        base_performance = {
            "accuracy": 0.75,
            "relevance": 0.80,
            "completeness": 0.70,
            "clarity": 0.75
        }
        
        # Adjust based on prompt characteristics
        if analysis["has_examples"]:
            base_performance["accuracy"] += 0.1
            base_performance["clarity"] += 0.1
        
        if analysis["has_step_by_step"]:
            base_performance["completeness"] += 0.15
            base_performance["clarity"] += 0.1
        
        if analysis["structure_score"] > 0.7:
            base_performance["relevance"] += 0.1
            base_performance["completeness"] += 0.1
        
        # Normalize to 0-1 range
        for key in base_performance:
            base_performance[key] = min(1.0, base_performance[key])
        
        return base_performance
    
    