 
from typing import Dict, Any

class DecisionEngine:
    """AI model selection engine"""
    
    def __init__(self):
        self.model_rules = {
            'code': ['code', 'programming', 'function', 'bug', 'debug', 'python', 'javascript'],
            'analysis': ['analyze', 'review', 'compare', 'evaluate', 'research'],
            'creative': ['write', 'create', 'story', 'content', 'blog'],
            'conversation': ['hello', 'hi', 'how', 'what', 'explain', 'help']
        }
        
        self.model_mapping = {
            'code': 'deepseek-coder',
            'analysis': 'claude-3-sonnet',
            'creative': 'gpt-4',
            'conversation': 'gpt-3.5-turbo'
        }
    
    async def decide_model(self, message: str) -> Dict[str, Any]:
        """Select appropriate AI model based on message content"""
        message_lower = message.lower()
        
        scores = {}
        for category, keywords in self.model_rules.items():
            score = sum(1 for keyword in keywords if keyword in message_lower)
            if score > 0:
                scores[category] = score
        
        if scores:
            selected_category = max(scores, key=scores.get)
            confidence = min(scores[selected_category] / 3.0, 1.0)
        else:
            selected_category = 'conversation'
            confidence = 0.5
        
        selected_model = self.model_mapping[selected_category]
        
        return {
            'model': selected_model,
            'confidence': confidence,
            'reasoning': {
                'category': selected_category,
                'scores': scores,
                'keywords_found': [kw for kw in self.model_rules[selected_category] 
                                 if kw in message_lower]
            }
        }