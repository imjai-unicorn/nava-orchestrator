 
import asyncio

class MockAIClient:
    """Mock AI client for Phase 1 testing"""
    
    async def generate_response(self, model: str, message: str) -> str:
        """Generate mock response based on selected model"""
        responses = {
            'deepseek-coder': f"[DeepSeek] นี่คือวิธีแก้ปัญหา code:\n\n```python\n# {message[:30]}...\nprint('Hello World')\n```",
            'claude-3-sonnet': f"[Claude] ให้ฉันวิเคราะห์เรื่องนี้:\n\n{message[:50]}... นี่เป็นเรื่องที่น่าสนใจมาก ฉันคิดว่า...",
            'gpt-4': f"[GPT-4] การตอบแบบสร้างสรรค์:\n\n{message[:50]}... เป็นหัวข้อที่มีความคิดสร้างสรรค์สูง",
            'gpt-3.5-turbo': f"[GPT-3.5] ฉันช่วยคุณได้: {message[:50]}... ตามที่คุณถาม ฉันแนะนำ..."
        }
        
        # Simulate processing time
        await asyncio.sleep(0.5)
        
        return responses.get(model, f"[{model}] Response: {message[:50]}...")