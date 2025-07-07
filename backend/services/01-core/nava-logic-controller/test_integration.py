python -c "
from app.core.decision_engine import EnhancedDecisionEngine
engine = EnhancedDecisionEngine()
result = engine.select_model('test dynamic weights system')
print('Model:', result[0])
print('Confidence:', result[1])
print('Reasoning:', result[2])
"