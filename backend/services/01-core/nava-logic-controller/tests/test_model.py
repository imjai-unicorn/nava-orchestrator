import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').parent / 'app'))

from core.decision_engine import EnhancedDecisionEngine
engine = EnhancedDecisionEngine()

try:
    result = engine.select_model('test message')
    print('Result:', result)
    print('Type:', type(result))
    print('Length:', len(result))
    print('Values:', [type(x) for x in result])
except Exception as e:
    print('Error:', e)
    import traceback
    traceback.print_exc()
"