 
"""Development runner for NAVA Logic Controller"""
import uvicorn
from main import app
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))

from backend.shared.common.config import settings

if __name__ == "__main__":
    print(f"🚀 Starting NAVA Logic Controller on port {settings.SERVICE_PORT}")
    print(f"📖 API docs: http://localhost:{settings.SERVICE_PORT}/docs")
    print(f"🔍 Health check: http://localhost:{settings.SERVICE_PORT}/health")
    
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=settings.SERVICE_PORT,
        reload=True,
        log_level="info"
    )