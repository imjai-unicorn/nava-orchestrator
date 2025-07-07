# app/service/__init__.py
"""NAVA Service Module"""

# Import all service components
from .logic_orchestrator import LogicOrchestrator
from .service_discovery import ServiceDiscovery  
from .real_ai_client import RealAIClient

# Make available at package level
__all__ = [
    'LogicOrchestrator',
    'ServiceDiscovery', 
    'RealAIClient'
]