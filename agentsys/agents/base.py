"""
Base agent implementation.
"""

from typing import Any, Dict, Optional

class BaseAgent:
    """Base class for all agents in the system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
