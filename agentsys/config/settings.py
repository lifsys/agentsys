"""
Settings management for the Swarm framework.
"""

from typing import Any, Dict, Optional

class Settings:
    """Global settings management."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        
    @property
    def config(self) -> Dict[str, Any]:
        return self._config.copy()
