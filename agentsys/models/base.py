"""
Base model interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class BaseModel(ABC):
    """Abstract base class for all models."""
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        """Generate a response from the model."""
        raise NotImplementedError("Subclasses must implement generate()")
    
    @abstractmethod
    def stream(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        """Stream a response from the model."""
        raise NotImplementedError("Subclasses must implement stream()")
