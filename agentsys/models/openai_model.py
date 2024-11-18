"""
OpenAI model implementation.
"""

from typing import Any, Dict, List, Optional
from openai import OpenAI

from .base import BaseModel

class OpenAIModel(BaseModel):
    """OpenAI model implementation."""
    
    def __init__(self, client: Optional[OpenAI] = None):
        self.client = client or OpenAI()
    
    def generate(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        """Generate a response using OpenAI's API."""
        return self.client.chat.completions.create(
            messages=messages,
            stream=False,
            **kwargs
        )
    
    def stream(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        """Stream a response using OpenAI's API."""
        return self.client.chat.completions.create(
            messages=messages,
            stream=True,
            **kwargs
        )
