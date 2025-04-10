from typing import List, Dict, Any

from abc import ABC, abstractmethod


class chatModel(ABC):
    """Abstract base class for model APIs."""
    
    @abstractmethod
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        pass
    
    @abstractmethod
    def select_model(self, model_name: str = None):
        """Select a model to use."""
        pass
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response to the prompt."""
        pass