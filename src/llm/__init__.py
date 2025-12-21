"""
Local LLM Module

Provides integration with local LLM providers including Ollama, llama.cpp, and GPT4All.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)


class LocalLLM(ABC):
    """Abstract base class for local LLM implementations."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False
    ) -> str:
        """Generate text from prompt."""
        pass

    @abstractmethod
    def stream_generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Iterator[str]:
        """Stream generated text from prompt."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        pass
