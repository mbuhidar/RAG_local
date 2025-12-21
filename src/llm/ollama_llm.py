"""
Ollama LLM Integration

Provides integration with Ollama for running local LLMs.
Ollama is the easiest way to run local models like Llama2, Mistral, etc.
"""

import json
import logging
from typing import Any, Dict, Iterator, Optional

import requests

from . import LocalLLM

logger = logging.getLogger(__name__)


class OllamaLLM(LocalLLM):
    """Ollama implementation for local LLM inference."""

    def __init__(
        self,
        model_name: str = "llama2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        context_window: int = 4096
    ):
        """
        Initialize Ollama LLM.

        Args:
            model_name: Name of the Ollama model (llama2, mistral, phi, etc.)
            base_url: Base URL for Ollama API
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            context_window: Context window size
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.context_window = context_window

        self._verify_connection()

    def _verify_connection(self):
        """Verify connection to Ollama server."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            logger.info("Successfully connected to Ollama server")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            logger.error("Make sure Ollama is running: ollama serve")
            raise ConnectionError(
                "Could not connect to Ollama. "
                "Please ensure Ollama is running (ollama serve)"
            )

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream response (not used in this method)

        Returns:
            Generated text
        """
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error generating response: {e}")
            raise

    def stream_generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Iterator[str]:
        """
        Stream generated text from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Yields:
            Chunks of generated text
        """
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=120
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "response" in chunk:
                        yield chunk["response"]
                    if chunk.get("done", False):
                        break
        except requests.exceptions.RequestException as e:
            logger.error(f"Error streaming response: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": self.model_name},
                timeout=10
            )
            response.raise_for_status()
            model_info = response.json()
            
            return {
                "provider": "ollama",
                "model_name": self.model_name,
                "base_url": self.base_url,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "context_window": self.context_window,
                "model_details": model_info
            }
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not fetch model info: {e}")
            return {
                "provider": "ollama",
                "model_name": self.model_name,
                "base_url": self.base_url,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "context_window": self.context_window
            }

    def list_available_models(self) -> list:
        """List all available Ollama models."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            models = response.json().get("models", [])
            return [model.get("name") for model in models]
        except requests.exceptions.RequestException as e:
            logger.error(f"Error listing models: {e}")
            return []

    def pull_model(self, model_name: Optional[str] = None) -> bool:
        """
        Pull a model from Ollama registry.

        Args:
            model_name: Name of model to pull (defaults to self.model_name)

        Returns:
            True if successful
        """
        model_name = model_name or self.model_name
        
        try:
            logger.info(f"Pulling model: {model_name}")
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=600
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    status = json.loads(line)
                    logger.info(f"Pull status: {status.get('status', '')}")

            logger.info(f"Successfully pulled model: {model_name}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Error pulling model: {e}")
            return False
