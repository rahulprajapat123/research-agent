"""
LLM client - unified interface for OpenAI, Anthropic, Azure OpenAI
"""
from typing import Optional
import openai
import anthropic
from config import get_settings

settings = get_settings()


class LLMClient:
    """Unified LLM client supporting multiple providers"""
    
    def __init__(self):
        self.provider = settings.llm_provider
        
        if self.provider == "openai":
            self.client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        elif self.provider == "anthropic":
            self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        elif self.provider == "azure":
            self.client = openai.AsyncAzureOpenAI(
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_key,
                api_version="2024-02-01"
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    async def complete(
        self,
        prompt: str,
        temperature: float = None,
        max_tokens: int = None,
        model: str = None
    ) -> str:
        """
        Generate completion from prompt
        
        Returns the response text
        """
        temperature = temperature or settings.llm_temperature
        max_tokens = max_tokens or settings.llm_max_tokens
        model = model or settings.llm_model
        
        if self.provider in ["openai", "azure"]:
            response = await self.client.chat.completions.create(
                model=model if self.provider == "openai" else settings.azure_openai_deployment,
                messages=[
                    {"role": "system", "content": "You are a research analysis assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        
        elif self.provider == "anthropic":
            response = await self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        
        raise ValueError(f"Unsupported provider: {self.provider}")
