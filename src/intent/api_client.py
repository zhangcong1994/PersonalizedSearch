import os
import requests
from .base import BaseLLMClient


class OpenAIClient(BaseLLMClient):
    """
    OpenAI API客户端 - 使用requests直接调用
    """
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo", 
                 max_tokens: int = 512, temperature: float = 0.7):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.base_url = "https://api.openai.com/v1"
    
    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        return response.json()["choices"][0]["message"]["content"].strip()


class DeepSeekClient(BaseLLMClient):
    """
    DeepSeek API客户端 - 使用requests直接调用
    """
    
    def __init__(self, api_key: str = None, model: str = "deepseek-chat",
                 max_tokens: int = 512, temperature: float = 0.7):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        if not self.api_key:
            raise ValueError("DeepSeek API key is required")
        
        self.base_url = "https://api.deepseek.com/v1"
    
    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        return response.json()["choices"][0]["message"]["content"].strip()


class APIClientFactory:
    """
    API客户端工厂类
    """
    
    @staticmethod
    def create(client_type: str, **kwargs) -> BaseLLMClient:
        client_type = client_type.lower().strip()
        
        if client_type == "openai":
            return OpenAIClient(**kwargs)
        elif client_type == "deepseek":
            return DeepSeekClient(**kwargs)
        else:
            raise ValueError(f"Unsupported client type: {client_type}. "
                           f"Supported types: ['openai', 'deepseek']")
