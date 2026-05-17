from abc import ABC, abstractmethod
from typing import Optional, List


class BaseLLMClient(ABC):
    """
    抽象基类：定义LLM客户端的接口
    """
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        调用LLM生成响应
        
        Args:
            prompt: 输入的Prompt文本
            
        Returns:
            LLM生成的响应文本
        """
        pass


class BaseQueryRewriter(ABC):
    """
    抽象基类：定义查询重写器的接口
    """
    
    @abstractmethod
    def rewrite(self, query: str) -> str:
        """
        重写用户查询
        
        Args:
            query: 用户原始查询
            
        Returns:
            重写后的查询文本
        """
        pass
