import os
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel


class LangChainLLMClient:
    """
    LangChain LLM客户端封装类
    
    使用LangChain的统一接口封装不同的LLM提供商，
    提供统一的generate方法调用接口。
    """
    
    def __init__(self, llm: BaseChatModel):
        """
        初始化客户端
        
        Args:
            llm: LangChain的BaseChatModel实例
        """
        self.llm = llm
    
    def generate(self, prompt: str) -> str:
        """
        调用LLM生成响应
        
        Args:
            prompt: 输入的Prompt文本
            
        Returns:
            LLM生成的响应文本
        """
        from langchain_core.messages import HumanMessage
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()


class APIClientFactory:
    """
    API客户端工厂类 - 使用LangChain创建LLM客户端
    """

    @staticmethod
    def create(client_type: str, **kwargs) -> LangChainLLMClient:
        """
        创建LLM客户端

        Args:
            client_type: 客户端类型 ("openai", "deepseek", "zhipu")
            **kwargs: 额外参数（api_key, model, max_tokens, temperature, thinking, base_url 等）

        Returns:
            LangChainLLMClient实例

        Raises:
            ValueError: 不支持的客户端类型
        """
        client_type = client_type.lower().strip()

        api_key = kwargs.get('api_key')
        model = kwargs.get('model')
        max_tokens = kwargs.get('max_tokens', 512)
        temperature = kwargs.get('temperature', 0.7)
        thinking = kwargs.get('thinking', False)
        extra_body = kwargs.get('extra_body', None)

        if client_type == "openai":
            return APIClientFactory._create_openai_client(
                api_key=api_key,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body=extra_body,
            )
        elif client_type == "deepseek":
            return APIClientFactory._create_deepseek_client(
                api_key=api_key,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                thinking=thinking,
                extra_body=extra_body,
            )
        elif client_type == "zhipu":
            return APIClientFactory._create_zhipu_client(
                api_key=api_key,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                thinking=thinking,
                extra_body=extra_body,
            )
        else:
            raise ValueError(f"Unsupported client type: {client_type}. "
                           f"Supported types: ['openai', 'deepseek', 'zhipu']")
    
    @staticmethod
    def _create_openai_client(
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 512,
        temperature: float = 0.7,
        extra_body: Optional[dict] = None,
    ) -> LangChainLLMClient:
        """创建OpenAI客户端"""
        openai_api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not openai_api_key:
            raise ValueError("OpenAI API key is required. "
                           "Set OPENAI_API_KEY environment variable or pass api_key parameter.")

        kwargs = dict(
            model=model,
            api_key=openai_api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=False,
        )
        if extra_body:
            kwargs["model_kwargs"] = extra_body

        llm = ChatOpenAI(**kwargs)
        return LangChainLLMClient(llm)

    @staticmethod
    def _create_deepseek_client(
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        max_tokens: int = 512,
        temperature: float = 0.7,
        thinking: bool = False,
        extra_body: Optional[dict] = None,
    ) -> LangChainLLMClient:
        """创建DeepSeek客户端（使用OpenAI兼容API）"""
        deepseek_api_key = api_key or os.getenv("DEEPSEEK_API_KEY")

        if not deepseek_api_key:
            raise ValueError("DeepSeek API key is required. "
                           "Set DEEPSEEK_API_KEY environment variable or pass api_key parameter.")

        kwargs = dict(
            model=model,
            api_key=deepseek_api_key,
            base_url="https://api.deepseek.com/v1",
            max_tokens=max_tokens,
            verbose=False,
        )

        if thinking:
            kwargs["temperature"] = 1.0
        else:
            kwargs["temperature"] = temperature

        if extra_body:
            kwargs["model_kwargs"] = extra_body

        llm = ChatOpenAI(**kwargs)
        return LangChainLLMClient(llm)

    @staticmethod
    def _create_zhipu_client(
        api_key: Optional[str] = None,
        model: str = "glm-4-flash",
        max_tokens: int = 512,
        temperature: float = 0.7,
        thinking: bool = False,
        extra_body: Optional[dict] = None,
    ) -> LangChainLLMClient:
        """创建智谱 GLM 客户端（使用 OpenAI 兼容 API）"""
        zhipu_api_key = api_key or os.getenv("ZHIPU_API_KEY")

        if not zhipu_api_key:
            raise ValueError("Zhipu API key is required. "
                           "Set ZHIPU_API_KEY environment variable or pass api_key parameter.")

        if extra_body is None:
            extra_body = {}

        if thinking:
            extra_body["thinking"] = {"type": "enabled"}

        kwargs = dict(
            model=model,
            api_key=zhipu_api_key,
            base_url="https://api.z.ai/api/paas/v4/",
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=False,
        )
        if extra_body:
            kwargs["model_kwargs"] = extra_body

        llm = ChatOpenAI(**kwargs)
        return LangChainLLMClient(llm)
