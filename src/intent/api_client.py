import os
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel


class LangChainLLMClient:
    """
    LangChain LLM客户端封装类

    使用LangChain的统一接口封装不同的LLM提供商，
    提供统一的generate方法调用接口。
    同时保留原始API参数，供 generate_with_reasoning 使用原生 OpenAI client
    来捕获 reasoning_content（LangChain ChatOpenAI 会丢弃此字段）。
    """

    def __init__(self, llm: BaseChatModel, raw_params: Optional[dict] = None):
        """
        初始化客户端

        Args:
            llm: LangChain的BaseChatModel实例
            raw_params: 原始API参数，用于原生 OpenAI client 调用
                {"api_key", "base_url", "model", "max_tokens", "temperature", "extra_body"}
        """
        self.llm = llm
        self._raw_params = raw_params or {}

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

    def generate_with_reasoning(self, prompt: str) -> dict[str, str]:
        """
        调用LLM生成响应，同时返回 thinking 和 content。

        使用原生 OpenAI client 直接调用，以捕获 LangChain ChatOpenAI
        会丢弃的 reasoning_content 字段。

        Args:
            prompt: 输入的Prompt文本

        Returns:
            {"content": "最终评分结果", "reasoning_content": "推理过程"}
            对于不支持 thinking 的模型，reasoning_content 为空字符串
        """
        if not self._raw_params:
            content = self.generate(prompt)
            return {"content": content, "reasoning_content": ""}

        from openai import OpenAI

        client = OpenAI(
            api_key=self._raw_params["api_key"],
            base_url=self._raw_params["base_url"],
        )

        kwargs = dict(
            model=self._raw_params["model"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self._raw_params.get("max_tokens", 4096),
            temperature=self._raw_params.get("temperature", 1.0),
        )
        if self._raw_params.get("extra_body"):
            kwargs["extra_body"] = self._raw_params["extra_body"]

        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        content = choice.message.content or ""
        reasoning = getattr(choice.message, "reasoning_content", "") or ""

        return {"content": content.strip(), "reasoning_content": reasoning}


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
            kwargs["extra_body"] = extra_body

        llm = ChatOpenAI(**kwargs)
        raw_params = {
            "api_key": openai_api_key,
            "base_url": "https://api.openai.com/v1",
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "extra_body": extra_body,
        }
        return LangChainLLMClient(llm, raw_params=raw_params)

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
            kwargs["extra_body"] = extra_body

        llm = ChatOpenAI(**kwargs)
        effective_temp = 1.0 if thinking else temperature
        raw_params = {
            "api_key": deepseek_api_key,
            "base_url": "https://api.deepseek.com/v1",
            "model": model,
            "max_tokens": max_tokens,
            "temperature": effective_temp,
            "extra_body": extra_body,
        }
        return LangChainLLMClient(llm, raw_params=raw_params)

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
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=False,
        )
        if extra_body:
            kwargs["extra_body"] = extra_body

        llm = ChatOpenAI(**kwargs)
        raw_params = {
            "api_key": zhipu_api_key,
            "base_url": "https://open.bigmodel.cn/api/paas/v4/",
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "extra_body": extra_body if extra_body else None,
        }
        return LangChainLLMClient(llm, raw_params=raw_params)
