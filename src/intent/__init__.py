"""
意图识别与查询重写模块（基于LangChain）

核心功能：
- 查询重写：将用户自然语言查询转换为精准的搜索关键词
- 支持多种LLM API（OpenAI、DeepSeek）
- 可配置的Prompt模板
- 使用LangChain构建完整的查询重写链

导出类：
- LangChainLLMClient: LangChain LLM客户端封装
- APIClientFactory: API客户端工厂类（使用LangChain）
- PromptManager: Prompt模板管理器（使用LangChain PromptTemplate）
- QueryRewriter: 查询重写器（使用LangChain LLMChain）
- IntentConfig: 配置管理器
"""

from .api_client import LangChainLLMClient, APIClientFactory
from .prompt_manager import PromptManager
from .query_rewriter import QueryRewriter
from .config import IntentConfig, load_config, get_config


__all__ = [
    'LangChainLLMClient',
    'APIClientFactory',
    'PromptManager',
    'QueryRewriter',
    'IntentConfig',
    'load_config',
    'get_config'
]
