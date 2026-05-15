"""
意图识别与查询重写模块

核心功能：
- 查询重写：将用户自然语言查询转换为精准的搜索关键词
- 支持多种LLM API（OpenAI、DeepSeek）
- 可配置的Prompt模板

导出类：
- BaseLLMClient: LLM客户端抽象基类
- OpenAIClient: OpenAI API客户端
- DeepSeekClient: DeepSeek API客户端
- APIClientFactory: API客户端工厂类
- PromptManager: Prompt模板管理器
- QueryRewriter: 查询重写器
- IntentConfig: 配置管理器
"""

from .base import BaseLLMClient
from .api_client import OpenAIClient, DeepSeekClient, APIClientFactory
from .prompt_manager import PromptManager
from .query_rewriter import QueryRewriter
from .config import IntentConfig, load_config, get_config


__all__ = [
    'BaseLLMClient',
    'OpenAIClient',
    'DeepSeekClient',
    'APIClientFactory',
    'PromptManager',
    'QueryRewriter',
    'IntentConfig',
    'load_config',
    'get_config'
]
