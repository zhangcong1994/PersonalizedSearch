from typing import Dict, List, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

from .prompt_manager import PromptManager


class QueryRewriter:
    """
    查询重写器：核心类，负责将用户查询重写为更精准的搜索词
    
    使用LangChain的BaseLanguageModel直接调用，支持：
    1. 基础查询重写
    2. 带上下文的查询重写
    3. 个性化查询重写（结合用户画像）
    """
    
    def __init__(self, llm: BaseLanguageModel, prompt_manager: PromptManager,
                 template_name: str = "query_rewrite_basic"):
        """
        初始化查询重写器
        
        Args:
            llm: LangChain的BaseLanguageModel实例
            prompt_manager: Prompt模板管理器实例
            template_name: 使用的Prompt模板名称
        """
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.template_name = template_name
    
    def rewrite(self, query: str, context: Optional[List[str]] = None, 
                user_profile: Optional[str] = None) -> str:
        """
        重写用户查询
        
        Args:
            query: 用户原始查询
            context: 对话历史上下文（可选）
            user_profile: 用户画像/偏好（可选）
            
        Returns:
            重写后的查询文本
            
        Raises:
            ValueError: 查询为空或模板不匹配
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        # 根据参数选择合适的模板
        template_name = self._select_template(context, user_profile)
        
        # 准备输入参数
        inputs: Dict[str, str] = {"query": query.strip()}
        
        if context:
            inputs["context"] = "\n".join(context)
        
        if user_profile:
            inputs["user_profile"] = user_profile
        
        # 格式化Prompt
        prompt = self.prompt_manager.format_template(template_name, **inputs)
        
        # 调用LLM生成响应
        try:
            if hasattr(self.llm, 'invoke'):
                response = self.llm.invoke([HumanMessage(content=prompt)])
            else:
                response = self.llm.generate([prompt])
            
            # 处理不同的响应格式
            if hasattr(response, 'content'):
                return response.content.strip()
            elif isinstance(response, str):
                return response.strip()
            elif hasattr(response, 'generations') and len(response.generations) > 0:
                return response.generations[0][0].text.strip()
            else:
                return str(response).strip()
        except Exception as e:
            raise ValueError(f"Failed to rewrite query: {str(e)}")
    
    def _select_template(self, context: Optional[List[str]], 
                        user_profile: Optional[str]) -> str:
        """
        根据可用参数选择合适的模板
        
        Args:
            context: 对话历史上下文
            user_profile: 用户画像
            
        Returns:
            模板名称
        """
        # 优先选择最匹配的模板
        if context and user_profile:
            # 检查是否有同时支持上下文和用户画像的模板
            if "query_rewrite_full" in self.prompt_manager.list_templates():
                return "query_rewrite_full"
            # 如果没有，使用个性化模板
            return "query_rewrite_personalized"
        elif context:
            return "query_rewrite_context"
        elif user_profile:
            return "query_rewrite_personalized"
        
        return self.template_name
    
    def batch_rewrite(self, queries: List[str], 
                      contexts: Optional[List[Optional[List[str]]]] = None,
                      user_profiles: Optional[List[Optional[str]]] = None) -> List[str]:
        """
        批量重写多个查询
        
        Args:
            queries: 查询列表
            contexts: 每个查询对应的上下文列表（可选）
            user_profiles: 每个查询对应的用户画像（可选）
            
        Returns:
            重写后的查询列表
        """
        results = []
        contexts = contexts or [None] * len(queries)
        user_profiles = user_profiles or [None] * len(queries)
        
        for i, query in enumerate(queries):
            try:
                result = self.rewrite(
                    query=query,
                    context=contexts[i],
                    user_profile=user_profiles[i]
                )
                results.append(result)
            except Exception as e:
                results.append(query)
        
        return results
    
    def set_template(self, template_name: str) -> None:
        """
        设置默认使用的Prompt模板
        
        Args:
            template_name: 模板名称
            
        Raises:
            ValueError: 模板不存在
        """
        if template_name not in self.prompt_manager.list_templates():
            raise ValueError(f"Template '{template_name}' not found. "
                           f"Available templates: {self.prompt_manager.list_templates()}")
        self.template_name = template_name
