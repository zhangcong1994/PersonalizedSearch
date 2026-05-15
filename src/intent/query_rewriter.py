from typing import Optional, List
from .base import BaseLLMClient
from .prompt_manager import PromptManager


class QueryRewriter:
    """
    查询重写器：核心类，负责将用户查询重写为更精准的搜索词
    """
    
    def __init__(self, llm_client: BaseLLMClient, prompt_manager: PromptManager,
                 template_name: str = "query_rewrite_basic"):
        """
        初始化查询重写器
        
        Args:
            llm_client: LLM客户端实例
            prompt_manager: Prompt模板管理器实例
            template_name: 使用的Prompt模板名称
        """
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager
        self.template_name = template_name
    
    def rewrite(self, query: str) -> str:
        """
        重写用户查询
        
        Args:
            query: 用户原始查询
            
        Returns:
            重写后的查询文本
            
        Raises:
            ValueError: 查询为空
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        # 格式化Prompt
        prompt = self.prompt_manager.format_template(
            self.template_name,
            query=query.strip()
        )
        
        # 调用LLM生成重写结果
        rewritten_query = self.llm_client.generate(prompt)
        
        return rewritten_query
    
    def batch_rewrite(self, queries: List[str]) -> List[str]:
        """
        批量重写多个查询
        
        Args:
            queries: 查询列表
            
        Returns:
            重写后的查询列表
        """
        results = []
        for query in queries:
            try:
                results.append(self.rewrite(query))
            except Exception as e:
                # 如果单个查询失败，返回原始查询
                results.append(query)
        
        return results
    
    def set_template(self, template_name: str) -> None:
        """
        设置使用的Prompt模板
        
        Args:
            template_name: 模板名称
            
        Raises:
            ValueError: 模板不存在
        """
        if template_name not in self.prompt_manager.list_templates():
            raise ValueError(f"Template '{template_name}' not found. "
                           f"Available templates: {self.prompt_manager.list_templates()}")
        self.template_name = template_name
