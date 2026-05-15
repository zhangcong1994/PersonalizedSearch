from typing import Dict, Optional


class PromptManager:
    """
    Prompt模板管理器：管理和格式化各种Prompt模板
    """
    
    def __init__(self):
        """初始化Prompt模板库"""
        self.templates: Dict[str, str] = {
            "query_rewrite_basic": (
                "请将以下用户查询改写成更明确、更适合搜索的关键词：\n"
                "用户查询：{query}\n"
                "重写后的搜索词："
            ),
            "query_rewrite_enhanced": (
                "你是一个搜索助手。请将用户的自然语言查询转换为精准的搜索关键词。\n"
                "要求：\n"
                "1. 提取核心概念\n"
                "2. 使用简洁的短语\n"
                "3. 保留关键实体\n"
                "\n"
                "用户查询：{query}\n"
                "搜索关键词："
            ),
            "query_rewrite_expanded": (
                "请将用户查询进行扩展和优化，使其更适合搜索引擎检索：\n"
                "用户查询：{query}\n"
                "优化后的查询："
            )
        }
    
    def register_template(self, name: str, template: str) -> None:
        """
        注册新的Prompt模板
        
        Args:
            name: 模板名称
            template: 模板内容，支持{变量}格式
        """
        self.templates[name] = template
    
    def get_template(self, name: str) -> Optional[str]:
        """
        获取指定的Prompt模板
        
        Args:
            name: 模板名称
            
        Returns:
            模板内容，如果不存在返回None
        """
        return self.templates.get(name)
    
    def format_template(self, name: str, **kwargs) -> str:
        """
        格式化指定的Prompt模板
        
        Args:
            name: 模板名称
            **kwargs: 模板变量的值
            
        Returns:
            格式化后的Prompt文本
            
        Raises:
            ValueError: 模板不存在
        """
        template = self.get_template(name)
        if template is None:
            raise ValueError(f"Prompt template '{name}' not found. "
                           f"Available templates: {list(self.templates.keys())}")
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required parameter for template '{name}': {e}")
    
    def list_templates(self) -> list:
        """
        获取所有已注册的模板名称
        
        Returns:
            模板名称列表
        """
        return list(self.templates.keys())
