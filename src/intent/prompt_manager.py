from typing import Dict, Optional

from langchain_core.prompts import PromptTemplate


class PromptManager:
    """
    Prompt模板管理器：使用LangChain的PromptTemplate进行模板管理和格式化
    
    功能：
    1. 注册和管理Prompt模板
    2. 使用LangChain PromptTemplate进行模板格式化
    3. 支持动态变量注入
    """
    
    def __init__(self):
        """初始化Prompt模板库"""
        self.templates: Dict[str, PromptTemplate] = {
            "query_rewrite_basic": PromptTemplate(
                input_variables=["query"],
                template="""请将以下用户查询改写成更明确、更适合搜索的关键词：
用户查询：{query}
重写后的搜索词："""
            ),
            "query_rewrite_enhanced": PromptTemplate(
                input_variables=["query"],
                template="""你是一个搜索助手。请将用户的自然语言查询转换为精准的搜索关键词。
要求：
1. 提取核心概念
2. 使用简洁的短语
3. 保留关键实体

用户查询：{query}
搜索关键词："""
            ),
            "query_rewrite_expanded": PromptTemplate(
                input_variables=["query"],
                template="""请将用户查询进行扩展和优化，使其更适合搜索引擎检索：
用户查询：{query}
优化后的查询："""
            ),
            "query_rewrite_context": PromptTemplate(
                input_variables=["query", "context"],
                template="""基于对话历史，将当前查询重写为独立搜索词：

对话历史：{context}
当前查询：{query}

重写后的搜索词："""
            ),
            "query_rewrite_personalized": PromptTemplate(
                input_variables=["query", "user_profile"],
                template="""根据用户偏好，将查询重写为更精准的搜索词：

用户偏好：{user_profile}
查询：{query}

重写后的搜索词："""
            )
        }
    
    def register_template(self, name: str, template: str, input_variables: list) -> None:
        """
        注册新的Prompt模板
        
        Args:
            name: 模板名称
            template: 模板内容，支持{变量}格式
            input_variables: 模板中使用的变量列表
        """
        self.templates[name] = PromptTemplate(
            input_variables=input_variables,
            template=template
        )
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """
        获取指定的Prompt模板
        
        Args:
            name: 模板名称
            
        Returns:
            PromptTemplate实例，如果不存在返回None
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
            ValueError: 模板不存在或缺少必要参数
        """
        template = self.get_template(name)
        if template is None:
            raise ValueError(f"Prompt template '{name}' not found. "
                           f"Available templates: {list(self.templates.keys())}")
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required parameter for template '{name}': {e}")
        except Exception as e:
            raise ValueError(f"Failed to format template '{name}': {str(e)}")
    
    def list_templates(self) -> list:
        """
        获取所有已注册的模板名称
        
        Returns:
            模板名称列表
        """
        return list(self.templates.keys())
    
    def get_template_variables(self, name: str) -> list:
        """
        获取指定模板的变量列表
        
        Args:
            name: 模板名称
            
        Returns:
            变量名称列表
            
        Raises:
            ValueError: 模板不存在
        """
        template = self.get_template(name)
        if template is None:
            raise ValueError(f"Prompt template '{name}' not found.")
        
        return template.input_variables
