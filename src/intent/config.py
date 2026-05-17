import os
import yaml
from typing import Dict, Any, Optional


class IntentConfig:
    """
    意图模块配置管理器：加载和管理配置
    """
    
    def __init__(self):
        self.config: Dict[str, Any] = {}
    
    def load(self, config_path: str) -> None:
        """
        从YAML文件加载配置
        
        Args:
            config_path: 配置文件路径
            
        Raises:
            FileNotFoundError: 配置文件不存在
            yaml.YAMLError: YAML解析错误
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    def get_api_client_config(self) -> Dict[str, Any]:
        """
        获取API客户端配置
        
        Returns:
            API客户端配置字典
        """
        return self.config.get('intent', {}).get('api_client', {})
    
    def get_prompt_template_config(self) -> str:
        """
        获取Prompt模板配置
        
        Returns:
            模板名称
        """
        return self.config.get('intent', {}).get('prompt_template', 'query_rewrite_basic')
    
    def get_intent_config(self) -> Dict[str, Any]:
        """
        获取意图模块的完整配置
        
        Returns:
            意图模块配置字典
        """
        return self.config.get('intent', {})
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键
            default: 默认值
            
        Returns:
            配置值
        """
        return self.config.get(key, default)


# 全局配置实例
_global_config = IntentConfig()


def load_config(config_path: str = "config.yaml") -> IntentConfig:
    """
    加载全局配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置实例
    """
    _global_config.load(config_path)
    return _global_config


def get_config() -> IntentConfig:
    """
    获取全局配置实例
    
    Returns:
        配置实例
    """
    return _global_config
