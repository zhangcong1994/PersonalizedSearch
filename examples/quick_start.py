"""
意图识别模块快速开始示例

使用方式：
1. 在 .env 文件中配置 DEEPSEEK_API_KEY
2. 运行: python examples/quick_start.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()  # 从 .env 文件加载环境变量
except ImportError:
    print("Warning: python-dotenv not installed, using system environment variables")

from src.intent import (
    APIClientFactory,
    PromptManager,
    QueryRewriter,
    load_config
)


def main():
    # 1. 加载配置
    print("Loading configuration...")
    try:
        config = load_config("config.yaml")
        api_config = config.get_api_client_config()
        template_name = config.get_prompt_template_config()
        print(f"API Client Type: {api_config.get('type')}")
        print(f"Prompt Template: {template_name}")
    except Exception as e:
        print(f"Config load failed, using defaults: {e}")
        api_config = {"type": "deepseek"}
        template_name = "query_rewrite_basic"
    
    # 2. 创建API客户端
    print("\nCreating API client...")
    try:
        client = APIClientFactory.create(
            client_type=api_config.get("type"),
            model=api_config.get("model", "deepseek-chat"),
            max_tokens=api_config.get("max_tokens", 512),
            temperature=api_config.get("temperature", 0.1)
        )
        print(f"Client created successfully: {type(client).__name__}")
    except Exception as e:
        print(f"Client creation failed: {e}")
        print("Please set DEEPSEEK_API_KEY environment variable in .env file")
        return
    
    # 3. 创建Prompt管理器
    prompt_manager = PromptManager()
    print(f"\nAvailable templates: {prompt_manager.list_templates()}")
    
    # 4. 创建查询重写器
    rewriter = QueryRewriter(client, prompt_manager, template_name)
    
    # 5. 测试查询重写
    test_queries = [
        "推荐一些好看的电影",
        "如何学习Python",
        "今天天气怎么样",
        "人工智能最新进展"
    ]
    
    print("\n=== Query Rewriting Results ===")
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Original Query: {query}")
        try:
            rewritten = rewriter.rewrite(query)
            print(f"   Rewritten Query: {rewritten}")
        except Exception as e:
            print(f"   Error: {e}")


if __name__ == "__main__":
    main()
