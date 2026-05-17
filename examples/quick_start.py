"""
意图识别模块快速开始示例（基于LangChain）

使用方式：
1. 在 .env 文件中配置 API_KEY（DEEPSEEK_API_KEY 或 OPENAI_API_KEY）
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
    
    # 2. 创建LangChain LLM客户端
    print("\nCreating LangChain LLM client...")
    try:
        llm_client = APIClientFactory.create(
            client_type=api_config.get("type"),
            model=api_config.get("model", "deepseek-chat"),
            max_tokens=api_config.get("max_tokens", 512),
            temperature=api_config.get("temperature", 0.1)
        )
        print(f"Client created successfully: {type(llm_client).__name__}")
        print(f"LLM Type: {type(llm_client.llm).__name__}")
    except Exception as e:
        print(f"Client creation failed: {e}")
        print("Please set DEEPSEEK_API_KEY or OPENAI_API_KEY environment variable in .env file")
        return
    
    # 3. 创建Prompt管理器（使用LangChain PromptTemplate）
    prompt_manager = PromptManager()
    print(f"\nAvailable templates: {prompt_manager.list_templates()}")
    
    # 4. 创建查询重写器（使用LangChain LLMChain）
    rewriter = QueryRewriter(llm_client.llm, prompt_manager, template_name)
    
    # 5. 测试基础查询重写
    test_queries = [
        "推荐一些好看的电影",
        "如何学习Python",
        "今天天气怎么样",
        "人工智能最新进展"
    ]
    
    print("\n=== Basic Query Rewriting ===")
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Original Query: {query}")
        try:
            rewritten = rewriter.rewrite(query)
            print(f"   Rewritten Query: {rewritten}")
        except Exception as e:
            print(f"   Error: {e}")
    
    # 6. 测试带上下文的查询重写
    print("\n=== Context-aware Query Rewriting ===")
    context = [
        "用户问：最近有什么新电影？",
        "助手答：最近有《奥本海默》《芭比》等热门影片"
    ]
    query = "推荐一些好看的"
    print(f"\nContext: {context}")
    print(f"Original Query: {query}")
    try:
        rewritten = rewriter.rewrite(query, context=context)
        print(f"Rewritten Query: {rewritten}")
    except Exception as e:
        print(f"Error: {e}")
    
    # 7. 测试个性化查询重写
    print("\n=== Personalized Query Rewriting ===")
    user_profile = "用户喜欢科幻和悬疑类型的电影，偏好高评分的影片"
    query = "推荐一些好看的"
    print(f"\nUser Profile: {user_profile}")
    print(f"Original Query: {query}")
    try:
        rewritten = rewriter.rewrite(query, user_profile=user_profile)
        print(f"Rewritten Query: {rewritten}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
