"""
意图识别模块测试（基于LangChain）
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.intent import (
    APIClientFactory,
    PromptManager,
    QueryRewriter
)


class TestPromptManager(unittest.TestCase):
    """测试Prompt管理器（基于LangChain PromptTemplate）"""
    
    def setUp(self):
        self.prompt_manager = PromptManager()
    
    def test_register_template(self):
        """测试注册新模板"""
        self.prompt_manager.register_template(
            "test_template", 
            "Hello {name}", 
            ["name"]
        )
        self.assertIn("test_template", self.prompt_manager.list_templates())
    
    def test_format_template(self):
        """测试格式化模板"""
        result = self.prompt_manager.format_template(
            "query_rewrite_basic",
            query="测试查询"
        )
        self.assertIn("测试查询", result)
    
    def test_list_templates(self):
        """测试获取模板列表"""
        templates = self.prompt_manager.list_templates()
        self.assertIsInstance(templates, list)
        self.assertIn("query_rewrite_basic", templates)
    
    def test_get_template_variables(self):
        """测试获取模板变量"""
        variables = self.prompt_manager.get_template_variables("query_rewrite_basic")
        self.assertEqual(variables, ["query"])


class TestAPIClientFactory(unittest.TestCase):
    """测试API客户端工厂（基于LangChain）"""
    
    def test_create_openai_client(self):
        """测试创建OpenAI客户端"""
        if os.getenv("OPENAI_API_KEY"):
            client = APIClientFactory.create("openai")
            self.assertIsNotNone(client)
            self.assertHasAttr(client, 'llm')
    
    def test_create_deepseek_client(self):
        """测试创建DeepSeek客户端"""
        if os.getenv("DEEPSEEK_API_KEY"):
            client = APIClientFactory.create("deepseek")
            self.assertIsNotNone(client)
            self.assertHasAttr(client, 'llm')
    
    def test_create_invalid_client(self):
        """测试创建无效客户端类型"""
        with self.assertRaises(ValueError):
            APIClientFactory.create("invalid")


class TestQueryRewriter(unittest.TestCase):
    """测试查询重写器（基于LangChain LLMChain）"""
    
    def setUp(self):
        self.prompt_manager = PromptManager()
        
        # 创建Mock LLM
        from langchain_core.language_models import FakeListLLM
        self.mock_llm = FakeListLLM(responses=["重写结果"])
    
    def test_rewrite_empty_query(self):
        """测试空查询"""
        rewriter = QueryRewriter(self.mock_llm, self.prompt_manager)
        
        with self.assertRaises(ValueError):
            rewriter.rewrite("")
    
    def test_rewrite_basic(self):
        """测试基础查询重写"""
        rewriter = QueryRewriter(self.mock_llm, self.prompt_manager)
        result = rewriter.rewrite("测试查询")
        self.assertEqual(result, "重写结果")
    
    def test_batch_rewrite(self):
        """测试批量重写"""
        rewriter = QueryRewriter(self.mock_llm, self.prompt_manager)
        results = rewriter.batch_rewrite(["查询1", "查询2"])
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], "重写结果")
    
    def test_context_aware_rewrite(self):
        """测试带上下文的查询重写"""
        rewriter = QueryRewriter(self.mock_llm, self.prompt_manager)
        context = ["历史对话1", "历史对话2"]
        result = rewriter.rewrite("当前查询", context=context)
        self.assertEqual(result, "重写结果")
    
    def test_personalized_rewrite(self):
        """测试个性化查询重写"""
        rewriter = QueryRewriter(self.mock_llm, self.prompt_manager)
        user_profile = "用户喜欢科技类内容"
        result = rewriter.rewrite("推荐一些内容", user_profile=user_profile)
        self.assertEqual(result, "重写结果")
    
    def test_set_template(self):
        """测试设置模板"""
        rewriter = QueryRewriter(self.mock_llm, self.prompt_manager)
        rewriter.set_template("query_rewrite_enhanced")
        self.assertEqual(rewriter.template_name, "query_rewrite_enhanced")


if __name__ == "__main__":
    unittest.main()
