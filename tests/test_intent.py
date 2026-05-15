"""
意图识别模块测试
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.intent import (
    OpenAIClient,
    DeepSeekClient,
    APIClientFactory,
    PromptManager,
    QueryRewriter
)


class TestPromptManager(unittest.TestCase):
    """测试Prompt管理器"""
    
    def setUp(self):
        self.prompt_manager = PromptManager()
    
    def test_register_template(self):
        """测试注册新模板"""
        self.prompt_manager.register_template("test_template", "Hello {name}")
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


class TestAPIClientFactory(unittest.TestCase):
    """测试API客户端工厂"""
    
    def test_create_openai_client(self):
        """测试创建OpenAI客户端"""
        # 注意：需要设置环境变量才能实际测试
        import os
        if os.getenv("OPENAI_API_KEY"):
            client = APIClientFactory.create("openai")
            self.assertIsInstance(client, OpenAIClient)
    
    def test_create_deepseek_client(self):
        """测试创建DeepSeek客户端"""
        import os
        if os.getenv("DEEPSEEK_API_KEY"):
            client = APIClientFactory.create("deepseek")
            self.assertIsInstance(client, DeepSeekClient)
    
    def test_create_invalid_client(self):
        """测试创建无效客户端类型"""
        with self.assertRaises(ValueError):
            APIClientFactory.create("invalid")


class TestQueryRewriter(unittest.TestCase):
    """测试查询重写器"""
    
    def test_rewrite_empty_query(self):
        """测试空查询"""
        prompt_manager = PromptManager()
        # 使用mock客户端避免实际API调用
        class MockClient:
            def generate(self, prompt):
                return "重写结果"
        
        rewriter = QueryRewriter(MockClient(), prompt_manager)
        
        with self.assertRaises(ValueError):
            rewriter.rewrite("")
    
    def test_batch_rewrite(self):
        """测试批量重写"""
        prompt_manager = PromptManager()
        
        class MockClient:
            def generate(self, prompt):
                return "重写结果"
        
        rewriter = QueryRewriter(MockClient(), prompt_manager)
        results = rewriter.batch_rewrite(["查询1", "查询2"])
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], "重写结果")


if __name__ == "__main__":
    unittest.main()
