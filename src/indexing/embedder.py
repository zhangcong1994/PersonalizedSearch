import numpy as np
import os
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning("sentence-transformers 未安装")

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("需要安装 sentence-transformers: pip install sentence-transformers")
        
        self.model_name = model_name
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        try:
            logger.info(f"加载嵌入模型: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"模型加载成功，嵌入维度: {self.get_embedding_dimension()}")
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise
    
    def encode(self, texts: List[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("模型未加载")
        
        try:
            embeddings = self.model.encode(texts)
            return embeddings
        except Exception as e:
            logger.error(f"生成嵌入失败: {str(e)}")
            raise
    
    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
    
    def get_embedding_dimension(self) -> int:
        if self.model is None:
            return 0
        return self.model.get_sentence_embedding_dimension()
    
    def save_model(self, path: str):
        try:
            os.makedirs(path, exist_ok=True)
            self.model.save(path)
            logger.info(f"模型已保存: {path}")
        except Exception as e:
            logger.error(f"保存模型失败: {str(e)}")
            raise
    
    def load_model(self, path: str):
        try:
            self.model = SentenceTransformer(path, device=self.device)
            logger.info(f"已从 {path} 加载模型")
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise