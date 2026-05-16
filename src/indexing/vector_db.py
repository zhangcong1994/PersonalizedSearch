import numpy as np
import os
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False
    logger.warning("chromadb 未安装")

class VectorDB:
    def __init__(self, db_path: str = "./data/vector_db", embedding_dim: int = 384):
        if not HAS_CHROMADB:
            raise ImportError("需要安装 chromadb: pip install chromadb")
        
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self.client = None
        self.collection = None
        self._init_client()
    
    def _init_client(self):
        try:
            os.makedirs(self.db_path, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False
                )
            )
            logger.info(f"ChromaDB客户端已初始化: {self.db_path}")
        except Exception as e:
            logger.error(f"初始化ChromaDB失败: {str(e)}")
            raise
    
    def create_collection(self, name: str = "documents"):
        try:
            self.collection = self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"集合 '{name}' 已创建/获取")
        except Exception as e:
            logger.error(f"创建集合失败: {str(e)}")
            raise
    
    def add_documents(self, chunks: List[str], embeddings: np.ndarray, metadata: Optional[List[Dict]] = None) -> bool:
        if self.collection is None:
            self.create_collection()
        
        try:
            ids = [f"doc_{i}" for i in range(len(chunks))]
            
            if metadata is None:
                metadata = [{} for _ in chunks]
            
            self.collection.add(
                documents=chunks,
                embeddings=embeddings.tolist(),
                metadatas=metadata,
                ids=ids
            )
            logger.info(f"已添加 {len(chunks)} 条文档")
            return True
        except Exception as e:
            logger.error(f"添加文档失败: {str(e)}")
            return False
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict]:
        if self.collection is None:
            self.create_collection()
        
        try:
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k
            )
            
            return self._format_results(results)
        except Exception as e:
            logger.error(f"搜索失败: {str(e)}")
            return []
    
    def _format_results(self, results: Dict) -> List[Dict]:
        formatted = []
        for i in range(len(results['ids'][0])):
            formatted.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                'distance': results['distances'][0][i] if results['distances'] else None
            })
        return formatted
    
    def delete_document(self, doc_id: str) -> bool:
        try:
            self.collection.delete(ids=[doc_id])
            logger.info(f"已删除文档: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"删除文档失败: {str(e)}")
            return False
    
    def update_document(self, doc_id: str, chunk: str, embedding: np.ndarray) -> bool:
        try:
            self.collection.update(
                ids=[doc_id],
                documents=[chunk],
                embeddings=embedding.tolist()
            )
            logger.info(f"已更新文档: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"更新文档失败: {str(e)}")
            return False
    
    def persist(self) -> bool:
        try:
            self.client.persist()
            logger.info("数据库已持久化")
            return True
        except Exception as e:
            logger.error(f"持久化失败: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict:
        if self.collection is None:
            return {}
        
        try:
            stats = self.collection.count()
            return {
                'document_count': stats
            }
        except Exception as e:
            logger.error(f"获取统计信息失败: {str(e)}")
            return {}