import pandas as pd
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.clean_patterns = [
            (r'<[^>]+>', ''),          
            (r'&[a-zA-Z]+;', ' '),     
            (r'[^\w\s\.\,\!\?\-]', ' '), 
            (r'\s+', ' '),             
        ]
    
    def load_raw_data(self, file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"加载原始数据失败 {file_path}: {str(e)}")
            raise
    
    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        
        text = text.strip()
        for pattern, replacement in self.clean_patterns:
            text = re.sub(pattern, replacement, text)
        
        return text.strip()
    
    def normalize_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r'\b(\d+)\b', lambda m: m.group(1), text)
        
        return text.strip()
    
    def filter_short_documents(self, data: pd.DataFrame, min_length: int = 10) -> pd.DataFrame:
        return data[data['content'].apply(lambda x: len(str(x)) >= min_length)]
    
    def remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        initial_count = len(data)
        data = data.drop_duplicates()
        removed_count = initial_count - len(data)
        
        if removed_count > 0:
            logger.info(f"移除了 {removed_count} 条重复记录")
        
        return data
    
    def save_processed_data(self, data: pd.DataFrame, output_path: str) -> bool:
        try:
            data.to_parquet(output_path, index=False)
            logger.info(f"处理后数据已保存: {output_path}")
            return True
        except Exception as e:
            logger.error(f"保存处理后数据失败 {output_path}: {str(e)}")
            return False
    
    def process_documents(self, documents: pd.DataFrame) -> pd.DataFrame:
        logger.info("开始处理文档数据...")
        
        if 'content' not in documents.columns:
            logger.warning("文档数据中没有 'content' 列")
            return documents
        
        documents = documents.copy()
        documents['content'] = documents['content'].apply(self.clean_text)
        documents['content'] = documents['content'].apply(self.normalize_text)
        
        documents = self.filter_short_documents(documents)
        documents = self.remove_duplicates(documents)
        
        logger.info(f"文档处理完成，剩余 {len(documents)} 条")
        return documents
    
    def process_queries(self, queries: pd.DataFrame) -> pd.DataFrame:
        logger.info("开始处理查询数据...")
        
        if 'Query' not in queries.columns:
            logger.warning("查询数据中没有 'Query' 列")
            return queries
        
        queries = queries.copy()
        queries['Query'] = queries['Query'].apply(self.clean_text)
        queries['Query'] = queries['Query'].apply(self.normalize_text)
        
        queries = queries[queries['Query'].str.len() > 0]
        queries = self.remove_duplicates(queries)
        
        logger.info(f"查询处理完成，剩余 {len(queries)} 条")
        return queries