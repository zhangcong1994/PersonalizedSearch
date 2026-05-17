import pandas as pd
import json
import os
import logging

logger = logging.getLogger(__name__)

class DataIO:
    @staticmethod
    def read_csv(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"读取CSV失败 {file_path}: {str(e)}")
            raise
    
    @staticmethod
    def write_csv(data: pd.DataFrame, file_path: str) -> bool:
        try:
            data.to_csv(file_path, index=False)
            logger.info(f"CSV已保存: {file_path}")
            return True
        except Exception as e:
            logger.error(f"写入CSV失败 {file_path}: {str(e)}")
            return False
    
    @staticmethod
    def read_json(file_path: str):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"读取JSON失败 {file_path}: {str(e)}")
            raise
    
    @staticmethod
    def write_json(data, file_path: str) -> bool:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"JSON已保存: {file_path}")
            return True
        except Exception as e:
            logger.error(f"写入JSON失败 {file_path}: {str(e)}")
            return False
    
    @staticmethod
    def read_parquet(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_parquet(file_path)
        except Exception as e:
            logger.error(f"读取Parquet失败 {file_path}: {str(e)}")
            raise
    
    @staticmethod
    def write_parquet(data: pd.DataFrame, file_path: str) -> bool:
        try:
            data.to_parquet(file_path, index=False)
            logger.info(f"Parquet已保存: {file_path}")
            return True
        except Exception as e:
            logger.error(f"写入Parquet失败 {file_path}: {str(e)}")
            return False
    
    @staticmethod
    def read_text(file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            logger.error(f"读取文本失败 {file_path}: {str(e)}")
            raise
    
    @staticmethod
    def write_text(content: str, file_path: str) -> bool:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"文本已保存: {file_path}")
            return True
        except Exception as e:
            logger.error(f"写入文本失败 {file_path}: {str(e)}")
            return False
    
    @staticmethod
    def list_files(directory: str, extension: str = None) -> list:
        try:
            files = []
            for filename in os.listdir(directory):
                if extension is None or filename.endswith(extension):
                    files.append(os.path.join(directory, filename))
            return sorted(files)
        except Exception as e:
            logger.error(f"列出文件失败 {directory}: {str(e)}")
            return []