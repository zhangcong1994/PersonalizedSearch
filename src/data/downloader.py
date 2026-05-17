import os
import urllib.request
import tarfile
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class AOL4PSDownloader:
    def __init__(self, save_dir: str = "./data/raw/aol4ps"):
        self.save_dir = save_dir
        self.base_url = "https://github.com/AnnKwok/AOL4PS/raw/main/"
        self.files_to_download = [
            "data.tar.gz",
            "query.tar.gz", 
            "doc.tar.gz"
        ]
        os.makedirs(save_dir, exist_ok=True)
    
    def download_file(self, url: str, filename: str) -> bool:
        try:
            file_path = os.path.join(self.save_dir, filename)
            if os.path.exists(file_path):
                logger.info(f"文件已存在，跳过下载: {filename}")
                return True
            
            logger.info(f"开始下载: {url}")
            urllib.request.urlretrieve(url, file_path)
            logger.info(f"下载完成: {filename}")
            return True
        except Exception as e:
            logger.error(f"下载失败 {filename}: {str(e)}")
            return False
    
    def download(self, timeout: int = 300, max_retries: int = 3) -> bool:
        """下载所有数据集文件"""
        success_count = 0
        
        for filename in self.files_to_download:
            url = self.base_url + filename
            success = False
            
            for attempt in range(max_retries):
                try:
                    success = self.download_file(url, filename)
                    if success:
                        success_count += 1
                        break
                    else:
                        logger.warning(f"第 {attempt + 1} 次尝试下载 {filename} 失败")
                except Exception as e:
                    logger.error(f"下载尝试 {attempt + 1} 失败: {str(e)}")
            
            if not success:
                logger.error(f"无法下载 {filename}")
        
        return success_count == len(self.files_to_download)
    
    def extract_tar(self, tar_path: str, extract_dir: str) -> bool:
        try:
            logger.info(f"解压文件: {tar_path}")
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=extract_dir)
            logger.info(f"解压完成: {tar_path}")
            return True
        except Exception as e:
            logger.error(f"解压失败 {tar_path}: {str(e)}")
            return False
    
    def extract_files(self, cleanup: bool = False) -> bool:
        """解压所有下载的压缩文件"""
        success_count = 0
        
        for filename in self.files_to_download:
            if filename.endswith('.tar.gz'):
                tar_path = os.path.join(self.save_dir, filename)
                if os.path.exists(tar_path):
                    if self.extract_tar(tar_path, self.save_dir):
                        success_count += 1
                        if cleanup:
                            os.remove(tar_path)
                            logger.info(f"已删除压缩文件: {filename}")
        
        return success_count == len(self.files_to_download)
    
    def verify_checksum(self) -> bool:
        """验证数据完整性（预留）"""
        logger.warning("校验和验证功能尚未实现")
        return True
    
    def cleanup(self):
        """清理临时文件"""
        for filename in self.files_to_download:
            file_path = os.path.join(self.save_dir, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"已清理: {file_path}")

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    downloader = AOL4PSDownloader()
    
    logger.info("开始下载AOL4PS数据集...")
    if downloader.download():
        logger.info("下载完成，开始解压...")
        if downloader.extract_files():
            logger.info("解压完成！")
            downloader.verify_checksum()
        else:
            logger.error("解压失败")
    else:
        logger.error("下载失败")

if __name__ == "__main__":
    main()
