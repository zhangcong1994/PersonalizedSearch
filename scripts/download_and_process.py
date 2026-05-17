import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging
from src.data import AOL4PSDownloader, DataProcessor, DataAnalyzer, DataIO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    logger.info("="*60)
    logger.info("开始下载和处理AOL4PS数据集")
    logger.info("="*60)
    
    downloader = AOL4PSDownloader()
    
    logger.info("\n【步骤1】下载数据集")
    if downloader.download():
        logger.info("下载成功！")
    else:
        logger.error("下载失败，请检查网络连接")
        return
    
    logger.info("\n【步骤2】解压文件")
    if downloader.extract_files():
        logger.info("解压成功！")
    else:
        logger.error("解压失败")
        return
    
    logger.info("\n【步骤3】探索数据")
    data_dir = downloader.save_dir
    files = DataIO.list_files(data_dir)
    
    if not files:
        logger.error("未找到数据文件")
        return
    
    logger.info(f"找到数据文件: {files}")
    
    for file in files:
        if file.endswith('.txt') or file.endswith('.csv'):
            try:
                logger.info(f"\n--- 分析文件: {os.path.basename(file)} ---")
                content = DataIO.read_text(file)
                lines = content.split('\n')[:10]
                logger.info(f"前10行内容:")
                for i, line in enumerate(lines):
                    logger.info(f"  {i+1}: {line[:100]}...")
            except Exception as e:
                logger.warning(f"无法读取文件 {file}: {str(e)}")
    
    logger.info("\n" + "="*60)
    logger.info("数据下载和初步探索完成！")
    logger.info("="*60)
    logger.info("\n下一步：")
    logger.info("1. 查看数据格式后，可以修改 processor.py 中的预处理逻辑")
    logger.info("2. 运行数据分析脚本查看统计信息")
    logger.info("3. 开始构建索引")

if __name__ == "__main__":
    main()