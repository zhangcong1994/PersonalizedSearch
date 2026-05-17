import pandas as pd
import numpy as np
import os
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class DataAnalyzer:
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def get_basic_stats(self) -> Dict:
        stats = {
            'total_records': len(self.data),
            'columns': list(self.data.columns),
            'missing_values': self.data.isnull().sum().to_dict(),
            'duplicate_count': self.data.duplicated().sum()
        }
        return stats
    
    def analyze_document_length(self) -> Dict:
        if 'content' not in self.data.columns:
            return {}
        
        lengths = self.data['content'].apply(lambda x: len(str(x)))
        return {
            'min_length': int(lengths.min()),
            'max_length': int(lengths.max()),
            'mean_length': float(lengths.mean()),
            'median_length': float(lengths.median()),
            'std_length': float(lengths.std()),
            'quantiles': {
                '25%': int(np.percentile(lengths, 25)),
                '50%': int(np.percentile(lengths, 50)),
                '75%': int(np.percentile(lengths, 75)),
                '90%': int(np.percentile(lengths, 90))
            }
        }
    
    def analyze_query_distribution(self) -> Dict:
        if 'Query' not in self.data.columns:
            return {}
        
        query_counts = self.data['Query'].value_counts()
        return {
            'unique_queries': len(query_counts),
            'most_common': query_counts.head(10).to_dict(),
            'query_frequency_distribution': {
                'appearing_once': (query_counts == 1).sum(),
                'appearing_2_5': ((query_counts >= 2) & (query_counts <= 5)).sum(),
                'appearing_6_10': ((query_counts >= 6) & (query_counts <= 10)).sum(),
                'appearing_more_than_10': (query_counts > 10).sum()
            }
        }
    
    def analyze_user_behavior(self) -> Dict:
        if 'AnonID' not in self.data.columns:
            return {}
        
        user_counts = self.data['AnonID'].value_counts()
        return {
            'unique_users': len(user_counts),
            'avg_queries_per_user': float(user_counts.mean()),
            'median_queries_per_user': float(user_counts.median()),
            'users_with_single_query': (user_counts == 1).sum(),
            'top_users_query_count': user_counts.head(10).to_dict()
        }
    
    def analyze_click_behavior(self) -> Dict:
        if 'ClickPos' not in self.data.columns:
            return {}
        
        click_positions = self.data['ClickPos'].dropna()
        return {
            'total_clicks': len(click_positions),
            'avg_click_position': float(click_positions.mean()),
            'median_click_position': float(click_positions.median()),
            'first_position_clicks': (click_positions == 1).sum(),
            'click_position_distribution': click_positions.value_counts().head(10).to_dict()
        }
    
    def generate_report(self, output_path: str) -> None:
        report = {
            'basic_stats': self.get_basic_stats(),
            'document_length': self.analyze_document_length(),
            'query_distribution': self.analyze_query_distribution(),
            'user_behavior': self.analyze_user_behavior(),
            'click_behavior': self.analyze_click_behavior()
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"分析报告已生成: {output_path}")
        
        self._print_report(report)
    
    def _print_report(self, report: Dict) -> None:
        print("\n" + "="*50)
        print("数据探索性分析报告")
        print("="*50)
        
        print("\n【基本统计】")
        stats = report['basic_stats']
        print(f"  总记录数: {stats['total_records']:,}")
        print(f"  列名: {', '.join(stats['columns'])}")
        print(f"  重复记录数: {stats['duplicate_count']}")
        
        if report['document_length']:
            print("\n【文档长度分析】")
            doc_len = report['document_length']
            print(f"  最短长度: {doc_len['min_length']}")
            print(f"  最长长度: {doc_len['max_length']}")
            print(f"  平均长度: {doc_len['mean_length']:.1f}")
            print(f"  中位数长度: {doc_len['median_length']}")
        
        if report['query_distribution']:
            print("\n【查询分布分析】")
            query_dist = report['query_distribution']
            print(f"  唯一查询数: {query_dist['unique_queries']:,}")
            print(f"  出现1次的查询: {query_dist['query_frequency_distribution']['appearing_once']:,}")
            print(f"  出现2-5次的查询: {query_dist['query_frequency_distribution']['appearing_2_5']:,}")
        
        if report['user_behavior']:
            print("\n【用户行为分析】")
            user_behavior = report['user_behavior']
            print(f"  唯一用户数: {user_behavior['unique_users']:,}")
            print(f"  平均每个用户查询数: {user_behavior['avg_queries_per_user']:.1f}")
        
        if report['click_behavior']:
            print("\n【点击行为分析】")
            click_behavior = report['click_behavior']
            print(f"  总点击数: {click_behavior['total_clicks']:,}")
            print(f"  平均点击位置: {click_behavior['avg_click_position']:.1f}")
        
        print("\n" + "="*50 + "\n")