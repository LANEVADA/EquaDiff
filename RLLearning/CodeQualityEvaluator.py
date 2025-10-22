import re
from typing import List
import numpy as np
import utils.import_utils  # 这会自动设置sys.path
from Prompt.Language import Language


class CodeQualityEvaluator:
    def __init__(self):
        self.quality_indicators = {
            'imports': ['numpy', 'scipy', 'matplotlib', 'import'],
            'functions': ['def ', 'function', 'lambda'],
            'comments': ['#', '"""', "'''"],
            'visualization': ['plt.', 'plot', 'show', 'figure'],
            'error_handling': ['try:', 'except', 'if', 'else']
        }
    
    def evaluate_code_quality(self, report: str, language: Language) -> float:
        """评估代码质量"""
        code_blocks = self._extract_code_blocks(report)
        if not code_blocks:
            return 0.3  # 没有代码的惩罚
            
        quality_scores = []
        for code in code_blocks:
            quality_scores.append(self._evaluate_single_code_block(code))
            
        return np.mean(quality_scores) if quality_scores else 0.3
    
    def _extract_code_blocks(self, report: str) -> List[str]:
        """提取代码块"""
        python_blocks = re.findall(r'```python\n(.*?)\n```', report, re.DOTALL)
        generic_blocks = re.findall(r'```\n(.*?)\n```', report, re.DOTALL)
        return python_blocks + generic_blocks
    
    def _evaluate_single_code_block(self, code: str) -> float:
        """评估单个代码块质量"""
        score = 0.0
        
        # 1. 基本结构检查
        if any(keyword in code for keyword in self.quality_indicators['imports']):
            score += 0.2
            
        if any(keyword in code for keyword in self.quality_indicators['functions']):
            score += 0.2
            
        # 2. 注释检查
        comment_lines = sum(1 for line in code.split('\n') if line.strip().startswith('#'))
        total_lines = len([line for line in code.split('\n') if line.strip()])
        if total_lines > 0:
            comment_ratio = comment_lines / total_lines
            if comment_ratio > 0.1:  # 至少10%的注释
                score += 0.2
                
        # 3. 代码长度适中
        code_length = len(code.strip())
        if 50 <= code_length <= 500:  # 合理的代码长度
            score += 0.2
        elif code_length > 500:  # 太长可能包含不必要内容
            score += 0.1
        else:  # 太短可能不完整
            score += 0.05
            
        # 4. 可视化相关
        if any(keyword in code for keyword in self.quality_indicators['visualization']):
            score += 0.2
            
        # 5. 错误处理
        if any(keyword in code for keyword in self.quality_indicators['error_handling']):
            score += 0.1
            
        return min(score, 1.0)