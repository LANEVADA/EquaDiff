import re
import torch.nn as nn
import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
import utils.import_utils  # 这会自动设置sys.path
from Prompt.Language import Language


class TextFeatureExtractor(nn.Module):
    def __init__(self, feature_dim: int = 512):
        super(TextFeatureExtractor, self).__init__()
        self.feature_dim = feature_dim
        
        # 简单的文本统计特征
        self.vectorizer = TfidfVectorizer(max_features=200, stop_words='english')
        
    def extract_features(self, text: str, language: Language) -> np.ndarray:
        """提取文本特征"""
        features = []
        
        # 1. 基础统计特征
        features.append(len(text) / 5000.0)  # 文本长度
        features.append(text.count('\n') / 50.0)  # 行数
        features.append(len(text.split()) / 1000.0)  # 词数
        
        # 2. 结构特征
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        features.append(len(paragraphs) / 20.0)  # 段落数
        
        # 3. 代码相关特征
        code_blocks = re.findall(r'```python.*?```', text, re.DOTALL)
        features.append(len(code_blocks) / 5.0)  # 代码块数量
        if code_blocks:
            avg_code_length = sum(len(block) for block in code_blocks) / len(code_blocks)
            features.append(avg_code_length / 1000.0)
        else:
            features.append(0.0)
            
        # 4. 数学内容特征
        math_keywords = self._get_math_keywords(language)
        for keyword in math_keywords:
            features.append(text.lower().count(keyword) / 10.0)
            
        # 5. 可视化特征
        viz_keywords = self._get_viz_keywords(language)
        viz_count = sum(text.lower().count(keyword) for keyword in viz_keywords)
        features.append(viz_count / 5.0)
        
        # 6. 质量指标
        features.append(self._calculate_readability(text, language))
        features.append(self._calculate_technical_depth(text, language))
        
        # 填充到固定维度
        while len(features) < self.feature_dim:
            features.append(0.0)
            
        return np.array(features[:self.feature_dim])
    
    def _get_math_keywords(self, language: Language) -> List[str]:
        """获取数学关键词"""
        keywords = {
            Language.CHINESE: [
                "微分", "积分", "方程", "数值", "收敛", "误差", "稳定性",
                "算法", "迭代", "离散", "连续", "导数", "偏导"
            ],
            Language.ENGLISH: [
                "derivative", "integral", "equation", "numerical", "convergence",
                "error", "stability", "algorithm", "iteration", "discrete",
                "continuous", "method", "analysis"
            ],
            Language.FRENCH: [
                "dérivée", "intégrale", "équation", "numérique", "convergence",
                "erreur", "stabilité", "algorithme", "itération", "discret",
                "continu", "méthode", "analyse"
            ]
        }
        return keywords[language]
    
    def _get_viz_keywords(self, language: Language) -> List[str]:
        """获取可视化关键词"""
        keywords = {
            Language.CHINESE: ["可视化", "图表", "图形", "绘图", "图像", "曲线"],
            Language.ENGLISH: ["visualization", "plot", "graph", "chart", "figure", "curve"],
            Language.FRENCH: ["visualisation", "graphique", "courbe", "diagramme", "figure"]
        }
        return keywords[language]
    
    def _calculate_readability(self, text: str, language: Language) -> float:
        """计算可读性分数"""
        # 简化版可读性计算
        sentences = re.split(r'[.!?。！？]+', text)
        words = text.split()
        
        if len(sentences) == 0 or len(words) == 0:
            return 0.5
            
        avg_sentence_length = len(words) / len(sentences)
        
        # 句子长度适中得分高
        if 10 <= avg_sentence_length <= 25:
            readability = 0.8
        elif 5 <= avg_sentence_length < 10 or 25 < avg_sentence_length <= 35:
            readability = 0.6
        else:
            readability = 0.4
            
        return readability
    
    def _calculate_technical_depth(self, text: str, language: Language) -> float:
        """计算技术深度"""
        technical_terms = {
            Language.CHINESE: ["收敛性", "稳定性", "误差分析", "数值方法", "离散化"],
            Language.ENGLISH: ["convergence", "stability", "error analysis", "numerical method", "discretization"],
            Language.FRENCH: ["convergence", "stabilité", "analyse d'erreur", "méthode numérique", "discrétisation"]
        }
        
        depth_score = 0.0
        for term in technical_terms[language]:
            if term in text:
                depth_score += 0.2
                
        return min(depth_score, 1.0)