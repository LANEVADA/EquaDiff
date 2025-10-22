import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import json
import utils.import_utils  # 这会自动设置sys.path
from Prompt.Language import Language
from .TextFeatureExtractor import TextFeatureExtractor
from .CodeQualityEvaluator import CodeQualityEvaluator
from .MathAccuracyEvaluator import MathAccuracyEvaluator
from .MultilingualEvaluationProcessor import MultilingualEvaluationProcessor


class ReportEvaluator(nn.Module):
    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256):
        super(ReportEvaluator, self).__init__()
        
        # 文本特征提取器
        self.text_encoder = TextFeatureExtractor(feature_dim)
        
        # 代码质量评估器
        self.code_evaluator = CodeQualityEvaluator()
        
        # 数学准确性评估器
        self.math_accuracy_evaluator = MathAccuracyEvaluator()
        
        # 综合评估网络
        self.evaluation_network = nn.Sequential(
            nn.Linear(feature_dim + 50, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 6)  # 6个评估维度
        )
        
        # 评估权重
        self.criteria_weights = nn.Parameter(torch.tensor([0.25, 0.20, 0.15, 0.15, 0.15, 0.10]))
        
        # 多语言支持
        self.multilingual_processor = MultilingualEvaluationProcessor()
        
    def forward(self, report_features: torch.Tensor, equation_features: torch.Tensor) -> torch.Tensor:
        """评估报告质量"""
        combined_features = torch.cat([report_features, equation_features], dim=-1)
        dimension_scores = torch.sigmoid(self.evaluation_network(combined_features))
        total_score = torch.sum(dimension_scores * self.criteria_weights)
        return total_score, dimension_scores
    
    def evaluate_report(self, report: str, equation: str, requirements: str, 
                       language: Language) -> Tuple[float, Dict]:
        """评估报告的综合质量 - 添加错误处理"""
        try:
            # 提取报告特征
            report_features = self.text_encoder.extract_features(report, language)
            
            # 提取方程特征
            equation_features = self._extract_equation_features(equation)
            
            # 神经网络评估
            report_tensor = torch.FloatTensor(report_features).unsqueeze(0)
            equation_tensor = torch.FloatTensor(equation_features).unsqueeze(0)
            total_score, dimension_scores = self.forward(report_tensor, equation_tensor)
            
            # 基于规则的补充评估
            rule_based_scores = self._rule_based_evaluation(report, equation, requirements, language)
            
            # 组合分数
            final_score = self._combine_scores(total_score.item(), rule_based_scores, dimension_scores[0])
            
            score_details = {
                "completeness": float(dimension_scores[0][0].item()),
                "accuracy": float(dimension_scores[0][1].item()),
                "code_quality": float(dimension_scores[0][2].item()),
                "clarity": float(dimension_scores[0][3].item()),
                "innovation": float(dimension_scores[0][4].item()),
                "practicality": float(dimension_scores[0][5].item()),
                "rule_based_scores": rule_based_scores,
                "final_score": final_score
            }
            
            return final_score, score_details
            
        except Exception as e:
            print(f"⚠️ 报告评估失败，使用默认分数: {e}")
            # 返回默认分数
            return 0.5, {
                "completeness": 0.5,
                "accuracy": 0.5,
                "code_quality": 0.5,
                "clarity": 0.5,
                "innovation": 0.5,
                "practicality": 0.5,
                "rule_based_scores": {},
                "final_score": 0.5
            }
    
    def _extract_equation_features(self, equation: str) -> np.ndarray:
        """提取方程特征"""
        features = []
        
        # 方程复杂度
        features.append(len(equation) / 100.0)
        features.append(equation.count('d') / 5.0)
        features.append(equation.count('∂') / 3.0)
        
        # 数学函数
        math_funcs = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt']
        for func in math_funcs:
            features.append(equation.count(func) / 2.0)
            
        # 填充到固定维度
        while len(features) < 50:
            features.append(0.0)
            
        return np.array(features[:50])
    
    def _rule_based_evaluation(self, report: str, equation: str, 
                             requirements: str, language: Language) -> Dict:
        """基于规则的评估 - 添加错误处理"""
        scores = {}
        
        try:
            # 1. 完整性检查
            scores["completeness"] = self._check_completeness(report, equation, language)
            
            # 2. 代码质量检查
            scores["code_quality"] = self.code_evaluator.evaluate_code_quality(report, language)
            
            # 3. 数学准确性
            scores["accuracy"] = self.math_accuracy_evaluator.evaluate_math_accuracy(report, equation, language)
            
            # 4. 需求满足度
            scores["requirements_met"] = self._check_requirements_satisfaction(report, requirements, language)
            
            # 5. 结构质量
            scores["structure"] = self._evaluate_structure(report, language)
            
        except Exception as e:
            print(f"⚠️ 规则评估失败: {e}")
            # 设置默认分数
            for key in ["completeness", "code_quality", "accuracy", "requirements_met", "structure"]:
                scores[key] = 0.5
        
        return scores
    
    def _check_completeness(self, report: str, equation: str, language: Language) -> float:
        """检查报告完整性"""
        completeness_score = 0.0
        
        # 检查必要部分
        required_sections = {
            Language.CHINESE: ["方程分析", "数值方法", "代码实现", "结果", "讨论"],
            Language.ENGLISH: ["equation analysis", "numerical method", "code", "results", "discussion"],
            Language.FRENCH: ["analyse", "méthode numérique", "code", "résultats", "discussion"]
        }
        
        sections = required_sections[language]
        found_sections = 0
        
        report_lower = report.lower()
        for section in sections:
            if section.lower() in report_lower:
                found_sections += 1
                
        completeness_score = found_sections / len(sections)
        
        # 检查代码块
        code_blocks = re.findall(r'```python.*?```', report, re.DOTALL)
        if code_blocks:
            completeness_score += 0.2
            
        # 检查可视化提及
        viz_keywords = {
            Language.CHINESE: ["可视化", "图表", "图形", "绘图"],
            Language.ENGLISH: ["visualization", "plot", "graph", "chart"],
            Language.FRENCH: ["visualisation", "graphique", "courbe", "diagramme"]
        }
        
        for keyword in viz_keywords[language]:
            if keyword in report.lower():
                completeness_score += 0.1
                break
                
        return min(completeness_score, 1.0)
    
    def _check_requirements_satisfaction(self, report: str, requirements: str, 
                                       language: Language) -> float:
        """检查需求满足度"""
        satisfaction = 0.0
        
        # 关键词匹配
        req_lower = requirements.lower()
        report_lower = report.lower()
        
        # 根据语言检测需求关键词
        requirement_keywords = self.multilingual_processor.extract_requirements_keywords(
            requirements, language
        )
        
        matched_keywords = 0
        for keyword in requirement_keywords:
            if keyword in report_lower:
                matched_keywords += 1
                
        if requirement_keywords:
            satisfaction = matched_keywords / len(requirement_keywords)
            
        return satisfaction
    
    def _evaluate_structure(self, report: str, language: Language) -> float:
        """评估报告结构"""
        structure_score = 0.0
        
        # 检查章节结构
        lines = report.split('\n')
        heading_count = sum(1 for line in lines if self._is_heading(line))
        
        if heading_count >= 3:  # 至少有3个标题
            structure_score += 0.3
            
        # 检查段落长度分布
        paragraphs = [p for p in report.split('\n\n') if p.strip()]
        if len(paragraphs) >= 5:  # 至少有5个段落
            structure_score += 0.3
            
        # 检查列表使用
        list_items = re.findall(r'\n\s*[-*•]\s+', report)
        if list_items:
            structure_score += 0.2
            
        # 检查代码块格式
        code_blocks = re.findall(r'```.*?```', report, re.DOTALL)
        if code_blocks:
            structure_score += 0.2
            
        return structure_score
    
    def _is_heading(self, line: str) -> bool:
        """判断是否为标题"""
        line = line.strip()
        if len(line) < 50 and (line.startswith('#') or line.endswith(':') or 
                              line.isupper() or line.replace(' ', '').isnumeric()):
            return True
        return False
    
    def _combine_scores(self, nn_score: float, rule_scores: Dict, 
                       dimension_scores: torch.Tensor) -> float:
        """组合神经网络分数和规则分数"""
        # 神经网络分数权重
        nn_weight = 0.6
        
        # 规则分数权重
        rule_weights = {
            "completeness": 0.15,
            "code_quality": 0.10,
            "accuracy": 0.08,
            "requirements_met": 0.05
        }
        
        rule_score = 0.0
        for key, weight in rule_weights.items():
            rule_score += rule_scores.get(key, 0) * weight
            
        # 最终分数
        final_score = nn_score * nn_weight + rule_score * (1 - nn_weight)
        
        return min(final_score, 1.0)