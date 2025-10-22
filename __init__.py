"""
Differential Equation AI Agent
自动化微分方程求解和报告生成的AI系统
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .Prompt import *
from .RLLearning import *

__all__ = [
    # 从prompt模块导出
    'Language',
    'MultilingualPromptGenerator',
    'EquationClassifier',
    # 从rl_learning模块导出  
    'PPOAgent',
    'ReportEvaluator',
    'PromptOptimizationSystem'
]