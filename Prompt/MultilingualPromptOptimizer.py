from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.import_utils  # 这会自动设置sys.path
from .Language import Language
import numpy as np
import re
from typing import Dict

class MultilingualPromptOptimizer(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super(MultilingualPromptOptimizer, self).__init__()
        
        self.optimization_layers = nn.Sequential(
            nn.Linear(20 + 50, hidden_dim),  # RL动作(20) + 方程特征(50)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 8)  # 8种优化策略
        )
        
    def forward(self, base_prompt: str, rl_actions: torch.Tensor, 
                equation_info: Dict, language: Language) -> str:
        """优化prompt - 修复维度问题"""
        # 提取方程特征
        equation_features = self._extract_equation_features(equation_info)
        
        # 确保rl_actions是1维的
        if rl_actions.dim() > 1:
            rl_actions = rl_actions.squeeze()  # 从 [1, 20] 变为 [20]
        
        # 确保equation_features是1维的
        if equation_features.dim() == 0:
            equation_features = equation_features.unsqueeze(0)
        
        # 现在两个张量都是1维的，可以拼接
        combined_input = torch.cat([rl_actions, equation_features], dim=-1)
        
        # 生成优化策略
        optimization_strategies = self.optimization_layers(combined_input.unsqueeze(0))  # 添加batch维度
        
        # 应用优化
        optimized_prompt = self._apply_optimizations(
            base_prompt, optimization_strategies.squeeze(0), equation_info, language
        )
        
        return optimized_prompt
    
    def _extract_equation_features(self, equation_info: Dict) -> torch.Tensor:
        """提取方程特征"""
        features = []
        
        # 数值特征
        features.append(equation_info.get("complexity", 0.5))
        features.append(1.0 if equation_info.get("linearity") == "nonlinear" else 0.0)
        features.append(equation_info.get("order", 1) / 3.0)  # 归一化到0-1
        
        # 布尔特征
        bool_features = [
            "has_partial_derivatives", "has_trig_functions", 
            "has_nonlinear_terms", "is_stochastic", "has_boundary_conditions"
        ]
        
        for feature in bool_features:
            features.append(1.0 if equation_info.get(feature, False) else 0.0)
        
        # 确保有足够的特征
        while len(features) < 50:
            features.append(0.0)
        
        return torch.tensor(features[:50], dtype=torch.float32)
    
    def _apply_optimizations(self, prompt: str, strategies: torch.Tensor,
                           equation_info: Dict, language: Language) -> str:
        """应用优化策略"""
        optimized = prompt
        
        strategy_weights = F.softmax(strategies, dim=-1)
        
        # 详细程度优化
        if strategy_weights[0] > 0.3:
            optimized = self._optimize_detail_level(optimized, strategy_weights[0], language)
            
        # 技术深度优化
        if strategy_weights[1] > 0.3:
            optimized = self._optimize_technical_depth(optimized, strategy_weights[1], language)
            
        # 代码重点优化
        if strategy_weights[2] > 0.3:
            optimized = self._optimize_code_emphasis(optimized, strategy_weights[2], language)
            
        # 可视化优化
        if strategy_weights[3] > 0.3:
            optimized = self._optimize_visualization(optimized, strategy_weights[3], language)
            
        return optimized
    
    def _optimize_detail_level(self, prompt: str, weight: float, language: Language) -> str:
        """优化详细程度"""
        if language == Language.CHINESE:
            if weight > 0.7:
                prompt = prompt.replace("请提供", "请详细提供")
                prompt += "\n\n请确保每个步骤都有详细解释和数学推导过程。"
            elif weight < 0.3:
                prompt = re.sub(r'请提供.*?包含：', '请简要提供：', prompt)
                
        elif language == Language.ENGLISH:
            if weight > 0.7:
                prompt = prompt.replace("Please provide", "Please provide in detail")
                prompt += "\n\nPlease ensure each step has detailed explanations and mathematical derivations."
            elif weight < 0.3:
                prompt = re.sub(r'Please provide.*?including:', 'Please briefly provide:', prompt)
                
        elif language == Language.FRENCH:
            if weight > 0.7:
                prompt = prompt.replace("Veuillez fournir", "Veuillez fournir en détail")
                prompt += "\n\nVeuillez vous assurer que chaque étape comporte des explications détaillées et des dérivations mathématiques."
            elif weight < 0.3:
                prompt = re.sub(r'Veuillez fournir.*?incluant :', 'Veuillez fournir brièvement :', prompt)
                
        return prompt
    
    def _optimize_technical_depth(self, prompt: str, weight: float, language: Language) -> str:
        """优化技术深度"""
        if language == Language.CHINESE:
            if weight > 0.6:
                prompt += "\n\n请包含：数值稳定性分析、收敛性证明、误差估计等高级数值分析内容。"
                
        elif language == Language.ENGLISH:
            if weight > 0.6:
                prompt += "\n\nPlease include: numerical stability analysis, convergence proofs, error estimation, and other advanced numerical analysis content."
                
        elif language == Language.FRENCH:
            if weight > 0.6:
                prompt += "\n\nVeuillez inclure : analyse de stabilité numérique, preuves de convergence, estimation d'erreur et autres contenus d'analyse numérique avancée."
                
        return prompt
    
    def _optimize_code_emphasis(self, prompt: str, weight: float, language: Language) -> str:
        """优化代码重点"""
        if language == Language.CHINESE:
            if weight > 0.6:
                prompt = prompt.replace("Python代码实现", "完整、高效、注释详细的Python代码实现")
                prompt += "\n代码要求：模块化设计、参数可配置、包含单元测试和性能分析。"
                
        elif language == Language.ENGLISH:
            if weight > 0.6:
                prompt = prompt.replace("Python code implementation", "complete, efficient, and well-documented Python code implementation")
                prompt += "\nCode requirements: modular design, configurable parameters, including unit tests and performance analysis."
                
        elif language == Language.FRENCH:
            if weight > 0.6:
                prompt = prompt.replace("Implémentation du code Python", "Implémentation complète, efficace et bien documentée du code Python")
                prompt += "\nExigences du code : conception modulaire, paramètres configurables, incluant des tests unitaires et une analyse des performances."
                
        return prompt
    
    def _optimize_visualization(self, prompt: str, weight: float, language: Language) -> str:
        """优化可视化要求"""
        if language == Language.CHINESE:
            if weight > 0.5:
                prompt += "\n\n可视化要求：多子图布局、专业配色、清晰标注、结果对比分析。"
                
        elif language == Language.ENGLISH:
            if weight > 0.5:
                prompt += "\n\nVisualization requirements: multi-subplot layout, professional color schemes, clear labeling, comparative result analysis."
                
        elif language == Language.FRENCH:
            if weight > 0.5:
                prompt += "\n\nExigences de visualisation : disposition multi-graphiques, schémas de couleurs professionnels, étiquetage clair, analyse comparative des résultats."
                
        return prompt