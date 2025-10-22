from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class EquationClassifier(nn.Module):
    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 128):
        super(EquationClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, 64)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8)  # 8种方程类型: ODE1, ODE2, PDE, SDE, Algebraic, Integral, Delay, Other
        )
        
        self.character_vocab = self._build_character_vocab()
        
    def _build_character_vocab(self) -> Dict[str, int]:
        """构建字符级词汇表"""
        chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789∂Δ∇+-*/=^()[]{}.,;:!? ")
        # 添加数学符号
        math_symbols = set("αβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΦΨΩ")
        chars.update(math_symbols)
        return {char: idx for idx, char in enumerate(chars)}
    
    def preprocess_equation(self, equation: str) -> torch.Tensor:
        """预处理方程文本"""
        # 字符级编码
        encoded = [self.character_vocab.get(c, 0) for c in equation if c in self.character_vocab]
        if len(encoded) == 0:
            encoded = [0]
        # 填充或截断到固定长度
        max_length = 100
        if len(encoded) < max_length:
            encoded = encoded + [0] * (max_length - len(encoded))
        else:
            encoded = encoded[:max_length]
        return torch.tensor(encoded, dtype=torch.long)
    
    def forward(self, equation: str) -> Dict[str, torch.Tensor]:
        """分类方程类型"""
        tokens = self.preprocess_equation(equation)
        embedded = self.embedding(tokens).unsqueeze(0).transpose(1, 2)  # (1, 64, seq_len)
        features = self.conv_layers(embedded).squeeze(-1)  # (1, 128)
        logits = self.classifier(features)  # (1, 8)
        
        return {
            "type_logits": logits,
            "type_probs": F.softmax(logits, dim=-1),
            "predicted_type": torch.argmax(logits, dim=-1)
        }
    
    def analyze_equation(self, equation: str) -> Dict:
        """分析方程特性"""
        classification = self.forward(equation)
        
        # 规则辅助分析
        rule_based_info = self._rule_based_analysis(equation)
        
        return {
            **classification,
            **rule_based_info,
            "complexity": self._compute_complexity(equation),
            "linearity": self._detect_linearity(equation),
            "order": self._detect_order(equation),
            "dimensionality": self._detect_dimensionality(equation),
            "stiffness": self._detect_stiffness(equation)
        }
    
    def _rule_based_analysis(self, equation: str) -> Dict:
        """基于规则的方程分析"""
        equation_lower = equation.lower()
        
        analysis = {
            "has_partial_derivatives": '∂' in equation or 'd²' in equation or 'd^2' in equation_lower,
            "has_trig_functions": any(fn in equation_lower for fn in ['sin', 'cos', 'tan', 'exp', 'log']),
            "has_nonlinear_terms": any(op in equation_lower for op in ['^2', '^3', 'sin(', 'cos(', 'y*y', 'y**2']),
            "is_stochastic": any(term in equation_lower for term in ['dw', 'σ', 'sigma', 'random']),
            "has_boundary_conditions": any(bc in equation for bc in ['y(0)', 'x(0)', 'u(0)', 't=0']),
            "has_initial_conditions": '=' in equation and any(ic in equation for ic in ['(0)', 't=0']),
            "is_autonomous": 't' not in equation_lower.replace('tan', '').replace('atan', '') or equation_lower.count('t') == 1
        }
        
        return analysis
    
    def _compute_complexity(self, equation: str) -> float:
        """计算方程复杂度"""
        complexity = 0.0
        equation_lower = equation.lower()
        
        # 长度复杂度
        complexity += len(equation) * 0.05
        
        # 微分项复杂度
        complexity += equation.count('d') * 2.0
        complexity += equation.count('∂') * 3.0
        complexity += equation.count('∇') * 4.0
        
        # 函数复杂度
        functions = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt']
        for func in functions:
            complexity += equation_lower.count(func) * 1.5
            
        # 非线性复杂度
        complexity += equation_lower.count('^') * 2.0
        complexity += equation_lower.count('**') * 2.0
        
        # 条件复杂度
        if '=' in equation and '(' in equation:
            complexity += 3.0
            
        return min(complexity / 20.0, 1.0)  # 归一化到0-1
    
    def _detect_linearity(self, equation: str) -> str:
        """检测线性性"""
        equation_lower = equation.lower()
        
        # 检查非线性项
        nonlinear_indicators = [
            '^2', '^3', 'sin(', 'cos(', 'tan(', 'exp(', 'log(',
            'y*y', 'x*x', 'u*u', 'y**2', 'x**2', 'u**2'
        ]
        
        for indicator in nonlinear_indicators:
            if indicator in equation_lower:
                return "nonlinear"
                
        return "linear"
    
    def _detect_order(self, equation: str) -> int:
        """检测阶数"""
        equation_lower = equation.lower()
        
        if 'd³' in equation or "∂³" in equation or 'd^3' in equation_lower:
            return 3
        elif 'd²' in equation or "∂²" in equation or 'd^2' in equation_lower:
            return 2
        elif 'd' in equation or '∂' in equation:
            return 1
        return 0
    
    def _detect_dimensionality(self, equation: str) -> str:
        """检测维度"""
        if ('∂x' in equation and '∂y' in equation and '∂z' in equation) or \
           ('dx' in equation and 'dy' in equation and 'dz' in equation):
            return "3D"
        elif ('∂x' in equation and '∂y' in equation) or \
             ('dx' in equation and 'dy' in equation):
            return "2D"
        elif '∂t' in equation and '∂x' in equation:
            return "1D+time"
        elif 't' in equation.lower() and any(var in equation for var in ['x', 'y', 'u']):
            return "1D+time"
        return "1D"
    
    def _detect_stiffness(self, equation: str) -> float:
        """检测刚性程度（0-1）"""
        stiffness = 0.0
        equation_lower = equation.lower()
        
        # 快速变化项
        if any(term in equation_lower for term in ['exp(', 'e^', 'exp(-']):
            stiffness += 0.7
            
        # 大系数
        import re
        large_coeffs = re.findall(r'[0-9]+[0-9]*\*', equation)
        if large_coeffs:
            stiffness += 0.3
            
        # 多时间尺度
        if equation_lower.count('d') > 1:
            stiffness += 0.2
            
        return min(stiffness, 1.0)