import torch
import torch.nn as nn

class EquationFeatureExtractor(nn.Module):
    def __init__(self, feature_dim: int = 50):
        super(EquationFeatureExtractor, self).__init__()
        self.feature_dim = feature_dim
        
    def forward(self, equation: str) -> torch.Tensor:
        """提取方程特征向量"""
        return self.extract_features(equation)
    
    def extract_features(self, equation: str) -> torch.Tensor:
        """提取方程特征向量"""
        features = []
        equation_lower = equation.lower()
        
        # 1. 基础统计特征
        features.append(len(equation) / 200.0)  # 归一化长度
        
        # 2. 数学符号特征
        math_symbols = ['d', '∂', '∫', '∇', 'Δ', '∑', '∏', '±']
        for symbol in math_symbols:
            features.append(min(equation.count(symbol) / 5.0, 1.0))
            
        # 3. 函数特征
        functions = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'arctan']
        for func in functions:
            features.append(min(equation_lower.count(func) / 3.0, 1.0))
            
        # 4. 运算符特征
        operators = ['+', '-', '*', '/', '^', '=', '**']
        for op in operators:
            features.append(min(equation.count(op) / 10.0, 1.0))
            
        # 5. 变量特征
        variables = ['x', 'y', 'z', 't', 'u', 'v', 'w']
        for var in variables:
            features.append(min(equation_lower.count(var) / 5.0, 1.0))
            
        # 6. 条件特征
        features.append(1.0 if any(bc in equation for bc in ['y(0)', 'x(0)', 'u(0)', 't=0']) else 0.0)
        features.append(1.0 if 't' in equation_lower and any(var in equation for var in ['x', 'y', 'u']) else 0.0)
        features.append(1.0 if '=' in equation and '(' in equation else 0.0)
        
        # 7. 复杂度特征
        features.append(self._compute_simple_complexity(equation))
        
        # 8. 特殊类型特征
        features.append(1.0 if '∂' in equation else 0.0)  # 偏微分
        features.append(1.0 if 'd²' in equation or 'd^2' in equation_lower else 0.0)  # 二阶
        features.append(1.0 if any(nl in equation_lower for nl in ['^2', '^3', 'sin(', 'cos(']) else 0.0)  # 非线性
        
        # 填充到固定维度
        while len(features) < self.feature_dim:
            features.append(0.0)
            
        return torch.tensor(features[:self.feature_dim], dtype=torch.float32)
    
    def _compute_simple_complexity(self, equation: str) -> float:
        """计算简化复杂度"""
        complexity = len(equation) * 0.1
        complexity += equation.count('d') * 2
        complexity += equation.count('∂') * 3
        complexity += equation.count('sin') * 1.5
        complexity += equation.count('cos') * 1.5
        complexity += equation.count('^') * 2
        return min(complexity / 30.0, 1.0)