import re
from typing import Dict, List, Tuple
from Prompt.Language import Language

class MathAccuracyEvaluator:
    def __init__(self):
        # 修复正则表达式 - 确保字符集正确终止
        self.math_patterns = {
            'derivative_notation': [
                r'd\w+/d\w+',           # dy/dx 格式
                r'∂\w+/∂\w+',           # ∂u/∂x 格式
                r'\\frac\{d\w+\}\{d\w+\}'  # \frac{dy}{dx} 格式
            ],
            'equation_patterns': [
                r'\\begin\{equation\}',  # LaTeX 方程环境
                r'\\\[',                 # LaTeX 显示数学
                r'\$\$',                 # LaTeX 显示数学
                r'\\('                   # LaTeX 行内数学
            ],
            'numerical_methods': ['euler', 'runge-kutta', 'finite difference', 'monte carlo']
        }
    
    def evaluate_math_accuracy(self, report: str, equation: str, language: Language) -> float:
        """评估数学准确性"""
        accuracy_score = 0.0
        
        # 1. 方程提及检查
        if self._check_equation_mention(report, equation):
            accuracy_score += 0.3
            
        # 2. 数学符号使用
        math_symbol_score = self._check_math_symbols(report)
        accuracy_score += math_symbol_score * 0.2
        
        # 3. 数值方法提及
        method_score = self._check_numerical_methods(report, language)
        accuracy_score += method_score * 0.3
        
        # 4. 误差分析
        error_analysis_score = self._check_error_analysis(report, language)
        accuracy_score += error_analysis_score * 0.2
        
        return min(accuracy_score, 1.0)
    
    def _check_equation_mention(self, report: str, equation: str) -> bool:
        """检查是否提及原方程"""
        # 简化检查：报告是否包含方程的关键部分
        # 移除空格和特殊字符进行简单匹配
        equation_simple = self._simplify_text(equation)
        report_simple = self._simplify_text(report)
        
        # 检查主要变量和运算符
        key_elements = re.findall(r'[a-zA-Z]+', equation)
        matches = sum(1 for elem in key_elements if elem.lower() in report_simple)
        
        return matches >= len(key_elements) * 0.5  # 至少匹配一半关键元素
    
    def _simplify_text(self, text: str) -> str:
        """简化文本用于匹配"""
        # 移除空格和数学符号
        simplified = re.sub(r'\s+', '', text.lower())
        simplified = re.sub(r'[^a-zA-Z0-9]', '', simplified)
        return simplified
    
    def _check_math_symbols(self, report: str) -> float:
        """检查数学符号使用 - 修复正则表达式问题"""
        symbol_count = 0
        
        for pattern in self.math_patterns['derivative_notation']:
            try:
                # 使用re.escape确保特殊字符被正确转义
                matches = re.findall(re.escape(pattern) if any(char in pattern for char in '\\[](){}^$*+?|') else pattern, report)
                symbol_count += len(matches)
            except re.error as e:
                print(f"⚠️ 正则表达式错误 '{pattern}': {e}")
                continue  # 跳过有问题的模式
            
        for pattern in self.math_patterns['equation_patterns']:
            try:
                matches = re.findall(re.escape(pattern), report)
                symbol_count += len(matches)
            except re.error as e:
                print(f"⚠️ 正则表达式错误 '{pattern}': {e}")
                continue
        
        # 添加基本的数学符号检查（不使用复杂正则）
        basic_symbols = ['∑', '∫', '∂', '∇', 'Δ', '±', '∞']
        for symbol in basic_symbols:
            symbol_count += report.count(symbol)
        
        return min(symbol_count / 5.0, 1.0)  # 最多5个符号得满分
    
    def _check_numerical_methods(self, report: str, language: Language) -> float:
        """检查数值方法提及"""
        method_keywords = {
            Language.CHINESE: ["欧拉法", "龙格库塔", "有限差分", "蒙特卡洛", "数值方法"],
            Language.ENGLISH: ["euler", "runge-kutta", "finite difference", "monte carlo", "numerical method"],
            Language.FRENCH: ["euler", "runge-kutta", "différence finie", "monte carlo", "méthode numérique"]
        }
        
        methods_found = 0
        report_lower = report.lower()
        
        for method in method_keywords[language]:
            if method.lower() in report_lower:
                methods_found += 1
                
        return methods_found / 3.0  # 最多3个方法得满分
    
    def _check_error_analysis(self, report: str, language: Language) -> float:
        """检查误差分析"""
        error_keywords = {
            Language.CHINESE: ["误差", "精度", "收敛", "稳定性", "误差分析"],
            Language.ENGLISH: ["error", "accuracy", "convergence", "stability", "error analysis"],
            Language.FRENCH: ["erreur", "précision", "convergence", "stabilité", "analyse d'erreur"]
        }
        
        error_terms_found = 0
        report_lower = report.lower()
        
        for term in error_keywords[language]:
            if term.lower() in report_lower:
                error_terms_found += 1
                
        return error_terms_found / 3.0  # 最多3个术语得满分
    
    def _safe_regex_search(self, pattern: str, text: str) -> int:
        """安全的正则表达式搜索，避免模式错误"""
        try:
            matches = re.findall(pattern, text)
            return len(matches)
        except re.error:
            # 如果正则表达式有问题，使用简单字符串搜索
            return text.count(pattern)