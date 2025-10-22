import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from typing import Dict, List, Tuple, Optional
import numpy as np
from enum import Enum
import utils.import_utils  # 这会自动设置sys.path
from .Language import Language
from .LanguageDetector import LanguageDetector
from .EquationClassifier import EquationClassifier
from .MultilingualTemplateSystem import MultilingualTemplateSystem
from .MultilingualStyleEncoder import MultilingualStyleEncoder
from .MultilingualPromptOptimizer import MultilingualPromptOptimizer
from .EquationFeatureExtractor import EquationFeatureExtractor

class MultilingualPromptGenerator(nn.Module):
    def __init__(self, template_dim: int = 64, style_dim: int = 32):
        super(MultilingualPromptGenerator, self).__init__()
        
        # 语言检测器
        self.language_detector = LanguageDetector()
        
        # 方程分类器
        self.equation_classifier = EquationClassifier()
        
        # 特征提取器
        self.feature_extractor = EquationFeatureExtractor()
        
        # 多语言模板系统
        self.template_system = MultilingualTemplateSystem(template_dim)
        
        # 多语言风格编码器
        self.style_encoder = MultilingualStyleEncoder(style_dim)
        
        # 优化器网络
        self.prompt_optimizer = MultilingualPromptOptimizer()
        
        # 当前语言设置
        self.current_language = Language.CHINESE
        
    def set_language(self, language: Language):
        """设置输出语言"""
        self.current_language = language
        self.template_system.set_language(language)
        
    def forward(self, equation: str, requirements: str, 
                rl_actions: Optional[torch.Tensor] = None,
                target_language: Optional[Language] = None) -> Dict[str, str]:
        """生成优化后的多语言prompt - 修复设备问题"""
        if target_language:
            self.set_language(target_language)
        
        # 检测输入语言
        input_lang = self.language_detector.detect(equation + " " + requirements)
        
        # 方程分析
        equation_info = self.analyze_equation(equation)
        
        # 需求解析
        req_analysis = self.analyze_requirements(requirements, input_lang)
        
        # 生成基础prompt
        base_prompt = self.template_system.generate_base_prompt(
            equation, requirements, equation_info, req_analysis
        )
        
        # 应用优化 - 确保rl_actions在CPU上
        if rl_actions is not None:
            # 确保rl_actions在CPU上
            if rl_actions.is_cuda:
                rl_actions = rl_actions.cpu()
            optimized_prompt = self.prompt_optimizer(
                base_prompt, rl_actions, equation_info, self.current_language
            )
        else:
            optimized_prompt = base_prompt
            
        return {
            "base_prompt": base_prompt,
            "optimized_prompt": optimized_prompt,
            "equation_info": equation_info,
            "requirements_analysis": req_analysis,
            "input_language": input_lang,
            "output_language": self.current_language
        }
    
    # ... 其他方法保持不变
    def analyze_equation(self, equation: str) -> Dict:
        """分析方程特性 - 集成方法"""
        # 使用EquationClassifier进行分析
        equation_analyzer = EquationClassifier()
        return equation_analyzer.analyze_equation(equation)
    def analyze_requirements(self, requirements: str, language: Language) -> Dict:
        """分析需求文本"""
        requirements_lower = requirements.lower()
        
        analysis = {
            "detail_level": "standard",
            "needs_validation": False,
            "needs_visualization": True,
            "needs_comparison": False,
            "needs_theory": False,
            "urgency": "normal"
        }
        
        # 根据语言选择关键词
        if language == Language.CHINESE:
            detail_keywords = {
                "high": ["详细", "深入", "全面", "完整", "彻底"],
                "low": ["简单", "简要", "概要", "快速"]
            }
            validation_keywords = ["验证", "检验", "测试", "误差分析"]
            visualization_keywords = ["可视化", "绘图", "图表", "图形"]
            comparison_keywords = ["比较", "对比", "不同方法"]
            theory_keywords = ["理论", "原理", "推导", "证明"]
            urgency_keywords = {"high": ["紧急", "尽快", "立即"], "low": ["不着急", "慢慢"]}
            
        elif language == Language.ENGLISH:
            detail_keywords = {
                "high": ["detailed", "comprehensive", "thorough", "complete", "in-depth"],
                "low": ["simple", "brief", "quick", "overview"]
            }
            validation_keywords = ["validation", "verification", "testing", "error analysis"]
            visualization_keywords = ["visualization", "plot", "graph", "chart"]
            comparison_keywords = ["compare", "comparison", "different methods"]
            theory_keywords = ["theory", "theoretical", "derivation", "proof"]
            urgency_keywords = {"high": ["urgent", "asap", "immediately"], "low": ["no hurry", "whenever"]}
            
        elif language == Language.FRENCH:
            detail_keywords = {
                "high": ["détaillé", "complet", "approfondi", "exhaustif"],
                "low": ["simple", "bref", "rapide", "aperçu"]
            }
            validation_keywords = ["validation", "vérification", "test", "analyse d'erreur"]
            visualization_keywords = ["visualisation", "graphique", "diagramme", "courbe"]
            comparison_keywords = ["comparer", "comparaison", "méthodes différentes"]
            theory_keywords = ["théorie", "théorique", "dérivation", "preuve"]
            urgency_keywords = {"high": ["urgent", "rapidement", "immédiatement"], "low": ["pas pressé", "quand"]}
        
        # 分析详细程度
        for high_word in detail_keywords["high"]:
            if high_word in requirements_lower:
                analysis["detail_level"] = "high"
                break
        else:
            for low_word in detail_keywords["low"]:
                if low_word in requirements_lower:
                    analysis["detail_level"] = "low"
                    break
        
        # 分析其他需求
        analysis["needs_validation"] = any(word in requirements_lower for word in validation_keywords)
        analysis["needs_visualization"] = any(word in requirements_lower for word in visualization_keywords)
        analysis["needs_comparison"] = any(word in requirements_lower for word in comparison_keywords)
        analysis["needs_theory"] = any(word in requirements_lower for word in theory_keywords)
        
        # 分析紧急程度
        for urgent_word in urgency_keywords["high"]:
            if urgent_word in requirements_lower:
                analysis["urgency"] = "high"
                break
        else:
            for low_urgent_word in urgency_keywords["low"]:
                if low_urgent_word in requirements_lower:
                    analysis["urgency"] = "low"
                    break
        
        return analysis
    
    # 测试多语言Prompt Generator
def test_multilingual_prompt_generator():
    generator = MultilingualPromptGenerator()
    
    test_cases = [
        {
            "equation": "dy/dt = -k*y, y(0) = 1",
            "requirements": "数值模拟和稳定性分析",
            "language": Language.CHINESE
        },
        {
            "equation": "∂u/∂t = α*∂²u/∂x²",
            "requirements": "numerical simulation and stability analysis", 
            "language": Language.ENGLISH
        },
        {
            "equation": "d²x/dt² + γ*dx/dt + ω²*x = 0",
            "requirements": "simulation numérique et analyse de stabilité",
            "language": Language.FRENCH
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*50}")
        print(f"测试案例 {i+1}: {test_case['language'].value}")
        print(f"方程: {test_case['equation']}")
        print(f"要求: {test_case['requirements']}")
        print(f"{'='*50}")
        
        result = generator(
            test_case["equation"],
            test_case["requirements"],
            target_language=test_case["language"]
        )
        
        print("生成的Prompt:")
        print(result["optimized_prompt"][:300] + "...")
        print(f"\n输入语言: {result['input_language'].value}")
        print(f"输出语言: {result['output_language'].value}")


# 交互式语言选择
def interactive_language_selection():
    generator = MultilingualPromptGenerator()
    
    print("请选择输出语言:")
    print("1. 中文")
    print("2. English") 
    print("3. Français")
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    language_map = {
        "1": Language.CHINESE,
        "2": Language.ENGLISH, 
        "3": Language.FRENCH
    }
    
    selected_language = language_map.get(choice, Language.CHINESE)
    generator.set_language(selected_language)
    
    equation = input("请输入微分方程: ")
    requirements = input("请输入要求: ")
    
    result = generator(equation, requirements)
    print(f"\n生成的{selected_language.value.upper()} Prompt:")
    print(result["optimized_prompt"])

if __name__ == "__main__":
    #test_multilingual_prompt_generator()
    interactive_language_selection()