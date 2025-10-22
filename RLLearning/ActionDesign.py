import numpy as np
import re
from typing import Dict
import utils.import_utils  # 这会自动设置sys.path
from Prompt.Language import Language

class ActionSpaceDesign:
    """RL动作空间设计"""
    
    @staticmethod
    def decode_actions(action_vector: np.ndarray) -> Dict[str, float]:
        """解码动作向量为具体的优化参数"""
        return {
            'detail_level': action_vector[0],           # 详细程度 [-1, 1]
            'technical_depth': action_vector[1],        # 技术深度 [-1, 1]
            'code_emphasis': action_vector[2],          # 代码重点 [-1, 1]
            'visualization_focus': action_vector[3],    # 可视化重点 [-1, 1]
            'theory_emphasis': action_vector[4],        # 理论重点 [-1, 1]
            'validation_focus': action_vector[5],       # 验证重点 [-1, 1]
            'comparison_level': action_vector[6],       # 比较程度 [-1, 1]
            'educational_tone': action_vector[7],       # 教育语气 [-1, 1]
            'formality': action_vector[8],              # 正式程度 [-1, 1]
            'conciseness': action_vector[9],            # 简洁程度 [-1, 1]
            'example_density': action_vector[10],       # 例子密度 [-1, 1]
            'step_by_step': action_vector[11],          # 逐步指导 [-1, 1]
            'intuition_focus': action_vector[12],       # 直觉重点 [-1, 1]
            'rigor_level': action_vector[13],           # 严谨程度 [-1, 1]
            'application_focus': action_vector[14],     # 应用重点 [-1, 1]
            'innovation_emphasis': action_vector[15],   # 创新重点 [-1, 1]
            'practicality': action_vector[16],          # 实用性 [-1, 1]
            'completeness': action_vector[17],          # 完整性 [-1, 1]
            'clarity': action_vector[18],               # 清晰度 [-1, 1]
            'engagement': action_vector[19]             # 参与度 [-1, 1]
        }
    
    @staticmethod
    def apply_actions_to_prompt(prompt: str, actions: Dict[str, float], 
                              equation_info: Dict, language: Language) -> str:
        """将动作应用到prompt上"""
        optimized_prompt = prompt
        
        # 根据语言应用不同的优化策略
        if language == Language.CHINESE:
            optimized_prompt = ActionSpaceDesign._apply_chinese_optimizations(
                optimized_prompt, actions, equation_info
            )
        elif language == Language.ENGLISH:
            optimized_prompt = ActionSpaceDesign._apply_english_optimizations(
                optimized_prompt, actions, equation_info
            )
        elif language == Language.FRENCH:
            optimized_prompt = ActionSpaceDesign._apply_french_optimizations(
                optimized_prompt, actions, equation_info
            )
        
        return optimized_prompt
    
    @staticmethod
    def _apply_chinese_optimizations(prompt: str, actions: Dict[str, float], 
                                   equation_info: Dict) -> str:
        """应用中文化化"""
        # 详细程度优化
        if actions['detail_level'] > 0.5:
            prompt = prompt.replace("请提供", "请详细提供")
            prompt += "\n\n请确保每个步骤都有详细解释和数学推导。"
        elif actions['detail_level'] < -0.5:
            prompt = prompt.replace("请提供", "请简要提供")
            prompt = re.sub(r'包含：.*?(?=\n\n|\n$)', '包含核心内容：', prompt, flags=re.DOTALL)
        
        # 技术深度优化
        if actions['technical_depth'] > 0.5:
            prompt += "\n\n请包含：数值稳定性分析、收敛性证明、误差估计等高级内容。"
        
        # 代码重点优化
        if actions['code_emphasis'] > 0.5:
            prompt = prompt.replace("Python代码实现", "完整、高效、注释详细的Python代码实现")
            prompt += "\n代码要求：模块化设计、参数可配置、包含单元测试和性能分析。"
        
        return prompt
    
    @staticmethod
    def _apply_english_optimizations(prompt: str, actions: Dict[str, float],
                                  equation_info: Dict) -> str:
        """应用英文优化"""
        # Detail level optimization
        if actions['detail_level'] > 0.5:
            prompt = prompt.replace("Please provide", "Please provide in detail")
            prompt += "\n\nPlease ensure each step has detailed explanations and mathematical derivations."
        elif actions['detail_level'] < -0.5:
            prompt = prompt.replace("Please provide", "Please briefly provide")
            prompt = re.sub(r'including:.*?(?=\n\n|\n$)', 'including key elements:', prompt, flags=re.DOTALL)
        
        # Technical depth optimization
        if actions['technical_depth'] > 0.5:
            prompt += "\n\nPlease include: numerical stability analysis, convergence proofs, error estimation, etc."
        
        # Code emphasis optimization
        if actions['code_emphasis'] > 0.5:
            prompt = prompt.replace("Python code implementation", "complete, efficient, and well-documented Python code implementation")
            prompt += "\nCode requirements: modular design, configurable parameters, including unit tests and performance analysis."
        
        return prompt
    
    @staticmethod
    def _apply_french_optimizations(prompt: str, actions: Dict[str, float],
                                 equation_info: Dict) -> str:
        """应用法文化化"""
        # Niveau de détail optimization
        if actions['detail_level'] > 0.5:
            prompt = prompt.replace("Veuillez fournir", "Veuillez fournir en détail")
            prompt += "\n\nVeuillez vous assurer que chaque étape comporte des explications détaillées et des dérivations mathématiques."
        elif actions['detail_level'] < -0.5:
            prompt = prompt.replace("Veuillez fournir", "Veuillez fournir brièvement")
            prompt = re.sub(r'incluant :.*?(?=\n\n|\n$)', 'incluant les éléments clés :', prompt, flags=re.DOTALL)
        
        # Profondeur technique optimization
        if actions['technical_depth'] > 0.5:
            prompt += "\n\nVeuillez inclure : analyse de stabilité numérique, preuves de convergence, estimation d'erreur, etc."
        
        # Accent sur le code optimization
        if actions['code_emphasis'] > 0.5:
            prompt = prompt.replace("Implémentation du code Python", "Implémentation complète, efficace et bien documentée du code Python")
            prompt += "\nExigences du code : conception modulaire, paramètres configurables, incluant des tests unitaires et une analyse des performances."
        
        return prompt