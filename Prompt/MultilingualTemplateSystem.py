import torch.nn as nn
from typing import Dict
import utils.import_utils  # 这会自动设置sys.path
from .Language import Language

class MultilingualTemplateSystem(nn.Module):
    def __init__(self, template_dim: int = 64):
        super(MultilingualTemplateSystem, self).__init__()
        
        self.template_embeddings = nn.Embedding(60, template_dim)  # 20模板 × 3语言
        
        # 初始化多语言模板
        self.templates = self._initialize_multilingual_templates()
        self.current_language = Language.CHINESE
        
    def set_language(self, language: Language):
        """设置模板语言"""
        self.current_language = language
        
    def _initialize_multilingual_templates(self) -> Dict[Language, Dict[str, str]]:
        """初始化多语言模板"""
        return {
            Language.CHINESE: {
                "ode_standard": """
作为计算数学专家，请对以下常微分方程进行数值模拟和分析：

方程：{equation}
要求：{requirements}

请提供详细的数值分析报告，包含：
1. 方程分类和数学特性分析
2. 数值方法选择和理论依据
3. Python代码实现（完整可执行）
4. 数值结果的可视化展示
5. 误差分析和收敛性验证
6. 物理意义的解释和讨论
""",
                "pde_complex": """
作为偏微分方程数值分析专家，请解决以下问题：

偏微分方程：{equation}
模拟要求：{requirements}

请生成专业的技术报告，重点包括：
1. 方程类型和数学特性分析
2. 空间和时间离散化方法
3. 稳定性分析和收敛条件
4. 完整的数值实现代码
5. 多维结果可视化
6. 工程应用讨论
""",
                "educational": """
作为数学教育助手，请用易于理解的方式解释：

方程：{equation}
学习目标：{requirements}

请提供：
1. 直观的数学解释
2. 逐步的数值求解过程
3. 清晰的代码示例
4. 可视化结果
5. 学习要点总结
"""
            },
            
            Language.ENGLISH: {
                "ode_standard": """
As a computational mathematics expert, please perform numerical simulation and analysis for the following ordinary differential equation:

Equation: {equation}
Requirements: {requirements}

Please provide a detailed numerical analysis report including:
1. Equation classification and mathematical properties analysis
2. Numerical method selection and theoretical justification
3. Python code implementation (complete and executable)
4. Visualization of numerical results
5. Error analysis and convergence verification
6. Interpretation and discussion of physical significance
""",
                "pde_complex": """
As a partial differential equation numerical analysis expert, please solve the following problem:

PDE: {equation}
Simulation requirements: {requirements}

Please generate a professional technical report focusing on:
1. Equation type and mathematical characteristics analysis
2. Spatial and temporal discretization methods
3. Stability analysis and convergence conditions
4. Complete numerical implementation code
5. Multi-dimensional result visualization
6. Engineering application discussion
""",
                "educational": """
As a mathematics education assistant, please explain in an easy-to-understand manner:

Equation: {equation}
Learning objectives: {requirements}

Please provide:
1. Intuitive mathematical explanation
2. Step-by-step numerical solution process
3. Clear code examples
4. Visualization results
5. Learning points summary
"""
            },
            
            Language.FRENCH: {
                "ode_standard": """
En tant qu'expert en mathématiques computationnelles, veuillez effectuer une simulation numérique et une analyse de l'équation différentielle ordinaire suivante :

Équation : {equation}
Exigences : {requirements}

Veuillez fournir un rapport d'analyse numérique détaillé incluant :
1. Classification de l'équation et analyse des propriétés mathématiques
2. Sélection de la méthode numérique et justification théorique
3. Implémentation du code Python (complet et exécutable)
4. Visualisation des résultats numériques
5. Analyse d'erreur et vérification de la convergence
6. Interprétation et discussion de la signification physique
""",
                "pde_complex": """
En tant qu'expert en analyse numérique des équations aux dérivées partielles, veuillez résoudre le problème suivant :

EDP : {equation}
Exigences de simulation : {requirements}

Veuillez générer un rapport technique professionnel axé sur :
1. Analyse du type d'équation et des caractéristiques mathématiques
2. Méthodes de discrétisation spatiale et temporelle
3. Analyse de stabilité et conditions de convergence
4. Code complet d'implémentation numérique
5. Visualisation des résultats multidimensionnels
6. Discussion sur les applications techniques
""",
                "educational": """
En tant qu'assistant en éducation mathématique, veuillez expliquer de manière compréhensible :

Équation : {equation}
Objectifs d'apprentissage : {requirements}

Veuillez fournir :
1. Explication mathématique intuitive
2. Processus de solution numérique étape par étape
3. Exemples de code clairs
4. Résultats de visualisation
5. Résumé des points d'apprentissage
"""
            }
        }
    
    def generate_base_prompt(self, equation: str, requirements: str, 
                           equation_info: Dict, req_analysis: Dict) -> str:
        """生成基础prompt（多语言）"""
        # 选择模板
        template_key = self.select_template(equation_info, req_analysis)
        language_templates = self.templates[self.current_language]
        template = language_templates.get(template_key, language_templates["ode_standard"])
        
        # 填充模板
        prompt = template.format(
            equation=equation,
            requirements=requirements
        )
        
        # 添加语言特定的指导
        prompt += self._add_language_specific_guidance(equation_info, req_analysis)
        
        return prompt
    
    def _add_language_specific_guidance(self, equation_info: Dict, req_analysis: Dict) -> str:
        """添加语言特定的指导"""
        if self.current_language == Language.CHINESE:
            guidance = "\n\n特别注意：\n"
            if equation_info["linearity"] == "nonlinear":
                guidance += "- 注意非线性项的处理和稳定性问题\n"
            if equation_info["order"] == 2:
                guidance += "- 二阶方程需要转化为一阶系统求解\n"
                
        elif self.current_language == Language.ENGLISH:
            guidance = "\n\nSpecial attention:\n"
            if equation_info["linearity"] == "nonlinear":
                guidance += "- Pay attention to nonlinear term handling and stability issues\n"
            if equation_info["order"] == 2:
                guidance += "- Second-order equations need to be converted to first-order systems\n"
                
        elif self.current_language == Language.FRENCH:
            guidance = "\n\nAttention particulière :\n"
            if equation_info["linearity"] == "nonlinear":
                guidance += "- Attention au traitement des termes non linéaires et aux problèmes de stabilité\n"
            if equation_info["order"] == 2:
                guidance += "- Les équations du second ordre doivent être converties en systèmes du premier ordre\n"
                
        return guidance
    
    def select_template(self, equation_info: Dict, req_analysis: Dict) -> str:
        """选择模板（语言无关逻辑）"""
        complexity = equation_info["complexity"]
        detail_level = req_analysis.get("detail_level", "standard")
        
        if complexity > 0.7:
            if equation_info["has_partial_derivatives"]:
                return "pde_complex"
            else:
                return "ode_standard"
        elif detail_level == "educational":
            return "educational"
        else:
            return "ode_standard"