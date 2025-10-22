import re
from typing import Dict, List
import utils.import_utils  # 这会自动设置sys.path
from Prompt.Language import Language

class MultilingualEvaluationProcessor:
    def __init__(self):
        # 多语言关键词库
        self.keyword_libraries = {
            Language.CHINESE: self._build_chinese_keywords(),
            Language.ENGLISH: self._build_english_keywords(),
            Language.FRENCH: self._build_french_keywords()
        }
        
        # 语言特定的评分标准
        self.language_standards = {
            Language.CHINESE: {"strictness": 0.9, "formality_weight": 0.7},
            Language.ENGLISH: {"strictness": 0.8, "formality_weight": 0.6},
            Language.FRENCH: {"strictness": 0.85, "formality_weight": 0.75}
        }
    
    def _build_chinese_keywords(self) -> Dict[str, List[str]]:
        """构建中文关键词库"""
        return {
            "completeness": [
                "方程分析", "数值方法", "代码实现", "结果展示", "误差分析", 
                "稳定性", "收敛性", "讨论", "结论", "可视化"
            ],
            "accuracy": [
                "微分", "积分", "导数", "偏导", "解析解", "数值解", 
                "欧拉法", "龙格库塔", "有限差分", "边界条件", "初始条件"
            ],
            "code_quality": [
                "import numpy", "import matplotlib", "def ", "plt.plot", 
                "np.linspace", "for循环", "函数定义", "注释", "参数"
            ],
            "clarity": [
                "清晰", "易懂", "结构", "章节", "标题", "段落", 
                "逻辑", "表述", "解释", "说明"
            ],
            "innovation": [
                "创新", "改进", "优化", "新方法", "比较", "分析", 
                "洞察", "发现", "建议", "应用"
            ]
        }
    
    def _build_english_keywords(self) -> Dict[str, List[str]]:
        """构建英文关键词库"""
        return {
            "completeness": [
                "equation analysis", "numerical method", "code implementation", 
                "results", "error analysis", "stability", "convergence", 
                "discussion", "conclusion", "visualization"
            ],
            "accuracy": [
                "derivative", "integral", "differential", "partial", 
                "analytical solution", "numerical solution", "euler method",
                "runge-kutta", "finite difference", "boundary condition"
            ],
            "code_quality": [
                "import numpy", "import matplotlib", "def ", "plt.plot", 
                "np.linspace", "for loop", "function definition", "comment",
                "parameter", "algorithm"
            ],
            "clarity": [
                "clear", "understandable", "structure", "section", "heading",
                "paragraph", "logical", "explanation", "description"
            ],
            "innovation": [
                "innovation", "improvement", "optimization", "new method",
                "comparison", "analysis", "insight", "finding", "suggestion"
            ]
        }
    
    def _build_french_keywords(self) -> Dict[str, List[str]]:
        """构建法文关键词库"""
        return {
            "completeness": [
                "analyse d'équation", "méthode numérique", "implémentation du code",
                "résultats", "analyse d'erreur", "stabilité", "convergence",
                "discussion", "conclusion", "visualisation"
            ],
            "accuracy": [
                "dérivée", "intégrale", "différentiel", "partiel",
                "solution analytique", "solution numérique", "méthode d'euler",
                "runge-kutta", "différence finie", "condition aux limites"
            ],
            "code_quality": [
                "import numpy", "import matplotlib", "def ", "plt.plot",
                "np.linspace", "boucle for", "définition de fonction", "commentaire",
                "paramètre", "algorithme"
            ],
            "clarity": [
                "clair", "compréhensible", "structure", "section", "titre",
                "paragraphe", "logique", "explication", "description"
            ],
            "innovation": [
                "innovation", "amélioration", "optimisation", "nouvelle méthode",
                "comparaison", "analyse", "perspicacité", "découverte", "suggestion"
            ]
        }
    def extract_requirements_keywords(self, requirements: str, language: Language) -> List[str]:
        """提取需求关键词 - 添加缺失的方法"""
        keywords = []
        req_lower = requirements.lower()
        
        patterns = self.requirement_patterns[language]
        for category, terms in patterns.items():
            for term in terms:
                if term.lower() in req_lower:
                    keywords.append(term)
                    break  # 每个类别只取一个关键词
                    
        return keywords
    def extract_keywords_by_dimension(self, text: str, dimension: str, language: Language) -> List[str]:
        """按维度提取关键词"""
        keywords = self.keyword_libraries[language].get(dimension, [])
        found_keywords = []
        
        text_lower = text.lower()
        for keyword in keywords:
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)
                
        return found_keywords
    
    def calculate_keyword_coverage(self, text: str, dimension: str, language: Language) -> float:
        """计算关键词覆盖率"""
        expected_keywords = self.keyword_libraries[language].get(dimension, [])
        if not expected_keywords:
            return 0.0
            
        found_count = 0
        text_lower = text.lower()
        for keyword in expected_keywords:
            if keyword.lower() in text_lower:
                found_count += 1
                
        return found_count / len(expected_keywords)
    
    def detect_language_specific_patterns(self, text: str, language: Language) -> Dict[str, float]:
        """检测语言特定模式"""
        patterns = {
            Language.CHINESE: {
                "formality": self._detect_chinese_formality(text),
                "technical_depth": self._detect_chinese_technical_depth(text),
                "structure_quality": self._detect_chinese_structure(text)
            },
            Language.ENGLISH: {
                "formality": self._detect_english_formality(text),
                "technical_depth": self._detect_english_technical_depth(text),
                "structure_quality": self._detect_english_structure(text)
            },
            Language.FRENCH: {
                "formality": self._detect_french_formality(text),
                "technical_depth": self._detect_french_technical_depth(text),
                "structure_quality": self._detect_french_structure(text)
            }
        }
        return patterns[language]
    
    def _detect_chinese_formality(self, text: str) -> float:
        """检测中文正式程度"""
        formal_indicators = ["本文", "所述", "据此", "综上所述", "恳请", "谨此"]
        informal_indicators = ["我觉得", "大概", "可能", "随便", "简单"]
        
        formal_count = sum(1 for indicator in formal_indicators if indicator in text)
        informal_count = sum(1 for indicator in informal_indicators if indicator in text)
        
        total_indicators = formal_count + informal_count
        if total_indicators == 0:
            return 0.5
            
        return formal_count / total_indicators
    
    def _detect_english_formality(self, text: str) -> float:
        """检测英文正式程度"""
        formal_indicators = ["therefore", "however", "moreover", "furthermore", "consequently"]
        informal_indicators = ["I think", "maybe", "probably", "just", "simple"]
        
        text_lower = text.lower()
        formal_count = sum(1 for indicator in formal_indicators if indicator in text_lower)
        informal_count = sum(1 for indicator in informal_indicators if indicator in text_lower)
        
        total_indicators = formal_count + informal_count
        if total_indicators == 0:
            return 0.5
            
        return formal_count / total_indicators
    
    def _detect_french_formality(self, text: str) -> float:
        """检测法文正式程度"""
        formal_indicators = ["par conséquent", "cependant", "de plus", "en outre", "ainsi"]
        informal_indicators = ["je pense", "peut-être", "probablement", "juste", "simple"]
        
        text_lower = text.lower()
        formal_count = sum(1 for indicator in formal_indicators if indicator in text_lower)
        informal_count = sum(1 for indicator in informal_indicators if indicator in text_lower)
        
        total_indicators = formal_count + informal_count
        if total_indicators == 0:
            return 0.5
            
        return formal_count / total_indicators
    
    def _detect_chinese_technical_depth(self, text: str) -> float:
        """检测中文技术深度"""
        advanced_terms = ["收敛性分析", "稳定性证明", "误差估计", "数值稳定性", "离散化误差"]
        found_terms = sum(1 for term in advanced_terms if term in text)
        return min(found_terms / 3.0, 1.0)
    
    def _detect_english_technical_depth(self, text: str) -> float:
        """检测英文技术深度"""
        advanced_terms = ["convergence analysis", "stability proof", "error estimation", 
                         "numerical stability", "discretization error"]
        text_lower = text.lower()
        found_terms = sum(1 for term in advanced_terms if term in text_lower)
        return min(found_terms / 3.0, 1.0)
    
    def _detect_french_technical_depth(self, text: str) -> float:
        """检测法文技术深度"""
        advanced_terms = ["analyse de convergence", "preuve de stabilité", "estimation d'erreur",
                         "stabilité numérique", "erreur de discrétisation"]
        text_lower = text.lower()
        found_terms = sum(1 for term in advanced_terms if term in text_lower)
        return min(found_terms / 3.0, 1.0)
    
    def _detect_chinese_structure(self, text: str) -> float:
        """检测中文结构质量"""
        score = 0.0
        lines = text.split('\n')
        
        # 检查标题
        heading_patterns = ['# ', '## ', '### ', '#### ', '标题', '章节']
        heading_count = sum(1 for line in lines if any(pattern in line for pattern in heading_patterns))
        if heading_count >= 3:
            score += 0.4
            
        # 检查段落
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) >= 5:
            score += 0.3
            
        # 检查列表
        list_items = re.findall(r'\n\s*[•·\-*]\s+', text)
        if list_items:
            score += 0.3
            
        return score
    
    def _detect_english_structure(self, text: str) -> float:
        """检测英文结构质量"""
        score = 0.0
        lines = text.split('\n')
        
        # 检查标题
        heading_patterns = ['# ', '## ', '### ', '#### ', 'section', 'chapter']
        heading_count = sum(1 for line in lines if any(pattern in line for pattern in heading_patterns))
        if heading_count >= 3:
            score += 0.4
            
        # 检查段落
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) >= 5:
            score += 0.3
            
        # 检查列表
        list_items = re.findall(r'\n\s*[•\-*]\s+', text)
        if list_items:
            score += 0.3
            
        return score
    
    def _detect_french_structure(self, text: str) -> float:
        """检测法文结构质量"""
        score = 0.0
        lines = text.split('\n')
        
        # 检查标题
        heading_patterns = ['# ', '## ', '### ', '#### ', 'section', 'chapitre']
        heading_count = sum(1 for line in lines if any(pattern in line for pattern in heading_patterns))
        if heading_count >= 3:
            score += 0.4
            
        # 检查段落
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) >= 5:
            score += 0.3
            
        # 检查列表
        list_items = re.findall(r'\n\s*[•\-*]\s+', text)
        if list_items:
            score += 0.3
            
        return score
    
    def adjust_score_for_language(self, score: float, dimension: str, language: Language) -> float:
        """根据语言调整分数"""
        standards = self.language_standards[language]
        
        # 不同维度可能有不同的调整
        if dimension == "accuracy":
            # 准确性在所有语言中都很重要
            return score
        elif dimension == "clarity":
            # 清晰度根据语言正式程度调整
            return score * standards["formality_weight"]
        else:
            # 其他维度根据严格程度调整
            return score * standards["strictness"]