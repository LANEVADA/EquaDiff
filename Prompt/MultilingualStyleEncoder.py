from typing import Dict
import torch
import torch.nn as nn
import utils.import_utils  # 这会自动设置sys.path
from .Language import Language

class MultilingualStyleEncoder(nn.Module):
    def __init__(self, style_dim: int = 32):
        super(MultilingualStyleEncoder, self).__init__()
        self.style_dim = style_dim
        
        # 多语言风格嵌入
        self.style_embeddings = nn.Embedding(30, style_dim)  # 10风格 × 3语言
        
    def get_style_parameters(self, style_id: int, language: Language) -> torch.Tensor:
        """获取风格参数"""
        # 根据语言偏移风格ID
        lang_offset = {
            Language.CHINESE: 0,
            Language.ENGLISH: 10,
            Language.FRENCH: 20
        }[language]
        
        return self.style_embeddings(torch.tensor(style_id + lang_offset))
    
    def apply_multilingual_style(self, prompt: str, style_vector: torch.Tensor, 
                               language: Language) -> str:
        """应用多语言风格"""
        style_components = self._decode_style_vector(style_vector)
        
        if language == Language.CHINESE:
            return self._apply_chinese_style(prompt, style_components)
        elif language == Language.ENGLISH:
            return self._apply_english_style(prompt, style_components)
        elif language == Language.FRENCH:
            return self._apply_french_style(prompt, style_components)
        
        return prompt
    
    def _apply_chinese_style(self, prompt: str, style: Dict[str, float]) -> str:
        """应用中文字体"""
        if style["formality"] > 0.7:
            prompt = prompt.replace("请", "恳请")
            prompt = prompt.replace("分析", "深入分析")
            prompt = prompt.replace("提供", "详尽提供")
            
        if style["technicality"] > 0.8:
            prompt += "\n\n请使用专业术语和严格数学表述。"
            
        return prompt
    
    def _apply_english_style(self, prompt: str, style: Dict[str, float]) -> str:
        """应用英文字体"""
        if style["formality"] > 0.7:
            prompt = prompt.replace("Please", "We kindly request")
            prompt = prompt.replace("provide", "comprehensively provide")
            
        if style["technicality"] > 0.8:
            prompt += "\n\nPlease use professional terminology and rigorous mathematical expressions."
            
        return prompt
    
    def _apply_french_style(self, prompt: str, style: Dict[str, float]) -> str:
        """应用法文字体"""
        if style["formality"] > 0.7:
            prompt = prompt.replace("Veuillez", "Nous vous prions de bien vouloir")
            prompt = prompt.replace("fournir", "fournir de manière exhaustive")
            
        if style["technicality"] > 0.8:
            prompt += "\n\nVeuillez utiliser une terminologie professionnelle et des expressions mathématiques rigoureuses."
            
        return prompt
    
    def _decode_style_vector(self, style_vector: torch.Tensor) -> Dict[str, float]:
        """解码风格向量"""
        return {
            "formality": style_vector[0].item(),
            "technicality": style_vector[1].item(),
            "conciseness": style_vector[2].item()
        }