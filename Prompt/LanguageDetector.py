from typing import Dict
import torch.nn as nn
import utils.import_utils  # 这会自动设置sys.path
from .Language import Language

class LanguageDetector(nn.Module):
    def __init__(self, vocab_size: int = 500, hidden_dim: int = 64):
        super(LanguageDetector, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, 32)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 3种语言
        )
        
        # 构建多语言字符词汇表
        self.char_vocab = self._build_multilingual_vocab()
        
    def _build_multilingual_vocab(self) -> Dict[str, int]:
        """构建多语言字符词汇表"""
        # 中文常见字符
        chinese_chars = set("的一是不了在人有关与中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严"""
        )
        vocab = {}
        idx = 1
        
        # 添加中文字符
        for char in chinese_chars:
            vocab[char] = idx
            idx += 1
            
        # 添加英文字符
        for char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
            vocab[char] = idx
            idx += 1
            
        # 添加法语特殊字符
        french_chars = set("àâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ")
        for char in french_chars:
            vocab[char] = idx
            idx += 1
            
        # 添加数字和标点
        for char in "0123456789 .,;:!?()-+*/=∂Δ∇^[]{}":
            vocab[char] = idx
            idx += 1
            
        return vocab
    
    def detect(self, text: str) -> Language:
        """检测文本语言"""
        if len(text.strip()) == 0:
            return Language.CHINESE
            
        # 特征提取
        chinese_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        french_count = sum(1 for c in text if c in "àâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ")
        english_count = sum(1 for c in text if c.isascii() and c.isalpha())
        
        total_chars = len([c for c in text if c.isalpha() or '\u4e00' <= c <= '\u9fff'])
        
        if total_chars == 0:
            return Language.ENGLISH
            
        chinese_ratio = chinese_count / total_chars if total_chars > 0 else 0
        french_ratio = french_count / total_chars if total_chars > 0 else 0
        
        if chinese_ratio > 0.3:
            return Language.CHINESE
        elif french_ratio > 0.1:
            return Language.FRENCH
        else:
            return Language.ENGLISH
    