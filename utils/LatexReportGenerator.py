import os
import subprocess
import tempfile
from typing import Dict, List, Optional
from datetime import datetime
from ..Prompt.Language import Language




class LatexReportGenerator:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.figure_counter = 0
        print(f"📁 LaTeX临时目录: {self.temp_dir}")
        
    def generate_latex_report(self, report_content: str, equation: str, 
                            language: Language, output_path: str = None) -> str:
        """生成完整的LaTeX报告 - 修复文件生成问题"""
        
        try:
            # 解析报告内容
            sections = self._parse_report_sections(report_content)
            
            # 生成LaTeX代码
            latex_code = self._build_latex_document(sections, equation, language)
            
            # 确保输出目录存在
            if output_path is None:
                output_dir = "reports"
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = os.path.join(output_dir, f"report_{timestamp}")
            else:
                output_dir = os.path.dirname(output_path)
                os.makedirs(output_dir, exist_ok=True)
            
            # 保存LaTeX文件
            tex_file = f"{output_path}.tex"
            print(f"📄 保存LaTeX文件: {tex_file}")
            
            with open(tex_file, 'w', encoding='utf-8') as f:
                f.write(latex_code)
            
            # 编译LaTeX
            print("🔄 编译LaTeX文件...")
            success = self._compile_latex(tex_file, output_path)
            
            if success and os.path.exists(f"{output_path}.pdf"):
                pdf_path = f"{output_path}.pdf"
                print(f"✅ PDF报告生成成功: {pdf_path}")
                return pdf_path
            else:
                print("⚠️ PDF生成失败，返回tex文件")
                return tex_file
                
        except Exception as e:
            print(f"❌ LaTeX报告生成失败: {e}")
            # 创建简单的文本报告作为fallback
            return self._create_text_fallback(report_content, equation, output_path)
    
    def _create_text_fallback(self, report_content: str, equation: str, output_path: str) -> str:
        """创建文本格式的fallback报告"""
        try:
            text_file = f"{output_path}.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(f"微分方程数值分析报告\n")
                f.write("=" * 50 + "\n")
                f.write(f"方程: {equation}\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
                f.write(report_content)
            
            print(f"📝 生成文本报告: {text_file}")
            return text_file
        except Exception as e:
            print(f"❌ 文本报告生成也失败: {e}")
            return "report_failed.txt"
    
    def _parse_report_sections(self, report_content: str) -> Dict[str, str]:
        """解析报告内容为各个章节"""
        sections = {
            "analysis": "",
            "method": "", 
            "code": "",
            "results": "",
            "discussion": ""
        }
        
        lines = report_content.split('\n')
        current_section = "analysis"
        
        for line in lines:
            line = line.strip()
            
            # 检测章节标题
            if self._is_section_header(line):
                section_name = self._extract_section_name(line)
                if section_name in sections:
                    current_section = section_name
                continue
                
            # 添加到当前章节
            if line and not line.startswith('```'):
                sections[current_section] += line + '\n'
        
        return sections
    
    def _is_section_header(self, line: str) -> bool:
        """判断是否为章节标题"""
        header_indicators = ['#', '##', '===', '---', '方程分析', '数值方法', 
                           '代码实现', '结果', '讨论', 'analysis', 'method', 
                           'code', 'results', 'discussion']
        return any(indicator in line for indicator in header_indicators)
    
    def _extract_section_name(self, line: str) -> str:
        """提取章节名称"""
        line_lower = line.lower()
        
        if any(word in line_lower for word in ['方程分析', 'equation', 'analysis']):
            return "analysis"
        elif any(word in line_lower for word in ['数值方法', 'method', '算法']):
            return "method" 
        elif any(word in line_lower for word in ['代码', 'code', '实现']):
            return "code"
        elif any(word in line_lower for word in ['结果', 'results', 'result']):
            return "results"
        elif any(word in line_lower for word in ['讨论', 'discussion', '结论']):
            return "discussion"
        else:
            return "analysis"
    
    def _build_latex_document(self, sections: Dict[str, str], equation: str, 
                            language: Language) -> str:
        """构建完整的LaTeX文档"""
        
        # 根据语言选择模板
        if language == Language.CHINESE:
            return self._chinese_latex_template(sections, equation)
        elif language == Language.ENGLISH:
            return self._english_latex_template(sections, equation)
        elif language == Language.FRENCH:
            return self._french_latex_template(sections, equation)
    
    def _chinese_latex_template(self, sections: Dict[str, str], equation: str) -> str:
        """中文LaTeX模板"""
        return f"""\\documentclass[12pt]{{article}}
\\usepackage[UTF8]{{ctex}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\usepackage{{graphicx}}
\\usepackage{{listings}}
\\usepackage{{xcolor}}
\\usepackage{{geometry}}
\\geometry{{a4paper, margin=2.5cm}}

\\title{{微分方程数值分析报告}}
\\author{{AI 数学助手}}
\\date{{\\today}}

\\lstset{{
    language=Python,
    basicstyle=\\ttfamily\\small,
    keywordstyle=\\color{{blue}},
    commentstyle=\\color{{green}},
    stringstyle=\\color{{red}},
    numbers=left,
    numberstyle=\\tiny\\color{{gray}},
    stepnumber=1,
    numbersep=5pt,
    backgroundcolor=\\color{{white}},
    showspaces=false,
    showstringspaces=false,
    showtabs=false,
    frame=single,
    tabsize=4,
    captionpos=b,
    breaklines=true,
    breakatwhitespace=false,
    escapeinside={{/*@}}{{@*/}}
}}

\\begin{{document}}

\\maketitle

\\section{{问题描述}}
\\subsection{{微分方程}}
\\begin{{equation}}
    {self._escape_latex(equation)}
\\end{{equation}}

\\section{{方程分析}}
{sections['analysis']}

\\section{{数值方法}}
{sections['method']}

\\section{{代码实现}}
{sections['code']}

\\section{{数值结果}}
{sections['results']}

\\section{{讨论与分析}}
{sections['discussion']}

\\end{{document}}
"""
    
    def _english_latex_template(self, sections: Dict[str, str], equation: str) -> str:
        """英文LaTeX模板"""
        return f"""\\documentclass[12pt]{{article}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\usepackage{{graphicx}}
\\usepackage{{listings}}
\\usepackage{{xcolor}}
\\usepackage{{geometry}}
\\geometry{{a4paper, margin=2.5cm}}

\\title{{Numerical Analysis of Differential Equations}}
\\author{{AI Math Assistant}}
\\date{{\\today}}

\\lstset{{
    language=Python,
    basicstyle=\\ttfamily\\small,
    keywordstyle=\\color{{blue}},
    commentstyle=\\color{{green}},
    stringstyle=\\color{{red}},
    numbers=left,
    numberstyle=\\tiny\\color{{gray}},
    stepnumber=1,
    numbersep=5pt,
    backgroundcolor=\\color{{white}},
    showspaces=false,
    showstringspaces=false,
    showtabs=false,
    frame=single,
    tabsize=4,
    captionpos=b,
    breaklines=true,
    breakatwhitespace=false,
    escapeinside={{/*@}}{{@*/}}
}}

\\begin{{document}}

\\maketitle

\\section{{Problem Description}}
\\subsection{{Differential Equation}}
\\begin{{equation}}
    {self._escape_latex(equation)}
\\end{{equation}}

\\section{{Equation Analysis}}
{sections['analysis']}

\\section{{Numerical Method}}
{sections['method']}

\\section{{Code Implementation}}
{sections['code']}

\\section{{Numerical Results}}
{sections['results']}

\\section{{Discussion and Analysis}}
{sections['discussion']}

\\end{{document}}
"""
    
    def _french_latex_template(self, sections: Dict[str, str], equation: str) -> str:
        """法文LaTeX模板"""
        return f"""\\documentclass[12pt]{{article}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\usepackage{{graphicx}}
\\usepackage{{listings}}
\\usepackage{{xcolor}}
\\usepackage{{geometry}}
\\geometry{{a4paper, margin=2.5cm}}

\\title{{Analyse Numérique des Équations Différentielles}}
\\author{{Assistant Mathématique IA}}
\\date{{\\today}}

\\lstset{{
    language=Python,
    basicstyle=\\ttfamily\\small,
    keywordstyle=\\color{{blue}},
    commentstyle=\\color{{green}},
    stringstyle=\\color{{red}},
    numbers=left,
    numberstyle=\\tiny\\color{{gray}},
    stepnumber=1,
    numbersep=5pt,
    backgroundcolor=\\color{{white}},
    showspaces=false,
    showstringspaces=false,
    showtabs=false,
    frame=single,
    tabsize=4,
    captionpos=b,
    breaklines=true,
    breakatwhitespace=false,
    escapeinside={{/*@}}{{@*/}}
}}

\\begin{{document}}

\\maketitle

\\section{{Description du Problème}}
\\subsection{{Équation Différentielle}}
\\begin{{equation}}
    {self._escape_latex(equation)}
\\end{{equation}}

\\section{{Analyse de l'Équation}}
{sections['analysis']}

\\section{{Méthode Numérique}}
{sections['method']}

\\section{{Implémentation du Code}}
{sections['code']}

\\section{{Résultats Numériques}}
{sections['results']}

\\section{{Discussion et Analyse}}
{sections['discussion']}

\\end{{document}}
"""
    
    def _escape_latex(self, text: str) -> str:
        """转义LaTeX特殊字符"""
        escape_chars = {
            '&': r'\&',
            '%': r'\%', 
            '$': r'\$',
            '#': r'\#',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}',
            '^': r'\textasciicircum{}',
            '\\': r'\textbackslash{}'
        }
        
        for char, replacement in escape_chars.items():
            text = text.replace(char, replacement)
            
        return text
    
    def _compile_latex(self, tex_file: str, output_path: str) -> bool:
        """编译LaTeX文件"""
        try:
            output_dir = os.path.dirname(output_path)
            tex_filename = os.path.basename(tex_file)
            
            print(f"🔧 编译LaTeX: {tex_filename}")
            
            # 第一次编译
            result1 = subprocess.run([
                'pdflatex', 
                '-interaction=nonstopmode',
                '-output-directory', output_dir,
                tex_file
            ], capture_output=True, text=True, timeout=30)
            
            if result1.returncode != 0:
                print(f"⚠️ 第一次编译可能有警告: {result1.stderr}")
            
            # 第二次编译解决交叉引用
            result2 = subprocess.run([
                'pdflatex',
                '-interaction=nonstopmode', 
                '-output-directory', output_dir,
                tex_file
            ], capture_output=True, text=True, timeout=30)
            
            # 检查是否生成PDF
            pdf_file = f"{output_path}.pdf"
            if os.path.exists(pdf_file):
                return True
            else:
                print(f"❌ PDF文件未生成: {pdf_file}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ LaTeX编译超时")
            return False
        except FileNotFoundError:
            print("❌ 未找到pdflatex命令，请安装LaTeX: sudo apt-get install texlive-latex-base texlive-latex-extra")
            return False
        except Exception as e:
            print(f"❌ LaTeX编译错误: {e}")
            return False
    
    def cleanup(self):
        """清理临时文件"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)