#!/usr/bin/env python3
import os
import subprocess

def test_latex_installation():
    """测试LaTeX安装"""
    print("🔧 测试LaTeX安装...")
    
    # 检查pdflatex
    try:
        result = subprocess.run(['which', 'pdflatex'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ 找到pdflatex: {result.stdout.strip()}")
        else:
            print("❌ 未找到pdflatex")
            return False
    except Exception as e:
        print(f"❌ 检查pdflatex失败: {e}")
        return False
    
    # 创建测试LaTeX文件
    test_content = r"""
\documentclass{article}
\begin{document}
\title{测试报告}
\author{测试}
\date{\today}
\maketitle

这是一个测试报告。

方程：$\frac{d^2x}{dt^2} + \frac{dx}{dt} = x$

\end{document}
"""
    
    with open('test_report.tex', 'w') as f:
        f.write(test_content)
    
    # 尝试编译
    try:
        result = subprocess.run([
            'pdflatex', '-interaction=nonstopmode', 'test_report.tex'
        ], capture_output=True, text=True, timeout=30)
        
        if os.path.exists('test_report.pdf'):
            print("✅ LaTeX编译成功！PDF文件已生成")
            # 清理测试文件
            for ext in ['.tex', '.log', '.aux']:
                if os.path.exists(f'test_report{ext}'):
                    os.remove(f'test_report{ext}')
            return True
        else:
            print("❌ LaTeX编译失败，未生成PDF")
            return False
            
    except Exception as e:
        print(f"❌ 编译测试失败: {e}")
        return False

if __name__ == "__main__":
    if test_latex_installation():
        print("\n🎉 LaTeX环境正常，可以生成PDF报告！")
    else:
        print("\n💡 请安装LaTeX: sudo dnf-get install texlive-latex-base texlive-latex-extra")