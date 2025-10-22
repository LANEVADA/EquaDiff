import sys
import os
import torch
from datetime import datetime
from typing import Dict, Optional
import numpy as np

# 设置导入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from Prompt import Language, MultilingualPromptGenerator
from RLLearning.PromptOptimizationSystem import PromptOptimizationSystem
from RLLearning.ReportEvaluator import ReportEvaluator

# 尝试导入新模块，如果不存在则创建模拟版本
try:
    from utils.DeepseekClient import DeepSeekClient
    from utils.LatexReportGenerator import LatexReportGenerator
except ImportError:
    print("警告: 未找到utils模块，使用模拟版本")
    
    # 模拟DeepSeekClient
    class DeepSeekClient:
        def generate_report(self, prompt: str, equation: str) -> str:
            return f"""
    # 微分方程数值分析报告 (模拟版本)
    
    ## 方程分析
    方程: {equation}
    
    这是一个微分方程的数值分析报告。由于未配置DeepSeek API，这是模拟内容。
    
    ## 数值方法
    采用合适的数值方法进行求解，如欧拉法或龙格-库塔法。
    
    ## 代码实现
    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 数值求解代码示例
    def solve_equation():
        t = np.linspace(0, 10, 100)
        # 根据具体方程实现求解逻辑
        return t, solution
    
    # 可视化和分析
    ```
    
    ## 结果与讨论
    数值求解显示了预期的数学行为。
    """
    
    # 模拟LatexReportGenerator
    class LatexReportGenerator:
        def generate_latex_report(self, report_content: str, equation: str, 
                                language: Language, output_path: str = None) -> str:
            print(f"模拟LaTeX生成: {output_path}")
            # 在实际系统中，这里会生成真实的LaTeX文件并编译
            return f"{output_path}.pdf" if output_path else "simulated_report.pdf"


class CompleteDifferentialEquationSystem:
    """
    完整的微分方程AI分析系统
    集成: Prompt生成 + RL优化 + DeepSeek API + LaTeX报告生成 + 质量评估
    """
    
    def __init__(self, deepseek_api_key: str = None):
        self.optimization_system = PromptOptimizationSystem()
        self.deepseek_client = DeepSeekClient()
        self.latex_generator = LatexReportGenerator()
        self.report_evaluator = ReportEvaluator()
        
        print("✅ 微分方程AI分析系统初始化完成")
        print("   - Prompt生成与优化")
        print("   - DeepSeek API集成") 
        print("   - LaTeX报告生成")
        print("   - 质量评估系统")
        
    def process_equation(self, equation: str, requirements: str, 
                        language: Language = Language.CHINESE,
                        output_dir: str = "reports",
                        use_rl: bool = True) -> Dict:
        """
        完整的方程处理流程
        
        Args:
            equation: 微分方程字符串
            requirements: 分析要求
            language: 输出语言
            output_dir: 输出目录
            use_rl: 是否使用RL优化
            
        Returns:
            包含所有处理结果的字典
        """
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n🚀 开始处理微分方程...")
        print(f"   方程: {equation}")
        print(f"   要求: {requirements}")
        print(f"   语言: {language.value}")
        
        # 步骤1: 生成优化prompt
        print("\n📝 步骤1: 生成优化prompt...")
        prompt_result = self.optimization_system.generate_optimized_prompt(
            equation, requirements, language, use_rl
        )
        
        optimized_prompt = prompt_result['optimized_prompt']
        print(f"   ✅ Prompt生成完成 (长度: {len(optimized_prompt)} 字符)")
        
        # 显示RL优化信息
        if use_rl and prompt_result.get('rl_actions') is not None:
            rl_actions = prompt_result['rl_actions']
            print(f"   🤖 RL优化动作: {len(rl_actions)}维")
        
        # 步骤2: 调用DeepSeek生成报告
        print("\n🧠 步骤2: 调用DeepSeek生成报告...")
        generated_report = self.deepseek_client.generate_report(
            optimized_prompt, equation
        )
        print(f"   ✅ 报告生成完成 (长度: {len(generated_report)} 字符)")
        
        # 步骤3: 生成LaTeX报告
        print("\n📄 步骤3: 生成LaTeX PDF报告...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"report_{timestamp}")
        
        pdf_path = self.latex_generator.generate_latex_report(
            generated_report, equation, language, output_path
        )
        print(f"   ✅ PDF报告生成: {pdf_path}")
        
        # 步骤4: 评估报告质量
        print("\n📊 步骤4: 评估报告质量...")
        score, eval_details = self.report_evaluator.evaluate_report(
            generated_report, equation, requirements, language
        )
        print(f"   ✅ 评估完成: {score:.3f}/1.0")
        
        # 构建返回结果
        result = {
            "prompt": optimized_prompt,
            "generated_report": generated_report,
            "pdf_path": pdf_path,
            "evaluation_score": score,
            "evaluation_details": eval_details,
            "equation_info": prompt_result.get('equation_info', {}),
            "rl_actions": prompt_result.get('rl_actions'),
            "language": language.value,
            "timestamp": timestamp
        }
        
        print(f"\n🎉 处理完成! 评估分数: {score:.3f}/1.0")
        
        return result
    
    def interactive_mode(self):
        """交互式模式 - 与用户对话处理方程"""
        print("=" * 60)
        print("🤖 微分方程AI分析系统 - 交互模式")
        print("=" * 60)
        
        while True:
            print("\n" + "="*50)
            print("请选择语言:")
            print("1. 中文 (Chinese)")
            print("2. English") 
            print("3. Français (French)")
            print("4. 退出系统")
            
            choice = input("\n请输入选择 (1/2/3/4): ").strip()
            
            if choice == "4":
                print("👋 感谢使用，再见!")
                break
                
            language_map = {
                "1": Language.CHINESE,
                "2": Language.ENGLISH,
                "3": Language.FRENCH
            }
            
            if choice not in language_map:
                print("❌ 无效选择，请重新输入")
                continue
                
            language = language_map[choice]
            language_name = {"zh": "中文", "en": "English", "fr": "Français"}[language.value]
            
            print(f"\n🌍 已选择语言: {language_name}")
            
            # 获取用户输入
            print("\n📝 请输入微分方程:")
            equation = input("方程: ").strip()
            
            if not equation:
                print("❌ 方程不能为空")
                continue
                
            print("\n🎯 请输入分析要求 (如: 数值模拟、稳定性分析等):")
            requirements = input("要求: ").strip()
            
            if not requirements:
                requirements = "数值模拟和分析"
                
            # 确认信息
            print(f"\n📋 确认信息:")
            print(f"   方程: {equation}")
            print(f"   要求: {requirements}") 
            print(f"   语言: {language_name}")
            
            confirm = input("\n是否开始处理? (y/n): ").strip().lower()
            if confirm != 'y':
                print("⏸️ 已取消")
                continue
            
            # 开始处理
            try:
                result = self.process_equation(equation, requirements, language)
                
                # 显示详细结果
                self._display_results(result)
                
                # 询问是否继续
                continue_choice = input("\n是否继续处理其他方程? (y/n): ").strip().lower()
                if continue_choice != 'y':
                    print("👋 感谢使用!")
                    break
                    
            except Exception as e:
                print(f"❌ 处理过程中出现错误: {e}")
                import traceback
                traceback.print_exc()
                
                retry = input("\n是否重试? (y/n): ").strip().lower()
                if retry != 'y':
                    break
    
    def _display_results(self, result: Dict):
        """显示处理结果"""
        print("\n" + "="*60)
        print("📊 处理结果摘要")
        print("="*60)
        
        print(f"✅ 评估分数: {result['evaluation_score']:.3f}/1.0")
        print(f"📄 PDF报告: {result['pdf_path']}")
        print(f"🌍 输出语言: {result['language']}")
        
        # 显示方程分析信息
        if result.get('equation_info'):
            eq_info = result['equation_info']
            print(f"🔬 方程分析:")
            if 'complexity' in eq_info:
                print(f"   - 复杂度: {eq_info['complexity']:.2f}/1.0")
            if 'linearity' in eq_info:
                print(f"   - 线性性: {eq_info['linearity']}")
            if 'order' in eq_info:
                print(f"   - 阶数: {eq_info['order']}")
        
        # 显示RL优化信息
        if result.get('rl_actions') is not None:
            print(f"🤖 RL优化: 已应用 {len(result['rl_actions'])} 维优化动作")
        
        # 显示报告预览
        print(f"\n📝 报告预览 (前300字符):")
        preview = result['generated_report'][:300] + "..." if len(result['generated_report']) > 300 else result['generated_report']
        print(preview)
        
        # 显示评估详情
        if result.get('evaluation_details'):
            eval_details = result['evaluation_details']
            if 'dimension_scores' in eval_details:
                print(f"\n📈 详细维度分数:")
                for dim, score in eval_details['dimension_scores'].items():
                    print(f"   - {dim}: {score:.3f}")
    
    def batch_process(self, equations_list: list, output_dir: str = "batch_reports"):
        """
        批量处理多个方程
        
        Args:
            equations_list: 方程列表，每个元素为 (equation, requirements, language) 元组
            output_dir: 输出目录
        """
        print(f"🔄 开始批量处理 {len(equations_list)} 个方程...")
        
        results = []
        for i, (equation, requirements, language) in enumerate(equations_list, 1):
            print(f"\n--- 处理第 {i}/{len(equations_list)} 个方程 ---")
            try:
                result = self.process_equation(equation, requirements, language, output_dir)
                results.append(result)
                print(f"✅ 第 {i} 个方程处理完成")
            except Exception as e:
                print(f"❌ 第 {i} 个方程处理失败: {e}")
                results.append({"error": str(e), "equation": equation})
        
        # 生成批量处理报告
        self._generate_batch_report(results, output_dir)
        return results
    
    def _generate_batch_report(self, results: list, output_dir: str):
        """生成批量处理报告"""
        report_path = os.path.join(output_dir, "batch_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("微分方程AI分析系统 - 批量处理报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总方程数: {len(results)}\n\n")
            
            successful = [r for r in results if 'evaluation_score' in r]
            failed = [r for r in results if 'error' in r]
            
            f.write(f"成功处理: {len(successful)} 个\n")
            f.write(f"处理失败: {len(failed)} 个\n\n")
            
            if successful:
                avg_score = np.mean([r['evaluation_score'] for r in successful])
                f.write(f"平均评估分数: {avg_score:.3f}/1.0\n\n")
                
                f.write("详细结果:\n")
                for i, result in enumerate(successful, 1):
                    f.write(f"{i}. 方程: {result.get('equation_info', {}).get('equation', 'N/A')}\n")
                    f.write(f"   分数: {result['evaluation_score']:.3f}\n")
                    f.write(f"   报告: {result['pdf_path']}\n\n")
        
        print(f"📊 批量处理报告已生成: {report_path}")


def main():
    """主函数 - 命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='微分方程AI分析系统')
    parser.add_argument('--api-key', type=str, help='DeepSeek API密钥')
    parser.add_argument('--equation', type=str, help='微分方程')
    parser.add_argument('--requirements', type=str, help='分析要求')
    parser.add_argument('--language', choices=['zh', 'en', 'fr'], default='zh', help='语言')
    parser.add_argument('--output-dir', type=str, default='reports', help='输出目录')
    parser.add_argument('--batch-file', type=str, help='批量处理文件路径')
    
    args = parser.parse_args()
    
    # 初始化系统
    system = CompleteDifferentialEquationSystem(args.api_key)
    
    if args.batch_file:
        # 批量处理模式
        equations_list = []
        try:
            with open(args.batch_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        parts = line.strip().split('|')
                        if len(parts) >= 2:
                            eq = parts[0].strip()
                            req = parts[1].strip() 
                            lang = Language.CHINESE
                            if len(parts) > 2:
                                lang_map = {'zh': Language.CHINESE, 'en': Language.ENGLISH, 'fr': Language.FRENCH}
                                lang = lang_map.get(parts[2].strip(), Language.CHINESE)
                            equations_list.append((eq, req, lang))
            
            system.batch_process(equations_list, args.output_dir)
            
        except FileNotFoundError:
            print(f"❌ 文件未找到: {args.batch_file}")
            
    elif args.equation and args.requirements:
        # 单方程命令行模式
        language_map = {'zh': Language.CHINESE, 'en': Language.ENGLISH, 'fr': Language.FRENCH}
        result = system.process_equation(
            args.equation, 
            args.requirements, 
            language_map[args.language], 
            args.output_dir
        )
        print(f"✅ 报告生成完成: {result['pdf_path']}")
        
    else:
        # 交互式模式
        system.interactive_mode()


if __name__ == "__main__":
    main()