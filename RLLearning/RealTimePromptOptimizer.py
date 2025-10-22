import utils.import_utils  # 这会自动设置sys.path
from .PromptOptimizationSystem import PromptOptimizationSystem
from .ActionDesign import ActionSpaceDesign
from Prompt.Language import Language

class RealTimePromptOptimizer:
    """实时Prompt优化演示"""
    
    def __init__(self, model_path: str = None):
        self.system = PromptOptimizationSystem()
        if model_path:
            self.system.load_model(model_path)
        self.system.training_mode = False  # 推理模式
    
    def optimize_prompt_interactive(self):
        """交互式Prompt优化"""
        print("=== 实时Prompt优化系统 ===")
        
        while True:
            print("\n" + "="*50)
            print("请选择语言:")
            print("1. 中文")
            print("2. English") 
            print("3. Français")
            print("4. 退出")
            
            choice = input("请输入选择 (1/2/3/4): ").strip()
            
            if choice == "4":
                break
                
            language_map = {
                "1": Language.CHINESE,
                "2": Language.ENGLISH,
                "3": Language.FRENCH
            }
            
            if choice not in language_map:
                print("无效选择，请重新输入")
                continue
                
            language = language_map[choice]
            
            # 获取用户输入
            equation = input("请输入微分方程: ")
            requirements = input("请输入要求: ")
            
            # 生成优化后的prompt
            print("\n生成优化后的prompt...")
            result = self.system.generate_optimized_prompt(
                equation, requirements, language, use_rl=True
            )
            
            print(f"\n优化后的Prompt ({language.value}):")
            print("="*50)
            print(result['optimized_prompt'])
            print("="*50)
            
            # 显示RL动作分析
            if result['rl_actions'] is not None:
                actions = ActionSpaceDesign.decode_actions(result['rl_actions'])
                print("\nRL优化策略:")
                for action_name, value in list(actions.items())[:5]:  # 显示前5个最重要的动作
                    if abs(value) > 0.3:  # 只显示显著的动作
                        print(f"  {action_name}: {value:.2f}")

# 运行实时优化器
def run_realtime_optimizer():
    optimizer = RealTimePromptOptimizer()  # 可以加载训练好的模型
    optimizer.optimize_prompt_interactive()

if __name__ == "__main__":
    run_realtime_optimizer()