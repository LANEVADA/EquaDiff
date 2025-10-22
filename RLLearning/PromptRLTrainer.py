import random
from typing import List, Dict
import utils.import_utils  # 这会自动设置sys.path
from Prompt.Language import Language
from .PromptOptimizationSystem import PromptOptimizationSystem


class PromptRLTrainer:
    def __init__(self):
        self.system = PromptOptimizationSystem()
        self.training_equations = self._load_training_data()
        
    def _load_training_data(self) -> List[Dict]:
        """加载训练数据"""
        return [
            {
                "equation": "dy/dt = -0.1*y, y(0) = 1",
                "requirements": "数值模拟和稳定性分析",
                "language": Language.CHINESE
            },
            {
                "equation": "d²x/dt² + 0.5*dx/dt + 2*x = 0, x(0)=1, dx/dt(0)=0",
                "requirements": "harmonic oscillator simulation with visualization",
                "language": Language.ENGLISH
            },
            {
                "equation": "∂u/∂t = 0.1*∂²u/∂x², u(0,t)=0, u(1,t)=0, u(x,0)=sin(πx)",
                "requirements": "simulation de l'équation de la chaleur",
                "language": Language.FRENCH
            },
            # 可以添加更多训练数据...
        ]
    
    def train(self, num_episodes: int = 1000):
        """训练循环"""
        print("开始Prompt优化RL训练...")
        
        for episode in range(num_episodes):
            # 随机选择训练样本
            training_sample = random.choice(self.training_equations)
            
            # 训练一个episode
            result = self.system.train_episode(
                training_sample["equation"],
                training_sample["requirements"],
                training_sample["language"]
            )
            
            # 记录训练进度
            if episode % 100 == 0:
                reward = result['learning_result']['reward']
                print(f"Episode {episode}, Reward: {reward:.3f}")
                
                if 'training_loss' in result['learning_result']:
                    loss = result['learning_result']['training_loss'].get('total_loss', 0)
                    print(f"  Loss: {loss:.4f}")
            
            # 定期保存模型
            if episode % 500 == 0 and episode > 0:
                self.system.save_model(f"prompt_rl_model_episode_{episode}.pth")
        
        print("训练完成!")
    
    def evaluate(self, test_equations: List[Dict]):
        """评估模型性能"""
        print("\n开始模型评估...")
        
        total_reward = 0
        for i, test_case in enumerate(test_equations):
            result = self.system.train_episode(
                test_case["equation"],
                test_case["requirements"], 
                test_case["language"]
            )
            
            reward = result['learning_result']['reward']
            total_reward += reward
            
            print(f"测试案例 {i+1}: Reward = {reward:.3f}")
            print(f"  方程: {test_case['equation']}")
            print(f"  生成的Prompt前100字符: {result['prompt_result']['optimized_prompt'][:100]}...")
        
        avg_reward = total_reward / len(test_equations)
        print(f"\n平均奖励: {avg_reward:.3f}")
        
        return avg_reward

# 使用示例
def main():
    # 创建训练器
    trainer = PromptRLTrainer()
    
    # 训练模型
    trainer.train(num_episodes=1000)
    
    # 测试模型
    test_cases = [
        {
            "equation": "dy/dt = -k*y + sin(t), y(0) = 0",
            "requirements": "带强迫项的阻尼振动数值模拟",
            "language": Language.CHINESE
        },
        {
            "equation": "dx/dt = y, dy/dt = -x - 0.1*y",
            "requirements": "phase portrait and stability analysis", 
            "language": Language.ENGLISH
        }
    ]
    
    trainer.evaluate(test_cases)
    
    # 保存最终模型
    trainer.system.save_model("final_prompt_rl_model.pth")

if __name__ == "__main__":
    main()