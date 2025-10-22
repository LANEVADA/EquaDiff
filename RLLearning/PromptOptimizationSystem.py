import torch
import torch.nn as nn
import numpy as np
from typing import Dict
import utils.import_utils  # 这会自动设置sys.path
from Prompt.Language import Language
from Prompt.MultilingualPromptGenerator import MultilingualPromptGenerator
from .PPOPolicy import PPOAgent
from .ReportEvaluator import ReportEvaluator

class PromptOptimizationSystem:
    def __init__(self):
        # 先确定设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 PromptOptimizationSystem 使用设备: {self.device}")
        
        # 核心组件
        self.prompt_generator = MultilingualPromptGenerator()
        self.rl_agent = PPOAgent(state_dim=50, action_dim=20)
        
        # 评估器
        self.evaluator = ReportEvaluator()
        
        # 训练状态
        self.training_mode = True
        self.episode_count = 0
        
    def generate_optimized_prompt(self, equation: str, requirements: str, 
                                language: Language = Language.CHINESE,
                                use_rl: bool = True) -> Dict:
        """生成优化后的prompt - 修复设备问题"""
        # 提取状态特征
        state = self._extract_state(equation, requirements)
        
        # 获取RL动作
        if use_rl and self.training_mode:
            try:
                rl_actions = self.rl_agent.get_action(state)
                # 转换为torch tensor并确保在CPU上（因为prompt生成器可能在CPU上）
                rl_actions_tensor = torch.FloatTensor(rl_actions).cpu()
            except Exception as e:
                print(f"⚠️ RL动作生成失败，使用默认动作: {e}")
                rl_actions = np.random.uniform(-1, 1, 20)  # 随机动作作为fallback
                rl_actions_tensor = torch.FloatTensor(rl_actions).cpu()
        else:
            rl_actions_tensor = None
            rl_actions = None
        
        # 生成prompt
        try:
            result = self.prompt_generator(
                equation, requirements, 
                rl_actions=rl_actions_tensor,
                target_language=language
            )
            
            result['state'] = state
            result['rl_actions'] = rl_actions
            
            return result
            
        except Exception as e:
            print(f"❌ Prompt生成失败: {e}")
            # 返回一个基本的prompt作为fallback
            return {
                'base_prompt': f"分析方程: {equation}\n要求: {requirements}",
                'optimized_prompt': f"分析方程: {equation}\n要求: {requirements}",
                'equation_info': {},
                'requirements_analysis': {},
                'input_language': language,
                'output_language': language,
                'state': state,
                'rl_actions': rl_actions
            }
    
    def _extract_state(self, equation: str, requirements: str) -> np.ndarray:
        """提取RL状态特征"""
        try:
            # 使用EquationFeatureExtractor提取方程特征
            equation_features = self.prompt_generator.feature_extractor.extract_features(equation).numpy()
            
            # 分析需求特征
            lang = self.prompt_generator.language_detector.detect(requirements)
            req_analysis = self.prompt_generator.analyze_requirements(requirements, lang)
            
            # 需求特征编码
            req_features = []
            req_features.append(1.0 if req_analysis.get("detail_level") == "high" else 0.0)
            req_features.append(1.0 if req_analysis.get("detail_level") == "low" else 0.0)
            req_features.append(1.0 if req_analysis.get("needs_validation", False) else 0.0)
            req_features.append(1.0 if req_analysis.get("needs_visualization", True) else 0.0)
            req_features.append(1.0 if req_analysis.get("needs_comparison", False) else 0.0)
            req_features.append(1.0 if req_analysis.get("needs_theory", False) else 0.0)
            
            # 组合特征
            state = np.concatenate([equation_features, req_features])
            
            # 确保状态维度一致
            if len(state) < 50:
                state = np.pad(state, (0, 50 - len(state)))
            else:
                state = state[:50]
                
            return state
            
        except Exception as e:
            print(f"⚠️ 状态提取失败，使用默认状态: {e}")
            # 返回默认状态
            return np.random.uniform(0, 1, 50)
    
    # ... 其他方法保持不变
    
    def evaluate_and_learn(self, equation: str, requirements: str, 
                          generated_report: str, language: Language) -> Dict:
        """评估报告并学习"""
        # 提取状态
        state = self._extract_state(equation, requirements)
        
        # 评估报告质量
        reward, eval_details = self.evaluator.evaluate_report(
            generated_report, equation, requirements, language
        )
        
        # 简单的下一个状态 (这里可以更复杂)
        next_state = state  # 在实际应用中，可以根据报告内容更新状态
        
        # 存储经验
        self.rl_agent.store_transition(
            state=state,
            action=self.rl_agent.get_action(state),
            reward=reward,
            next_state=next_state,
            done=True  # 每个方程求解是独立episode
        )
        
        # 定期更新策略
        if self.episode_count % 10 == 0:  # 每10个episode更新一次
            loss_info = self.rl_agent.update()
        else:
            loss_info = {}
        
        self.episode_count += 1
        
        return {
            'reward': reward,
            'evaluation_details': eval_details,
            'training_loss': loss_info
        }
    
    def train_episode(self, equation: str, requirements: str, 
                     language: Language = Language.CHINESE) -> Dict:
        """训练一个episode"""
        # 生成优化后的prompt
        prompt_result = self.generate_optimized_prompt(
            equation, requirements, language, use_rl=True
        )
        
        # 这里应该调用DeepSeek API生成报告
        # 为演示目的，我们模拟一个报告
        simulated_report = self._simulate_report_generation(prompt_result['optimized_prompt'])
        
        # 评估和学习
        learning_result = self.evaluate_and_learn(
            equation, requirements, simulated_report, language
        )
        
        return {
            'prompt_result': prompt_result,
            'learning_result': learning_result,
            'generated_report': simulated_report
        }
    
    def _simulate_report_generation(self, prompt: str) -> str:
        """模拟报告生成 (实际应该调用DeepSeek API)"""
        # 这里应该调用真实的DeepSeek API
        # 为演示目的，返回模拟报告
        return f"""
        基于prompt生成的模拟报告:
        
        {prompt[:100]}...
        
        这是一个模拟的报告内容，包含：
        1. 方程分析
        2. 数值方法选择
        3. Python代码实现
        4. 结果可视化
        5. 误差分析
        
        在实际应用中，这里应该是DeepSeek生成的真实报告。
        """
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'policy_state_dict': self.rl_agent.policy.state_dict(),
            'optimizer_state_dict': self.rl_agent.optimizer.state_dict(),
            'episode_count': self.episode_count
        }, path)
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.rl_agent.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.rl_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_count = checkpoint['episode_count']