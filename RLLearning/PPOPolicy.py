import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import torch.optim as optim
class PPOPolicy(nn.Module):
    def __init__(self, state_dim=50, action_dim=20, hidden_dim=256):
        super(PPOPolicy, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Actor网络 (策略网络)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2)  # 输出均值和log标准差
        )
        
        # Critic网络 (价值网络)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 初始化参数
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        
    def forward(self, state):
        return self.get_action(state)
    
    def get_action(self, state, deterministic=False):
        """采样动作 - 修复设备问题"""
        # 确保state是tensor并且在正确设备上
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        
        # 如果state在GPU上，确保模型也在GPU上
        if state.is_cuda and next(self.parameters()).is_cuda:
            state = state.to(next(self.parameters()).device)
        elif state.is_cuda and not next(self.parameters()).is_cuda:
            state = state.cpu()
        elif not state.is_cuda and next(self.parameters()).is_cuda:
            state = state.cuda()
        
        policy_output = self.actor(state)
        mean = policy_output[:, :self.action_dim]
        std = torch.exp(self.log_std).expand_as(mean)
        
        if deterministic:
            action = mean
        else:
            normal = torch.distributions.Normal(mean, std)
            action = normal.rsample()  # 重参数化技巧
        
        # 使用tanh将动作限制在[-1, 1]范围内
        action = torch.tanh(action)
        
        return action
    
    def evaluate_actions(self, state, action):
        """评估动作的概率和熵"""
        policy_output = self.actor(state)
        mean = policy_output[:, :self.action_dim]
        std = torch.exp(self.log_std).expand_as(mean)
        
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(action).sum(dim=-1)
        entropy = normal.entropy().sum(dim=-1)
        
        return log_prob, entropy
    
    def get_value(self, state):
        """状态价值函数"""
        return self.critic(state)

class PPOAgent:
    def __init__(self, state_dim=50, action_dim=20, lr=3e-4, gamma=0.99, 
                 clip_epsilon=0.2, entropy_coef=0.01):
        # 先确定设备，再创建模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🤖 PPOAgent 使用设备: {self.device}")
        
        self.policy = PPOPolicy(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        
        self.memory = []
        
    def store_transition(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
    
    def get_action(self, state, deterministic=False):
        """获取动作 - 修复设备问题"""
        # 确保state是numpy数组或list
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        
        # 创建tensor并确保在正确设备上
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        # 如果需要batch维度，添加它
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)  # [state_dim] -> [1, state_dim]
        
        with torch.no_grad():
            action = self.policy.get_action(state_tensor, deterministic)
        
        # 返回1维numpy数组
        return action.cpu().numpy()[0]  # 从 [1, action_dim] 变为 [action_dim]
    
    def update(self, batch_size=64, epochs=10):
        """PPO更新"""
        if len(self.memory) < batch_size:
            return {}
        
        # 将所有数据转移到设备上
        states = torch.FloatTensor([t['state'] for t in self.memory]).to(self.device)
        actions = torch.FloatTensor([t['action'] for t in self.memory]).to(self.device)
        rewards = torch.FloatTensor([t['reward'] for t in self.memory]).to(self.device)
        next_states = torch.FloatTensor([t['next_state'] for t in self.memory]).to(self.device)
        dones = torch.FloatTensor([t['done'] for t in self.memory]).to(self.device)
        
        # 计算优势估计
        with torch.no_grad():
            values = self.policy.get_value(states).squeeze()
            next_values = self.policy.get_value(next_states).squeeze()
            targets = rewards + self.gamma * next_values * (1 - dones)
            advantages = targets - values
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO更新循环
        losses = []
        for _ in range(epochs):
            # 随机打乱数据
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_targets = targets[batch_indices]
                
                # 计算新旧策略的概率
                old_log_prob, _ = self.policy.evaluate_actions(batch_states, batch_actions)
                new_log_prob, entropy = self.policy.evaluate_actions(batch_states, batch_actions)
                
                # 策略比率
                ratio = torch.exp(new_log_prob - old_log_prob.detach())
                
                # PPO损失
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                values_pred = self.policy.get_value(batch_states).squeeze()
                value_loss = F.mse_loss(values_pred, batch_targets)
                
                # 总损失
                total_loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy.mean()
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
                losses.append(total_loss.item())
        
        # 清空记忆
        self.memory = []
        
        return {'total_loss': np.mean(losses) if losses else 0.0}