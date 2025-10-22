#!/usr/bin/env python3
import sys
import os

# 设置路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from RLLearning.CompleteSystem import CompleteDifferentialEquationSystem

if __name__ == "__main__":
    # 设置你的DeepSeek API密钥
    api_key = "your_deepseek_api_key_here"  # 替换为你的API密钥
    
    system = CompleteDifferentialEquationSystem(api_key)
    system.interactive_mode()