import requests
import json
import os
import time
from typing import Dict, Optional

class DeepSeekClient:
    def __init__(self, api_key: str = None, use_mock: bool = False):
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.use_mock = use_mock or (self.api_key is None)
        
        if self.use_mock:
            print("🔶 使用模拟模式 - 生成示例报告")
        else:
            print("✅ 使用真实DeepSeek API")
    
    def generate_report(self, prompt: str, equation: str, timeout: int = 60) -> str:
        """生成报告 - 支持真实API和模拟模式"""
        
        if self.use_mock:
            return self._generate_mock_report(prompt, equation)
        
        return self._call_real_api(prompt, equation, timeout)
    
    def _call_real_api(self, prompt: str, equation: str, timeout: int) -> str:
        """调用真实DeepSeek API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system", 
                    "content": """你是一个专业的计算数学专家，擅长微分方程数值分析。请生成详细、专业的技术报告，包含以下部分：
1. 方程分析：数学特性和物理意义
2. 数值方法：方法选择和理论依据  
3. 代码实现：完整可执行的Python代码
4. 数值结果：结果展示和可视化
5. 误差分析：收敛性和稳定性讨论
6. 应用讨论：实际意义和扩展应用

请使用专业术语，提供详细的数学推导和代码注释。"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 4000
        }
        
        try:
            print("🌐 调用DeepSeek API...")
            response = requests.post(self.base_url, headers=headers, json=data, timeout=timeout)
            response.raise_for_status()
            
            result = response.json()
            report = result["choices"][0]["message"]["content"]
            print("✅ API调用成功")
            return report
            
        except requests.exceptions.RequestException as e:
            print(f"❌ API调用失败: {e}")
            return self._generate_mock_report(prompt, equation)
        except Exception as e:
            print(f"❌ 解析响应失败: {e}")
            return self._generate_mock_report(prompt, equation)
    
    def _generate_mock_report(self, prompt: str, equation: str) -> str:
        """生成高质量的模拟报告"""
        # 模拟API调用的延迟
        time.sleep(2)
        
        return f"""
# 微分方程数值分析报告

## 方程分析
**方程**: {equation}

这是一个典型的微分方程，具有重要的数学和物理意义。我们对其进行系统的数值分析。

### 数学特性
- **方程类型**: {self._classify_equation(equation)}
- **线性性**: {self._detect_linearity(equation)}
- **阶数**: {self._detect_order(equation)}

## 数值方法

### 方法选择
基于方程特性，选择**四阶龙格-库塔法 (RK4)** 进行数值求解：

$$
k_1 = h f(t_n, y_n)
$$
$$
k_2 = h f(t_n + \\frac{{h}}{{2}}, y_n + \\frac{{k_1}}{{2}})
$$
$$
k_3 = h f(t_n + \\frac{{h}}{{2}}, y_n + \\frac{{k_2}}{{2}})
$$
$$
k_4 = h f(t_n + h, y_n + k_3)
$$
$$
y_{{n+1}} = y_n + \\frac{{1}}{{6}}(k_1 + 2k_2 + 2k_3 + k_4)
$$

### 理论依据
- **精度**: 四阶精度，局部截断误差 $O(h^5)$
- **稳定性**: 条件稳定，适合刚性方程
- **效率**: 计算量与精度平衡良好

## 代码实现
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import seaborn as sns

class DifferentialEquationSolver:
    # \"\"\"微分方程数值求解器\"\"\"
    
    def __init__(self, equation_type: str):
        self.equation_type = equation_type
        
    def define_equation(self, t, y):
        # \"\"\"定义微分方程右端函数\"\"\"
        if 'dy/dt = -k*y' in self.equation_type:
            # 指数衰减方程
            k = 0.1
            return -k * y
        elif 'd²x/dt²' in self.equation_type:
            # 简谐振动方程
            omega = 2.0
            return [y[1], -omega**2 * y[0]]
        else:
            # 默认方程
            return -0.1 * y
    
    def solve_numerically(self, t_span, y0, method='RK45'):
        # \"\"\"数值求解\"\"\"
        solution = solve_ivp(
            self.define_equation, 
            t_span, 
            y0, 
            method=method,
            t_eval=np.linspace(t_span[0], t_span[1], 1000)
        )
        return solution
    
    def visualize_results(self, solution, title="数值解"):
        # \"\"\"可视化结果\"\"\"
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(solution.t, solution.y[0] if len(solution.y.shape) > 1 else solution.y)
        plt.xlabel('时间 t')
        plt.ylabel('y(t)')
        plt.title(f'{title} - 时间序列')
        plt.grid(True, alpha=0.3)
        
        if len(solution.y.shape) > 1 and solution.y.shape[0] > 1:
            plt.subplot(2, 2, 2)
            plt.plot(solution.y[0], solution.y[1])
            plt.xlabel('y(t)')
            plt.ylabel("dy/dt")
            plt.title('相图')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# 使用示例
if __name__ == "__main__":
    # 创建求解器实例
    solver = DifferentialEquationSolver("{equation}")
    
    # 设置初始条件
    t_span = (0, 10)
    y0 = [1.0]  # 初始条件
    
    # 数值求解
    solution = solver.solve_numerically(t_span, y0)
    
    # 可视化
    solver.visualize_results(solution, "{equation}")