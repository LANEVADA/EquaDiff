import sys
import os

def setup_imports():
    """设置导入路径 - 用于独立运行单个文件时"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

setup_imports()