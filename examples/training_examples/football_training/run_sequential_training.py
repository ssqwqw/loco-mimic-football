#!/usr/bin/env python3
"""
连续运行多个训练配置的脚本
"""

import subprocess
import sys
import time

def run_training(config_name):
    """运行单个训练配置"""
    cmd = [
        "python", "train_football_multdatasets.py",
        "--config-path", "1-8-yaml",
        "--config-name", config_name
    ]
    
    print("=" * 70)
    print(f"开始运行配置: {config_name}")
    print("=" * 70)
    
    try:
        # 运行训练命令
        result = subprocess.run(cmd, check=True)
        print("=" * 70)
        print(f"配置 {config_name} 训练完成！")
        print("=" * 70)
        return True
    except subprocess.CalledProcessError as e:
        print(f"配置 {config_name} 训练失败！错误码: {e.returncode}")
        return False

def main():
    # 定义要运行的配置文件列表
    configs = ["1",]
    
    success_count = 0
    total_count = len(configs)
    
    for config in configs:
        if run_training(config):
            success_count += 1
        else:
            print(f"由于配置 {config} 失败，停止后续训练")
            break
        
        # 可选：添加延迟避免系统过载
        time.sleep(5)
        print()
    
    print("=" * 70)
    print(f"训练完成！成功: {success_count}/{total_count}")
    print("=" * 70)

if __name__ == "__main__":
    main()