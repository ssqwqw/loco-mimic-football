#!/usr/bin/env python3
"""
测试训练好的足球动作模型
"""

import os
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.task_factories import ImitationFactory, CustomDatasetConf
from loco_mujoco.trajectory import Trajectory

# ⭐ 修改为您的模型路径
MODEL_PATH = "/home/user/loco-mujoco/examples/training_examples/football_training/outputs/2025-11-17/13-13-00/PPOJax_saved.pkl"
TRAJECTORY_PATH = "/home/user/loco-mujoco/examples/training_examples/football_training/datasets-npz/football_motion.npz"

def main():
    print("=" * 70)
    print("LOADING TRAINED MODEL")
    print("=" * 70)
    
    # 1. 加载轨迹
    traj = Trajectory.load(TRAJECTORY_PATH)
    print(f"✓ Loaded trajectory: {traj.data.qpos.shape[0]} frames")
    
    # 2. 加载训练时的配置
    model_dir = os.path.dirname(MODEL_PATH)
    config_path = os.path.join(model_dir, ".hydra", "config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Training config not found at: {config_path}\n"
            f"Make sure the model directory contains .hydra/config.yaml"
        )
    
    print(f"✓ Loading training config from: {config_path}")
    config = OmegaConf.load(config_path)
    
    # 3. 从训练配置中提取环境参数
    env_params = OmegaConf.to_container(config.experiment.env_params, resolve=True)
    
    # 添加自定义轨迹和可视化参数
    env_params['custom_dataset_conf'] = CustomDatasetConf(traj)
    env_params['headless'] = False  # 启用可视化
    
    # ⭐⭐⭐ 关键修改：使用 GoalTrajMimicv2 来显示幽灵机器人 ⭐⭐⭐
    env_params['goal_type'] = 'GoalTrajMimicv2'
    
    # ⭐⭐⭐ 必须添加 visualize_goal: True！⭐⭐⭐
    env_params['goal_params'] = {
        'visualize_goal': True,  # 这是显示幽灵机器人的关键！
        'target_geom_rgba': (1.0, 0.0, 0.0, 0.3)   
                            # 紫色（默认）
                            # (0.471, 0.38, 0.812, 0.5)
                            # # 红色
                            # (1.0, 0.0, 0.0, 0.5)
                            # # 绿色
                            # (0.0, 1.0, 0.0, 0.5)
                            # # 蓝色
                            # (0.0, 0.5, 1.0, 0.5)

    }
    
    print(f"✓ Using training environment configuration:")
    print(f"  - Observation space will match training")
    print(f"  - Goal type: {env_params.get('goal_type', 'Unknown')}")
    print(f"  - Reward type: {env_params.get('reward_type', 'Unknown')}")
    
    # 4. 创建环境（使用训练时的完整配置）
    env_name = env_params.pop('env_name', 'MjxUnitreeG1')
    env = ImitationFactory.make(env_name, **env_params)
    print(f"✓ Environment created")
    print(f"  - Observation space: {env.info.observation_space.shape}")
    print(f"  - Action space: {env.info.action_space.shape}")
    
    # 5. 加载训练好的 agent
    agent_conf, agent_state = PPOJax.load_agent(MODEL_PATH)
    print(f"✓ Model loaded from: {MODEL_PATH}")
    
    # 6. 简单诊断
    print(f"\n🔍 Environment info:")
    print(f"  - Observation space: {env.info.observation_space.shape}")
    print(f"  - Action space: {env.info.action_space.shape}")
    print(f"  ✓ Using same config as training - dimensions will match!")
    print(f"  👻 Ghost robot visualization enabled!")
    
    # 7. 循环播放训练好的策略（按 Ctrl+C 停止）
    print("\n▶ 开始循环播放（按 Ctrl+C 停止）...")
    try:
        PPOJax.play_policy(
            env, 
            agent_conf, 
            agent_state,
            deterministic=True,
            n_steps=1000000,  # ← 超大数字，基本就是无限循环
            n_envs=1,
            record=False
        )
    except KeyboardInterrupt:
        print("\n⏹ 播放已停止")
    
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()