#!/usr/bin/env python3
"""
测试训练好的足球动作模型 - 多数据集训练版本
"""

import os
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.task_factories import ImitationFactory, CustomDatasetConf
from loco_mujoco.trajectory import Trajectory

# ⭐ 修改为您的多数据集训练模型路径
MODEL_PATH = "/home/user/loco-mujoco/examples/training_examples/football_training/outputs/2025-11-29/23-17-36/PPOJax_saved.pkl"

# ⭐ 定义要测试的轨迹文件列表（与训练时使用的相同）
TRAJECTORY_PATHS = [
    "/home/user/loco-mujoco/examples/training_examples/football_training/datasets-npz/11-18-1.npz",
    "/home/user/loco-mujoco/examples/training_examples/football_training/datasets-npz/11-18-2.npz",
    "/home/user/loco-mujoco/examples/training_examples/football_training/datasets-npz/11-18-3.npz",
    "/home/user/loco-mujoco/examples/training_examples/football_training/datasets-npz/11-18-4.npz",
    "/home/user/loco-mujoco/examples/training_examples/football_training/datasets-npz/11-18-5.npz",

]

def test_single_trajectory(env, agent_conf, agent_state, traj_path, traj_name):
    """测试单个轨迹"""
    print(f"\n{'='*50}")
    print(f"测试轨迹: {traj_name}")
    print(f"{'='*50}")
    
    # 加载当前轨迹
    traj = Trajectory.load(traj_path)
    print(f"✓ 加载轨迹: {traj.data.qpos.shape[0]} 帧")
    
    # 更新环境的自定义数据集配置
    env.custom_dataset_conf = CustomDatasetConf(traj)
    
    # 循环播放策略（较短时间用于测试）
    print(f"▶ 开始播放 {traj_name}...")
    try:
        PPOJax.play_policy(
            env, 
            agent_conf, 
            agent_state,
            deterministic=True,
            n_steps=300,  # 测试每个轨迹300步（约10秒）
            n_envs=1,
            record=True
        )
        print(f"✓ {traj_name} 测试完成")
    except KeyboardInterrupt:
        print(f"\n⏹ {traj_name} 播放被中断")
    
    return True

def main():
    print("=" * 70)
    print("LOADING TRAINED MODEL (MULTI-DATASET VERSION)")
    print("=" * 70)
    
    # 1. 加载训练好的 agent
    agent_conf, agent_state = PPOJax.load_agent(MODEL_PATH)
    print(f"✓ 多数据集训练模型已加载: {MODEL_PATH}")
    
    # 2. 加载训练时的配置（从模型目录获取）
    model_dir = os.path.dirname(MODEL_PATH)
    config_path = os.path.join(model_dir, ".hydra", "config.yaml")
    
    if not os.path.exists(config_path):
        print("⚠️ 未找到训练配置文件，使用默认配置...")
        # 使用默认配置（需要手动指定关键参数）
        env_params = {
            'env_name': 'MjxUnitreeG1',
            'headless': False,
            'disable_arms': False,
            'horizon': 300,
            'spec': "/home/user/loco-mujoco/loco_mujoco/models/unitree_g1/g1_29dof.xml",
            'goal_type': 'GoalTrajMimicv2',
            'goal_params': {'visualize_goal': True, 'target_geom_rgba': (0.0, 0.5, 1.0, 0.5)},
            'control_type': 'DefaultControl',
            'control_params': {'max_torque': 1.5, 'min_torque': -1.5},
            'reward_type': 'MimicReward',
            'reward_params': {
                'qpos_w_sum': 0.7, 'qvel_w_sum': 0.3, 'rpos_w_sum': 0.8,
                'rquat_w_sum': 0.5, 'rvel_w_sum': 0.2,
                'sites_for_mimic': ['upper_body_mimic', 'left_hand_mimic', 'left_foot_mimic', 'right_hand_mimic', 'right_foot_mimic']
            }
        }
    else:
        print(f"✓ 加载训练配置: {config_path}")
        config = OmegaConf.load(config_path)
        env_params = OmegaConf.to_container(config.experiment.env_params, resolve=True)
        # 启用可视化
        env_params['headless'] = False
        env_params['goal_type'] = 'GoalTrajMimicv2'
        env_params['goal_params'] = {
            'visualize_goal': False,
            'target_geom_rgba': (0.0, 0.5, 1.0, 0.5)
                            # 紫色（默认）
                            # (0.471, 0.38, 0.812, 0.5)
                            # # 红色
                            # (1.0, 0.0, 0.0, 0.5)
                            # # 绿色
                            # (0.0, 1.0, 0.0, 0.5)
                            # # 蓝色
                            # (0.0, 0.5, 1.0, 0.5)
        }
    
    # 3. 创建环境模板（轨迹会在后面动态设置）
    env_name = env_params.pop('env_name', 'MjxUnitreeG1')
    env_params['custom_dataset_conf'] = CustomDatasetConf(Trajectory.load(TRAJECTORY_PATHS[0]))  # 临时轨迹
    env = ImitationFactory.make(env_name, **env_params)
    
    print("✓ 环境创建完成")
    print(f"  - 观测空间: {env.info.observation_space.shape}")
    print(f"  - 动作空间: {env.info.action_space.shape}")
    print("  - 幽灵机器人可视化已启用!")
    
    # 4. 依次测试每个轨迹
    print(f"\n{'='*70}")
    print(f"开始测试 {len(TRAJECTORY_PATHS)} 个轨迹文件")
    print(f"{'='*70}")
    
    for i, traj_path in enumerate(TRAJECTORY_PATHS):
        traj_name = f"11-18-{i+1}"
        try:
            test_single_trajectory(env, agent_conf, agent_state, traj_path, traj_name)
        except Exception as e:
            print(f"❌ 测试 {traj_name} 时出错: {e}")
            continue
    
    print(f"\n{'='*70}")
    print("所有轨迹测试完成!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()