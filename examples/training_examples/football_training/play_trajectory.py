#!/usr/bin/env python3
"""
演示幽灵机器人轨迹 - 只有幽灵机器人，不显示实体机器人
"""

import os
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf
from loco_mujoco.task_factories import ImitationFactory, CustomDatasetConf
from loco_mujoco.trajectory import Trajectory

# ⭐ 定义要演示的轨迹文件列表
TRAJECTORY_PATHS = [
    "/home/user/loco-mujoco/examples/training_examples/football_training/datasets-npz/11-18-1.npz",
    # "/home/user/loco-mujoco/examples/training_examples/football_training/datasets-npz/11-18-2.npz",
    # "/home/user/loco-mujoco/examples/training_examples/football_training/datasets-npz/11-18-3.npz",
    # "/home/user/loco-mujoco/examples/training_examples/football_training/datasets-npz/11-18-4.npz",
    # "/home/user/loco-mujoco/examples/training_examples/football_training/datasets-npz/11-18-5.npz",
]

def test_single_trajectory(env, traj_path, traj_name):
    """演示单个轨迹（只有幽灵机器人）"""
    print(f"\n{'='*50}")
    print(f"演示轨迹: {traj_name}")
    print(f"{'='*50}")
    
    # 加载当前轨迹
    traj = Trajectory.load(traj_path)
    print(f"✓ 加载轨迹: {traj.data.qpos.shape[0]} 帧")
    
    # 更新环境的自定义数据集配置
    env.custom_dataset_conf = CustomDatasetConf(traj)
    
    # 播放轨迹（只有幽灵机器人可见）
    print(f"▶ 开始演示 {traj_name}...")
    try:
        env.play_trajectory(
            n_episodes=1,
            n_steps_per_episode=300,  # 演示每个轨迹300步（约10秒）
            render=True,
            record=True
        )
        print(f"✓ {traj_name} 演示完成")
    except KeyboardInterrupt:
        print(f"\n⏹ {traj_name} 演示被中断")
    
    return True

def main():
    print("=" * 70)
    print("GHOST ROBOT ONLY DEMONSTRATION (NO PHYSICAL ROBOT)")
    print("=" * 70)
    
    # 1. 使用透明机器人模型的环境配置（实体机器人不可见）
    env_params = {
        'env_name': 'MjxUnitreeG1',
        'headless': False,
        'disable_arms': False,
        'horizon': 300,
        # 使用透明机器人模型，只有幽灵机器人可见
        'spec': "/home/user/loco-mujoco/loco_mujoco/models/unitree_g1/g1_29dof.xml",
        'goal_type': 'GoalTrajMimicv2',
        'goal_params': {'visualize_goal': True, 'target_geom_rgba': (0.0, 0.5, 1.0, 0.8)},  # 更亮的蓝色
        'control_type': 'DefaultControl',
        'control_params': {'max_torque': 1.5, 'min_torque': -1.5},
        'reward_type': 'MimicReward',
        'reward_params': {
            'qpos_w_sum': 0.7, 'qvel_w_sum': 0.3, 'rpos_w_sum': 0.8,
            'rquat_w_sum': 0.5, 'rvel_w_sum': 0.2,
            'sites_for_mimic': ['upper_body_mimic', 'left_hand_mimic', 'left_foot_mimic', 'right_hand_mimic', 'right_foot_mimic']
        }
    }
    
    # 2. 创建环境模板（轨迹会在后面动态设置）
    env_name = env_params.pop('env_name', 'MjxUnitreeG1')
    env_params['custom_dataset_conf'] = CustomDatasetConf(Trajectory.load(TRAJECTORY_PATHS[0]))  # 临时轨迹
    env = ImitationFactory.make(env_name, **env_params)
    
    print("✓ 环境创建完成")
    print(f"  - 观测空间: {env.info.observation_space.shape}")
    print(f"  - 动作空间: {env.info.action_space.shape}")
    print("  - 实体机器人: 透明（不可见）")
    print("  - 幽灵机器人: 可见（蓝色）")
    
    # 3. 依次演示每个轨迹
    print(f"\n{'='*70}")
    print(f"开始演示 {len(TRAJECTORY_PATHS)} 个轨迹文件")
    print(f"{'='*70}")
    
    for i, traj_path in enumerate(TRAJECTORY_PATHS):
        traj_name = f"11-18-{i+1}"
        try:
            test_single_trajectory(env, traj_path, traj_name)
        except Exception as e:
            print(f"❌ 演示 {traj_name} 时出错: {e}")
            continue
    
    print(f"\n{'='*70}")
    print("所有轨迹演示完成!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()