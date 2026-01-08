#!/usr/bin/env python3
"""
测试足球设置：验证足球是否正确放置在机器人正前方，以及奖励函数是否正常工作
"""
import os
import sys
import jax
import jax.numpy as jnp

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from loco_mujoco.task_factories import ImitationFactory, CustomDatasetConf
from loco_mujoco.trajectory import Trajectory

# 导入自定义模块
sys.path.insert(0, os.path.dirname(__file__))
from football_init_state import FootballInitialStateHandler
from football_reward import FootballApproachReward

# 测试轨迹路径
TRAJECTORY_PATH = "/home/user/loco-mujoco/examples/training_examples/football_training/datasets-npz/11-18-1.npz"

def test_football_setup():
    """测试足球设置"""
    print("=" * 70)
    print("测试足球设置")
    print("=" * 70)
    
    # 加载轨迹
    traj = Trajectory.load(TRAJECTORY_PATH)
    print(f"✓ 加载轨迹: {TRAJECTORY_PATH}")
    
    # 创建环境配置
    env_params = {
        'env_name': 'MjxUnitreeG1',
        'headless': False,
        'disable_arms': False,
        'horizon': 300,
        'spec': "/home/user/loco-mujoco/loco_mujoco/models/unitree_g1/g1_29dof.xml",
        'goal_type': 'GoalTrajMimic',
        'goal_params': {'visualize_goal': True},
        'control_type': 'DefaultControl',
        'control_params': {'max_torque': 1.5, 'min_torque': -1.5},
        # 使用自定义的足球接近奖励函数
        'reward_type': 'FootballApproachReward',
        'reward_params': {
            'qpos_w_sum': 0.7, 'qvel_w_sum': 0.3, 'rpos_w_sum': 0.8,
            'rquat_w_sum': 0.5, 'rvel_w_sum': 0.2,
            'sites_for_mimic': ['upper_body_mimic', 'left_hand_mimic',
                                 'left_foot_mimic', 'right_hand_mimic',
                                 'right_foot_mimic'],
            # 足球接近奖励参数
            'football_approach_weight': 0.3,
            'football_distance_scale': 0.5,
        },
        # 使用自定义的初始状态处理器
        'initial_state_type': 'FootballInitialStateHandler',
        'initial_state_params': {
            'football_distance': 0.8,
            'football_height': 0.11,
        },
        'custom_dataset_conf': CustomDatasetConf(traj)
    }
    
    # 创建环境
    print("\n创建环境...")
    env = ImitationFactory.make(**env_params)
    print(f"✓ 环境创建成功")
    print(f"  - 观测空间: {env.info.observation_space.shape}")
    print(f"  - 动作空间: {env.info.action_space.shape}")
    
    # 重置环境
    print("\n重置环境...")
    key = jax.random.PRNGKey(0)
    obs, info = env.reset(key)
    print(f"✓ 环境重置成功")
    
    # 检查足球位置
    print("\n检查足球位置...")
    football_pos = env.data.xpos[env._model.body("football").id]
    robot_pos = env.data.xpos[env._model.body("pelvis").id]
    
    print(f"  - 机器人位置: ({robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f})")
    print(f"  - 足球位置: ({football_pos[0]:.3f}, {football_pos[1]:.3f}, {football_pos[2]:.3f})")
    
    # 计算距离
    distance_2d = jnp.linalg.norm(robot_pos[:2] - football_pos[:2])
    distance_3d = jnp.linalg.norm(robot_pos - football_pos)
    print(f"  - 水平距离: {distance_2d:.3f} 米")
    print(f"  - 3D距离: {distance_3d:.3f} 米")
    
    # 测试奖励函数
    print("\n测试奖励函数...")
    action = jnp.zeros(env.info.action_space.shape[0])
    obs_next, reward, terminated, truncated, info = env.step(action)
    print(f"  - 奖励值: {reward:.4f}")
    print(f"  - 终止: {terminated}, 截断: {truncated}")
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)
    print("\n提示：")
    print("  - 足球应该位于机器人正前方约0.8米处")
    print("  - 奖励函数会根据机器人与足球的距离给予奖励")
    print("  - 距离越近，奖励越高")


if __name__ == "__main__":
    test_football_setup()

