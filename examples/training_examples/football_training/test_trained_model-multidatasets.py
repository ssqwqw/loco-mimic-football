#!/usr/bin/env python3
"""
测试训练好的足球动作模型 - 多数据集训练版本
"""

import os
import sys
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.task_factories import ImitationFactory, CustomDatasetConf
from loco_mujoco.trajectory import Trajectory

# 导入自定义的初始状态处理器和奖励函数
sys.path.insert(0, os.path.dirname(__file__))
from football_init_state import FootballInitialStateHandler
from football_reward import FootballApproachReward


# ⭐ 修改为您的多数据集训练模型路径
MODEL_PATH = "/home/user/loco-mujoco/examples/training_examples/football_training/outputs/2026-01-08/18-42-42/PPOJax_saved.pkl"

# ⭐ 定义要测试的轨迹文件列表（与训练时使用的相同）
TRAJECTORY_PATHS = [
    "/home/user/loco-mujoco/examples/training_examples/football_training/datasets-npz/11-18-1.npz",
    "/home/user/loco-mujoco/examples/training_examples/football_training/datasets-npz/11-18-2.npz",
    "/home/user/loco-mujoco/examples/training_examples/football_training/datasets-npz/11-18-3.npz",
    "/home/user/loco-mujoco/examples/training_examples/football_training/datasets-npz/11-18-4.npz",
    "/home/user/loco-mujoco/examples/training_examples/football_training/datasets-npz/11-18-5.npz",
]


def fix_run_stats_dimension(run_stats, target_dim):
    """
    修复 run_stats 的维度以匹配目标维度
    
    Args:
        run_stats: 原始的 run_stats 字典（可能是嵌套结构）
        target_dim: 目标维度
    
    Returns:
        修复后的 run_stats 字典
    """
    # 尝试多种方式访问 run_stats
    old_mean = None
    old_var = None
    old_count = jnp.array(1e6)
    stats_key = None
    
    # 方法1: 检查是否是嵌套结构（如 {'RunningMeanStd_0': {'mean': ..., 'var': ...}}）
    if isinstance(run_stats, dict):
        # 查找包含 'mean' 的子字典
        for key, value in run_stats.items():
            if isinstance(value, dict) and 'mean' in value:
                stats_key = key
                old_mean = value['mean']
                old_var = value['var']
                if 'count' in value:
                    old_count = value['count']
                break
    
    # 方法2: 直接字典访问（扁平结构）
    if old_mean is None:
        try:
            old_mean = run_stats['mean']
            old_var = run_stats['var']
            if 'count' in run_stats:
                old_count = run_stats['count']
        except (KeyError, TypeError):
            pass
    
    # 方法3: 如果是 FrozenDict，尝试解冻
    if old_mean is None and hasattr(run_stats, 'unfreeze'):
        try:
            unfrozen = run_stats.unfreeze()
            # 检查嵌套结构
            for key, value in unfrozen.items():
                if isinstance(value, dict) and 'mean' in value:
                    stats_key = key
                    old_mean = value['mean']
                    old_var = value['var']
                    if 'count' in value:
                        old_count = value['count']
                    break
            # 如果不是嵌套结构，直接访问
            if old_mean is None and 'mean' in unfrozen:
                old_mean = unfrozen['mean']
                old_var = unfrozen['var']
                if 'count' in unfrozen:
                    old_count = unfrozen['count']
        except Exception:
            pass
    
    # 方法4: 尝试作为属性访问
    if old_mean is None:
        try:
            old_mean = getattr(run_stats, 'mean', None)
            old_var = getattr(run_stats, 'var', None)
            if hasattr(run_stats, 'count'):
                old_count = getattr(run_stats, 'count', jnp.array(1e6))
        except AttributeError:
            pass
    
    # 如果仍然无法访问，返回 None
    if old_mean is None:
        return None
    
    # 获取当前维度
    current_dim = old_mean.shape[0]
    
    # 如果维度已经匹配，直接返回
    if current_dim == target_dim:
        return run_stats
    
    # 调整维度
    if current_dim > target_dim:
        # 截取前 target_dim 维
        # 注意：这可能导致归一化不准确，因为统计值是基于完整维度计算的
        new_mean = old_mean[:target_dim]
        new_var = old_var[:target_dim]
    else:
        # 填充到 target_dim 维
        new_mean = jnp.concatenate([old_mean, jnp.zeros(target_dim - current_dim)])
        new_var = jnp.concatenate([old_var, jnp.ones(target_dim - current_dim)])
    
    # 创建新的统计字典
    new_stats_dict = {
        'mean': new_mean,
        'var': new_var,
        'count': old_count
    }
    
    # 如果是嵌套结构，保持嵌套
    if stats_key is not None:
        new_run_stats = {stats_key: new_stats_dict}
    else:
        new_run_stats = new_stats_dict
    
    return new_run_stats


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
    except Exception as e:
        print(f"❌ {traj_name} 测试失败: {e}")
        raise
    
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
            'goal_type': 'GoalTrajMimic',
            'goal_params': {'visualize_goal': False},
            'control_type': 'DefaultControl',
            'control_params': {'max_torque': 1.5, 'min_torque': -1.5},
            # 使用自定义的足球接近奖励函数（与训练时相同）
            'reward_type': 'FootballApproachReward',
            'reward_params': {
                # MimicReward 基础参数（与训练配置保持一致）
                'qpos_w_sum': 0.7,
                'qvel_w_sum': 0.34,
                'rpos_w_sum': 0.8,
                'rquat_w_sum': 0.5,
                'rvel_w_sum': 0.24,
                'upper_body_stability_w': 0.3,
                'waist_stability_w': 0.2,
                'sites_for_mimic': ['upper_body_mimic', 'left_hand_mimic',
                                     'left_foot_mimic', 'right_hand_mimic',
                                     'right_foot_mimic'],
                # 足球接近奖励参数（与训练配置保持一致）
                'football_approach_weight': 0.3,  # 足球接近奖励的权重
                'football_distance_scale': 0.5,   # 距离衰减尺度
            },
            # 使用自定义的初始状态处理器，将足球放在机器人正前方
            'initial_state_type': 'FootballInitialStateHandler',
            'initial_state_params': {
                'football_distance': 0.8,  # 足球距离机器人0.8米
                'football_height': 0.11,    # 足球高度0.11米（半径）
            }
        }
    else:
        print(f"✓ 加载训练配置: {config_path}")
        config = OmegaConf.load(config_path)
        env_params = OmegaConf.to_container(config.experiment.env_params,
                                             resolve=True)
        # 启用可视化，但保持与训练时完全相同的配置（包括 goal_type）
        env_params['headless'] = False
    
    # 3. 创建环境模板（轨迹会在后面动态设置）
    env_name = env_params.pop('env_name', 'MjxUnitreeG1')
    env_params['custom_dataset_conf'] = CustomDatasetConf(
        Trajectory.load(TRAJECTORY_PATHS[0]))  # 临时轨迹
    
    print(f"\n环境配置信息:")
    print(f"  - goal_type: {env_params.get('goal_type')}")
    print(f"  - goal_params: {env_params.get('goal_params')}")
    print(f"  - reward_params sites_for_mimic: "
          f"{env_params.get('reward_params', {}).get('sites_for_mimic')}")
    print(f"  - spec: {env_params.get('spec')}")
    
    env = ImitationFactory.make(env_name, **env_params)
    
    print("✓ 环境创建完成")
    actual_dim = env.info.observation_space.shape[0]
    print(f"  - 观测空间: ({actual_dim},)")
    print(f"  - 动作空间: {env.info.action_space.shape}")
    
    # 4. 检查并修复观测空间维度不匹配问题
    print(f"\n检查观测空间维度...")
    print(f"  - 环境观测空间维度: {actual_dim}")
    
    try:
        run_stats = agent_state.train_state.run_stats
        print(f"  - run_stats 类型: {type(run_stats)}")
        
        # 打印 run_stats 的内容（用于调试）
        if isinstance(run_stats, dict):
            print(f"  - run_stats 键: {list(run_stats.keys())}")
            for key, value in run_stats.items():
                if isinstance(value, dict):
                    print(f"    - {key}: {type(value)}")
                    for sub_key, sub_value in value.items():
                        if hasattr(sub_value, 'shape'):
                            print(f"      - {sub_key}: shape={sub_value.shape}, "
                                  f"dtype={sub_value.dtype}")
                        else:
                            print(f"      - {sub_key}: {type(sub_value)}")
                elif hasattr(value, 'shape'):
                    print(f"    - {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"    - {key}: {type(value)}")
        elif hasattr(run_stats, 'keys'):
            print(f"  - run_stats 键: {list(run_stats.keys())}")
        else:
            print(f"  - run_stats 内容: {run_stats}")
        
        # 获取训练时的维度
        expected_dim = None
        for key, value in run_stats.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if sub_key == 'mean' and hasattr(sub_value, 'shape'):
                        expected_dim = sub_value.shape[0]
                        break
                if expected_dim is not None:
                    break
        
        if expected_dim is not None:
            print(f"  - 训练时观测空间维度: {expected_dim}")
            
            if actual_dim != expected_dim:
                print(f"\n  ⚠️  维度不匹配：测试环境 {actual_dim} vs 训练时 {expected_dim}")
                print(f"  - 差异: {expected_dim - actual_dim} 维")
                print(f"  - 将尝试修复维度（截取前 {actual_dim} 维）")
                print(f"  - ⚠️  注意：截取维度可能导致归一化不准确，但可以避免报错")
                
                # 修复维度：截取前 actual_dim 维
                fixed_run_stats = fix_run_stats_dimension(run_stats, actual_dim)
                
                if fixed_run_stats is not None:
                    # 更新 train_state
                    new_train_state = agent_state.train_state.replace(
                        run_stats=fixed_run_stats)
                    # 更新 agent_state
                    if hasattr(agent_state, 'replace'):
                        agent_state = agent_state.replace(train_state=new_train_state)
                    else:
                        from dataclasses import replace
                        agent_state = replace(agent_state, train_state=new_train_state)
                    
                    # 验证修复
                    verify_run_stats = agent_state.train_state.run_stats
                    if isinstance(verify_run_stats, dict):
                        verify_mean = verify_run_stats.get(
                            'RunningMeanStd_0', {}).get('mean', None)
                        if verify_mean is not None:
                            verify_dim = verify_mean.shape[0]
                            if verify_dim == actual_dim:
                                print(f"  - ✓ 维度已修复：{verify_dim}")
                            else:
                                print(f"  - ❌ 修复失败：修复后维度 {verify_dim} 仍不匹配")
                        else:
                            print(f"  - ⚠️  无法验证修复结果")
                    else:
                        print(f"  - ⚠️  修复后 run_stats 类型不正确")
                else:
                    print(f"  - ❌ 无法修复维度，将直接报错")
            else:
                print(f"  - ✓ 维度匹配：{actual_dim}，可以正常测试")
        else:
            print(f"  - ⚠️  无法获取训练时的维度")
    except Exception as e:
        print(f"  - ⚠️  检查维度时出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("  - 幽灵机器人可视化已启用!")
    
    # 5. 最终验证 agent_state 的维度
    print(f"\n最终验证 agent_state 维度...")
    final_run_stats = agent_state.train_state.run_stats
    if isinstance(final_run_stats, dict) and 'RunningMeanStd_0' in final_run_stats:
        final_mean = final_run_stats['RunningMeanStd_0'].get('mean', None)
        if final_mean is not None:
            final_dim = final_mean.shape[0]
            print(f"  - 最终 run_stats mean 维度: {final_dim}")
            print(f"  - 环境观测空间维度: {actual_dim}")
            if final_dim == actual_dim:
                print(f"  - ✓ 维度匹配，可以正常测试")
            else:
                print(f"  - ❌ 维度仍不匹配！这会导致身体扭曲问题")
                print(f"  - 建议：检查维度修复代码是否正确执行")
    
    # 6. 依次测试每个轨迹
    print(f"\n{'='*70}")
    print(f"开始测试 {len(TRAJECTORY_PATHS)} 个轨迹文件")
    print(f"{'='*70}")
    
    for i, traj_path in enumerate(TRAJECTORY_PATHS):
        traj_name = f"11-18-{i+1}"
        try:
            test_single_trajectory(env, agent_conf, agent_state, traj_path,
                                   traj_name)
        except Exception as e:
            print(f"❌ 测试 {traj_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*70}")
    print("所有轨迹测试完成!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
