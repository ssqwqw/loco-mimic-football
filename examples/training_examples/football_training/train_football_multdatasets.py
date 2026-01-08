#!/usr/bin/env python3
"""
训练 UnitreeG1 模仿足球动作
"""

import os
import sys
sys.path.insert(0, '/home/user/loco-mujoco')  # 强制优先使用这个路径
import jax
import jax.numpy as jnp
import wandb
from dataclasses import fields

from loco_mujoco import TaskFactory
from loco_mujoco.task_factories import ImitationFactory, CustomDatasetConf
from loco_mujoco.trajectory import Trajectory
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.utils.metrics import QuantityContainer
from loco_mujoco.utils import MetricsHandler

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import traceback

# ⭐ 导入自定义的奖励函数和初始状态处理器，确保在训练时被注册
training_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, training_dir)
from football_reward import FootballApproachReward
from football_init_state import FootballInitialStateHandler


@hydra.main(version_base=None, config_path="./", config_name="conf_football")
def experiment(config: DictConfig):
    try:
        os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'

        result_dir = HydraConfig.get().runtime.output_dir
        
        # Setup wandb
        wandb.login()
        config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        run = wandb.init(project=config.wandb.project, config=config_dict)

        print("=" * 70)
        print("LOADING FOOTBALL TRAJECTORIES")
        print("=" * 70)
        
        # ⭐ 定义要加载的轨迹文件列表
        traj_files = [
            "/home/user/loco-mujoco/examples/training_examples/football_training/datasets-npz/11-18-1.npz",
            "/home/user/loco-mujoco/examples/training_examples/football_training/datasets-npz/11-18-2.npz",
            "/home/user/loco-mujoco/examples/training_examples/football_training/datasets-npz/11-18-3.npz",
            "/home/user/loco-mujoco/examples/training_examples/football_training/datasets-npz/11-18-4.npz",
            "/home/user/loco-mujoco/examples/training_examples/football_training/datasets-npz/11-18-5.npz",

        ]
        
        # 加载所有轨迹
        all_trajs = []
        for traj_file in traj_files:
            if os.path.exists(traj_file):
                traj = Trajectory.load(traj_file)
                all_trajs.append(traj)
                print(f"✓ Loaded trajectory from: {traj_file}")
                print(f"  - Frames: {traj.data.qpos.shape[0]}")
                print(f"  - Duration: {traj.data.qpos.shape[0] / traj.info.frequency:.2f}s")
            else:
                print(f"⚠️  File not found: {traj_file}, skipping...")
        
        if not all_trajs:
            raise ValueError("No trajectory files found!")
        
        # ⭐ 合并所有轨迹
        combined_traj = Trajectory.concatenate(all_trajs)
        
        print(f"\n✓ Combined {len(all_trajs)} trajectories:")
        print(f"  - Total frames: {combined_traj.data.qpos.shape[0]}")
        print(f"  - Total duration: {combined_traj.data.qpos.shape[0] / combined_traj.info.frequency:.2f}s")
        print(f"  - Frequency: {combined_traj.info.frequency} Hz")

        print("\n" + "=" * 70)
        print("CREATING ENVIRONMENT")
        print("=" * 70)
        
        # ⭐ 创建环境（使用合并后的轨迹）
        env = ImitationFactory.make(
            **config.experiment.env_params,
            custom_dataset_conf=CustomDatasetConf(combined_traj)
        )
        
        print(f"✓ Environment created: {config.experiment.env_params.env_name}")
        print(f"  - Observation space: {env.info.observation_space.shape}")
        print(f"  - Action space: {env.info.action_space.shape}")



        # 初始化 agent 配置
        agent_conf = PPOJax.init_agent_conf(env, config)
        print(f"✓ Agent configuration initialized")
        print(f"  - Hidden layers: {config.experiment.hidden_layers}")
        print(f"  - Learning rate: {config.experiment.lr}")
        print(f"  - Number of environments: {config.experiment.num_envs}")
        print(f"  - Total timesteps: {config.experiment.total_timesteps:.0e}")

        # 设置 metric handler
        mh = MetricsHandler(config, env) if config.experiment.validation.active else None

        # 构建训练函数
        train_fn = PPOJax.build_train_fn(env, agent_conf, mh=mh)

        # JIT 编译和向量化
        if config.experiment.n_seeds > 1:
            train_fn = jax.jit(jax.vmap(train_fn))
        else:
            train_fn = jax.jit(train_fn)

        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        
        # 运行训练
        rngs = [jax.random.PRNGKey(i) for i in range(config.experiment.n_seeds + 1)]
        rng, _rng = rngs[0], jnp.squeeze(jnp.vstack(rngs[1:]))
        
        out = train_fn(_rng)

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)

        # 保存 agent
        agent_state = out["agent_state"]
        save_path = PPOJax.save_agent(result_dir, agent_conf, agent_state)
        run.config.update({"agent_save_path": save_path})
        print(f"✓ Agent saved to: {save_path}")

        # 记录指标
        if not config.experiment.debug:
            print("\n" + "=" * 70)
            print("LOGGING METRICS")
            print("=" * 70)
            
            training_metrics = out["training_metrics"]
            validation_metrics = out["validation_metrics"]

            # 计算多个 seed 的平均值
            training_metrics = jax.tree.map(
                lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), 
                training_metrics
            )
            validation_metrics = jax.tree.map(
                lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), 
                validation_metrics
            )

            # 获取奖励配置参数
            reward_params = config.experiment.env_params.reward_params
            football_approach_weight = reward_params.get('football_approach_weight', 0.3)
            football_distance_scale = reward_params.get('football_distance_scale', 0.5)
            
            for i in range(len(training_metrics.mean_episode_return)):
                # 基础指标
                log_dict = {
                    "Mean Episode Return": training_metrics.mean_episode_return[i],
                    "Mean Episode Length": training_metrics.mean_episode_length[i]
                }
                
                # 添加奖励配置信息（用于参考）
                log_dict.update({
                    "Reward Config/Football Approach Weight": football_approach_weight,
                    "Reward Config/Football Distance Scale": football_distance_scale,
                    "Reward Config/Qpos Weight": reward_params.get('qpos_w_sum', 0.7),
                    "Reward Config/Qvel Weight": reward_params.get('qvel_w_sum', 0.34),
                    "Reward Config/Rpos Weight": reward_params.get('rpos_w_sum', 0.8),
                    "Reward Config/Rquat Weight": reward_params.get('rquat_w_sum', 0.5),
                    "Reward Config/Rvel Weight": reward_params.get('rvel_w_sum', 0.24),
                    "Reward Config/Upper Body Stability Weight": reward_params.get('upper_body_stability_w', 0.3),
                    "Reward Config/Waist Stability Weight": reward_params.get('waist_stability_w', 0.2),
                })
                
                # 计算奖励组件的估计值（基于总奖励和配置参数）
                # 注意：这是估计值，实际值可能略有不同
                total_return = training_metrics.mean_episode_return[i]
                
                # 估计基础奖励和足球奖励的比例
                # 假设基础奖励占总奖励的 (1 - football_weight) 部分
                # 这是一个简化的估计，实际值可能更复杂
                estimated_base_reward = total_return * (1 - football_approach_weight) / (1 + football_approach_weight)
                estimated_football_reward = total_return * football_approach_weight / (1 + football_approach_weight)
                
                # 添加奖励组件到日志
                log_dict.update({
                    "Reward Components/Base Reward (estimated)": estimated_base_reward,
                    "Reward Components/Football Reward (estimated)": estimated_football_reward,
                    "Reward Components/Football Reward Weighted (estimated)": estimated_football_reward * football_approach_weight,
                    "Reward Components/Total Reward": total_return,
                })
                
                run.log(log_dict, step=int(training_metrics.max_timestep[i]))

                if ((i + 1) % config.experiment.validation_interval == 0 and 
                    config.experiment.validation.active):
                    
                    # 验证指标
                    val_log_dict = {
                        "Validation Info/Mean Episode Return": validation_metrics.mean_episode_return[i],
                        "Validation Info/Mean Episode Length": validation_metrics.mean_episode_length[i]
                    }
                    
                    # 添加验证阶段的奖励组件估计值
                    val_total_return = validation_metrics.mean_episode_return[i]
                    
                    val_estimated_base_reward = val_total_return * (1 - football_approach_weight) / (1 + football_approach_weight)
                    val_estimated_football_reward = val_total_return * football_approach_weight / (1 + football_approach_weight)
                    
                    val_log_dict.update({
                        "Validation Info/Reward Components/Base Reward (estimated)": val_estimated_base_reward,
                        "Validation Info/Reward Components/Football Reward (estimated)": val_estimated_football_reward,
                        "Validation Info/Reward Components/Football Reward Weighted (estimated)": val_estimated_football_reward * football_approach_weight,
                        "Validation Info/Reward Components/Total Reward": val_total_return,
                    })
                    
                    run.log(val_log_dict, step=int(training_metrics.max_timestep[i]))

                    # 记录所有度量
                    metrics_to_log = {}
                    for field in fields(validation_metrics):
                        attr = getattr(validation_metrics, field.name)
                        if isinstance(attr, QuantityContainer):
                            measure_name = field.name
                            for field_attr in fields(attr):
                                attr_name = field_attr.name
                                attr_value = getattr(attr, attr_name)
                                if attr_value.size > 0:
                                    metrics_to_log[f"Validation Measures/{measure_name}/{attr_name}"] = attr_value[i]

                    run.log(metrics_to_log, step=int(training_metrics.max_timestep[i]))

        # 生成演示视频
        print("\n" + "=" * 70)
        print("GENERATING DEMO VIDEO")
        print("=" * 70)
        
        PPOJax.play_policy(
            env, agent_conf, agent_state, 
            deterministic=True, 
            n_steps=200, 
            n_envs=20, 
            record=True,
            train_state_seed=0
        )
        video_file = env.video_file_path
        run.log({"Agent Video": wandb.Video(video_file)})
        print(f"✓ Video saved: {video_file}")

        # 在 wandb.finish() 之前，打印所有的指标值（包括最后的 summary）
        print("\n" + "=" * 70)
        print("WANDB RUN SUMMARY - 所有指标")
        print("=" * 70)
        
        # 获取 run 的 summary（所有记录的指标）
        summary = run.summary
        
        # 打印所有指标，按字母顺序排序
        if summary:
            print("\n所有指标值（按字母顺序排序）：")
            print("-" * 70)
            # 获取所有键并按字母顺序排序
            sorted_keys = sorted(summary.keys())
            for key in sorted_keys:
                value = summary[key]
                # 跳过内部指标（以下划线开头）
                if key.startswith('_'):
                    continue
                # 格式化显示
                if isinstance(value, (int, float)):
                    # 如果是数字，格式化为6位小数
                    if abs(value) >= 1000 or (abs(value) < 0.001 and value != 0):
                        print(f"  {key:50} {value:>15.6e}")
                    else:
                        print(f"  {key:50} {value:>15.6f}")
                elif isinstance(value, bool):
                    print(f"  {key:50} {str(value):>15}")
                elif isinstance(value, (list, tuple)):
                    print(f"  {key:50} {str(value)[:50]:>15}")
                else:
                    print(f"  {key:50} {str(value)[:50]:>15}")
            print("-" * 70)
            # 统计显示的指标数量（排除内部指标）
            displayed_count = len([k for k in sorted_keys if not k.startswith('_')])
            print(f"\n总共显示了 {displayed_count} 个指标（已排除内部指标）")
            print(f"总共有 {len(sorted_keys)} 个指标（包括内部指标）")
        else:
            print("⚠️  没有找到 summary 数据")
        
        print("=" * 70)
        print()

        wandb.finish()
        
        print("\n" + "=" * 70)
        print("ALL DONE! 🎉")
        print("=" * 70)

    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    experiment()