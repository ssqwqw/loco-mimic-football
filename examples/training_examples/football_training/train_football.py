#!/usr/bin/env python3
"""
训练 UnitreeG1 模仿足球动作
"""

import os
import sys
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
        print("LOADING FOOTBALL TRAJECTORY")
        print("=" * 70)
        
        # 加载自定义轨迹
        traj_path = "/home/user/loco-mujoco/examples/training_examples/football_training/datasets-npz/football_29dof.npz"
        traj = Trajectory.load(traj_path)
        print(f"✓ Loaded trajectory from: {traj_path}")
        print(f"  - Number of frames: {traj.data.qpos.shape[0]}")
        print(f"  - Frequency: {traj.info.frequency} Hz")
        print(f"  - Duration: {traj.data.qpos.shape[0] / traj.info.frequency:.2f} seconds")

        print("\n" + "=" * 70)
        print("CREATING ENVIRONMENT")
        print("=" * 70)
        
        # 创建环境
        env = ImitationFactory.make(
            **config.experiment.env_params,
            custom_dataset_conf=CustomDatasetConf(traj)
        )
        
        print(f"✓ Environment created: {config.experiment.env_params.env_name}")
        print(f"  - Observation space: {env.info.observation_space.shape}")
        print(f"  - Action space: {env.info.action_space.shape}")

        print("\n" + "=" * 70)
        print("INITIALIZING TRAINING")
        print("=" * 70)
        
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

            for i in range(len(training_metrics.mean_episode_return)):
                run.log({
                    "Mean Episode Return": training_metrics.mean_episode_return[i],
                    "Mean Episode Length": training_metrics.mean_episode_length[i]
                }, step=int(training_metrics.max_timestep[i]))

                if ((i + 1) % config.experiment.validation_interval == 0 and 
                    config.experiment.validation.active):
                    
                    run.log({
                        "Validation Info/Mean Episode Return": validation_metrics.mean_episode_return[i],
                        "Validation Info/Mean Episode Length": validation_metrics.mean_episode_length[i]
                    }, step=int(training_metrics.max_timestep[i]))

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

        wandb.finish()
        
        print("\n" + "=" * 70)
        print("ALL DONE! 🎉")
        print("=" * 70)

    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    experiment()