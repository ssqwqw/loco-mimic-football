"""
自定义奖励函数：奖励机器人靠近足球
"""
from types import ModuleType
from typing import Any, Dict, Tuple, Union

import numpy as np
import jax.numpy as jnp
import mujoco
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.reward.trajectory_based import MimicReward, MimicRewardState
from loco_mujoco.core.utils import mj_jntname2qposid


class FootballApproachReward(MimicReward):
    """
    扩展MimicReward，添加靠近足球的奖励。
    
    奖励计算：
    - 基础奖励：MimicReward的所有奖励
    - 足球接近奖励：根据机器人与足球的距离给予奖励
      - 距离越近，奖励越高
      - 使用指数衰减函数：exp(-distance / scale)
    """
    
    def __init__(self, env: Any,
                 football_approach_weight: float = 0.3,
                 football_distance_scale: float = 0.5,
                 **kwargs):
        """
        初始化足球接近奖励函数。
        
        Args:
            env (Any): 环境实例
            football_approach_weight (float): 足球接近奖励的权重，默认0.3
            football_distance_scale (float): 距离衰减的尺度参数，默认0.5（越小衰减越快）
            **kwargs: MimicReward的其他参数
        """
        super().__init__(env, **kwargs)
        
        self._football_approach_weight = football_approach_weight
        self._football_distance_scale = football_distance_scale
        
        # 获取足球body ID
        model = env._model
        self._football_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "football")
        if self._football_body_id < 0:
            raise ValueError("未找到足球body 'football'，请检查XML文件")
        
        # 获取机器人根body ID
        root_body_name = self._info_props["root_body_name"]
        self._root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)
        if self._root_body_id < 0:
            raise ValueError(f"未找到根body '{root_body_name}'")
    
    def __call__(self,
                 state: Union[np.ndarray, jnp.ndarray],
                 action: Union[np.ndarray, jnp.ndarray],
                 next_state: Union[np.ndarray, jnp.ndarray],
                 absorbing: bool,
                 info: Dict[str, Any],
                 env: Any,
                 model: Union[MjModel, Model],
                 data: Union[MjData, Data],
                 carry: Any,
                 backend: ModuleType) -> Tuple[float, Any]:
        """
        计算奖励，包括MimicReward的基础奖励和足球接近奖励。
        
        Args:
            state: 上一个状态
            action: 执行的动作
            next_state: 当前状态
            absorbing: 是否终止
            info: 额外信息
            env: 环境实例
            model: MuJoCo模型
            data: MuJoCo数据
            carry: 额外的状态信息
            backend: 后端模块（numpy或jax.numpy）
        
        Returns:
            Tuple[float, Any]: 奖励值和更新后的carry
        """
        # 先计算基础的MimicReward
        base_reward, carry = super().__call__(
            state, action, next_state, absorbing, info, env, model, data, carry, backend
        )
        
        # 计算足球接近奖励
        # 获取机器人和足球的位置
        robot_pos = data.xpos[self._root_body_id]  # 机器人根位置
        football_pos = data.xpos[self._football_body_id]  # 足球位置
        
        # 计算水平距离（忽略高度差）
        # 处理向量化环境：在 Mjx 中，data.xpos 可能是 2D 数组 (n_envs, 3)
        if backend == jnp:
            # JAX/Mjx 环境：可能是向量化的
            if robot_pos.ndim == 2:
                # 向量化环境：robot_pos 形状为 (n_envs, 3)
                distance_2d = jnp.linalg.norm(robot_pos[:, :2] - football_pos[:, :2], axis=1)
            else:
                # 单个环境：robot_pos 形状为 (3,)
                distance_2d = jnp.linalg.norm(robot_pos[:2] - football_pos[:2])
        else:
            # NumPy 环境：单个环境
            distance_2d = np.linalg.norm(robot_pos[:2] - football_pos[:2])
        
        # 计算奖励：距离越近，奖励越高
        # 使用指数衰减：exp(-distance / scale)
        # 当distance=0时，reward=1；当distance=scale时，reward≈0.37
        football_reward = backend.exp(-distance_2d / self._football_distance_scale)
        football_reward_weighted = self._football_approach_weight * football_reward
        
        # 组合奖励
        total_reward = base_reward + football_reward_weighted
        
        # 确保奖励非负
        total_reward = backend.maximum(total_reward, 0.0)
        
        # 处理NaN值
        total_reward = backend.nan_to_num(total_reward, nan=0.0)
        football_reward = backend.nan_to_num(football_reward, nan=0.0)
        football_reward_weighted = backend.nan_to_num(football_reward_weighted, nan=0.0)
        base_reward = backend.nan_to_num(base_reward, nan=0.0)
        
        return total_reward, carry


# 注册奖励函数
FootballApproachReward.register()

