"""
自定义初始状态处理器：将足球放在机器人正前方
"""
from typing import Any, Union, Tuple
from types import ModuleType

import numpy as np
import jax.numpy as jnp
import mujoco
from mujoco import MjData, MjModel
from mujoco.mjx import Data, Model

from loco_mujoco.core.utils import assert_backend_is_supported
from loco_mujoco.core.initial_state_handler.base import InitialStateHandler


class FootballInitialStateHandler(InitialStateHandler):
    """
    自定义初始状态处理器，将足球放在机器人正前方合适的位置。
    
    足球位置：
    - 在机器人前方 0.8-1.2 米（可配置）
    - 高度 0.11 米（足球半径）
    - 与机器人相同的 y 坐标
    """
    
    def __init__(self, env: Any, 
                 football_distance: float = 0.8,
                 football_height: float = 0.11,
                 qpos_init=None, 
                 qvel_init=None):
        """
        初始化足球初始状态处理器。
        
        Args:
            env (Any): 环境实例
            football_distance (float): 足球距离机器人的距离（米），默认0.8米
            football_height (float): 足球的高度（米），默认0.11米
            qpos_init: 初始关节位置（可选）
            qvel_init: 初始关节速度（可选）
        """
        super().__init__(env)
        
        self.football_distance = football_distance
        self.football_height = football_height
        
        # 获取足球body ID（使用mocap body）
        model = env._model
        self._football_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "football")
        if self._football_body_id < 0:
            raise ValueError("未找到足球body 'football'，请检查XML文件")
        
        # 获取足球的mocap ID
        self._football_mocap_id = model.body_mocapid[self._football_body_id]
        if self._football_mocap_id < 0:
            raise ValueError("足球body不是mocap body，请检查XML文件")
        
        # 获取机器人根关节的索引
        root_joint_name = env._get_all_info_properties()["root_free_joint_xml_name"]
        root_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, root_joint_name)
        if root_joint_id < 0:
            raise ValueError(f"未找到根关节 '{root_joint_name}'")
        
        self._root_qpos_start = model.jnt_qposadr[root_joint_id]
        self._root_qvel_start = model.jnt_dofadr[root_joint_id]
        
        # 保存可选的初始关节位置和速度
        self.qpos_init = np.array(qpos_init) if qpos_init is not None else None
        self.qvel_init = np.array(qvel_init) if qvel_init is not None else None
    
    def reset(self, 
              env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType) -> Tuple[Union[MjData, Data], Any]:
        """
        重置环境，将足球放在机器人正前方。
        
        Args:
            env (Any): 环境实例
            model (Union[MjModel, Model]): MuJoCo模型
            data (Union[MjData, Data]): MuJoCo数据
            carry (Any): 额外的状态信息
            backend (ModuleType): 后端模块（numpy或jax.numpy）
        
        Returns:
            Tuple[Union[MjData, Data], Any]: 更新后的数据和carry
        """
        assert_backend_is_supported(backend)
        
        # 如果提供了初始关节位置，先设置它们
        if self.qpos_init is not None:
            data = self.set_qpos(self.qpos_init, data, backend)
        if self.qvel_init is not None:
            data = self.set_qvel(self.qvel_init, data, backend)
        
        # 获取机器人根位置（x, y, z）
        if backend == np:
            robot_pos = data.qpos[self._root_qpos_start:self._root_qpos_start + 3]
            robot_quat = data.qpos[self._root_qpos_start + 3:self._root_qpos_start + 7]
        else:
            robot_pos = data.qpos[self._root_qpos_start:self._root_qpos_start + 3]
            robot_quat = data.qpos[self._root_qpos_start + 3:self._root_qpos_start + 7]
        
        # 计算机器人朝向（从四元数提取yaw角）
        # MuJoCo四元数格式：[w, x, y, z]，scipy使用[x, y, z, w]
        if backend == np:
            from scipy.spatial.transform import Rotation as R
            # 转换格式：MuJoCo [w,x,y,z] -> scipy [x,y,z,w]
            quat_scipy = np.array([robot_quat[1], robot_quat[2], robot_quat[3], robot_quat[0]])
            rot = R.from_quat(quat_scipy)
            euler = rot.as_euler('zyx')
            yaw = euler[0]  # z轴旋转（yaw角）
            forward_dir = np.array([np.cos(yaw), np.sin(yaw), 0])
        else:
            from jax._src.scipy.spatial.transform import Rotation as R
            # 转换格式：MuJoCo [w,x,y,z] -> scipy [x,y,z,w]
            quat_scipy = jnp.array([robot_quat[1], robot_quat[2], robot_quat[3], robot_quat[0]])
            rot = R.from_quat(quat_scipy)
            euler = rot.as_euler('zyx')
            yaw = euler[0]  # z轴旋转（yaw角）
            forward_dir = jnp.array([jnp.cos(yaw), jnp.sin(yaw), 0])
        
        # 计算足球位置：机器人位置 + 前方方向 * 距离
        football_pos = robot_pos + forward_dir * self.football_distance
        football_pos = backend.array([
            football_pos[0],
            football_pos[1],
            self.football_height  # 固定高度
        ])
        
        # 设置足球的位置（mocap body使用mocap_pos）
        if backend == np:
            data.mocap_pos[self._football_mocap_id] = football_pos
            # 设置足球的四元数（单位四元数，表示无旋转）
            data.mocap_quat[self._football_mocap_id] = backend.array([1, 0, 0, 0])
        else:
            # JAX版本：使用replace
            new_mocap_pos = data.mocap_pos.at[self._football_mocap_id, :].set(football_pos)
            new_mocap_quat = data.mocap_quat.at[self._football_mocap_id, :].set(
                backend.array([1, 0, 0, 0])
            )
            data = data.replace(mocap_pos=new_mocap_pos, mocap_quat=new_mocap_quat)
        
        return data, carry
    
    @staticmethod
    def set_qpos(qpos: Union[np.ndarray, jnp.ndarray],
                 data: Union[MjData, Data],
                 backend: ModuleType) -> Union[MjData, Data]:
        """设置关节位置"""
        if backend == np:
            data.qpos[:] = qpos
        else:
            data = data.replace(qpos=data.qpos.at[:].set(qpos))
        return data
    
    @staticmethod
    def set_qvel(qvel: Union[np.ndarray, jnp.ndarray],
                 data: Union[MjData, Data],
                 backend: ModuleType) -> Union[MjData, Data]:
        """设置关节速度"""
        if backend == np:
            data.qvel[:] = qvel
        else:
            data = data.replace(qvel=data.qvel.at[:].set(qvel))
        return data


# 注册初始状态处理器
FootballInitialStateHandler.register()

