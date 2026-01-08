import mujoco
from mujoco import MjSpec
from mujoco.mjx import Model, Data
import jax.numpy as jnp
from typing import Tuple

from .unitreeG1 import UnitreeG1
from loco_mujoco.core.mujoco_mjx import MjxAdditionalCarry


class MjxUnitreeG1(UnitreeG1):

    mjx_enabled = True

    def __init__(self, timestep=0.002, n_substeps=5, **kwargs):
        if "model_option_conf" not in kwargs.keys():
            model_option_conf = dict(iterations=2, ls_iterations=4, disableflags=mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
        else:
            model_option_conf = kwargs["model_option_conf"]
            del kwargs["model_option_conf"]
        super().__init__(timestep=timestep, n_substeps=n_substeps, model_option_conf=model_option_conf, **kwargs)
        
        # 获取足球body ID和mocap ID
        self._football_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "football")
        if self._football_body_id >= 0:
            self._football_mocap_id = self._model.body_mocapid[self._football_body_id]
            if self._football_mocap_id < 0:
                self._football_mocap_id = None
        else:
            self._football_mocap_id = None

    def _modify_spec_for_mjx(self, spec: MjSpec):
        """
        Mjx is bad in handling many complex contacts. To speed-up simulation significantly we apply
        some changes to the XML:
            1. Replace the complex foot meshes with primitive shapes. Here, one foot mesh is replaced with
               two capsules.
            2. Disable all contacts except the ones between feet and the floor.
            3. Keep the football visible (disable its contacts to avoid Mjx constraint issues).

        Args:
            spec: Handle to Mujoco XML.

        Returns:
            Mujoco XML handle.

        """

        foot_geoms = ["right_foot_1_col", "right_foot_2_col", "right_foot_3_col", "right_foot_4_col",
                      "left_foot_1_col", "left_foot_2_col", "left_foot_3_col", "left_foot_4_col"]

        # --- Make all geoms have contype and conaffinity of 0 ---
        for g in spec.geoms:
            # Disable contacts for all geoms by default
            g.contype = 0
            g.conaffinity = 0

        # --- Define specific contact pairs ---
        for g_name in foot_geoms:
            spec.add_pair(geomname1="floor", geomname2=g_name)
        
        # 注意：暂时禁用足球与地面的碰撞，避免MJX约束问题
        # 足球仍然可见，但不会与地面发生物理交互
        # 如果需要足球物理交互，可以考虑使用CPU版本的MuJoCo
        # spec.add_pair(geomname1="floor", geomname2="football_geom")

        return spec
    
    def _mjx_reset_carry(self, model: Model,
                         data: Data,
                         carry: MjxAdditionalCarry) -> Tuple[Data, MjxAdditionalCarry]:
        """
        重写以设置每个环境实例的球位置。
        注意：足球使用mocap body，位置由初始状态处理器或这里设置。
        
        Args:
            model (Model): Mujoco model.
            data (Data): Mujoco data structure.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            Tuple[Data, MjxAdditionalCarry]: Updated data and carry.
        """
        # 调用父类方法
        data, carry = super()._mjx_reset_carry(model, data, carry)
        
        # 如果足球mocap ID存在，设置足球位置（作为备用，初始状态处理器也会设置）
        if self._football_mocap_id is not None:
            # 检查 data.qpos 的维度
            qpos_ndim = data.qpos.ndim
            
            if qpos_ndim == 1:
                # 单个环境（未向量化）
                robot_x = data.qpos[0]  # 机器人的 x 位置
                robot_y = data.qpos[1]  # 机器人的 y 位置
                
                # 球的位置：机器人前方 0.8 米，高度 0.11 米
                football_pos = jnp.array([robot_x + 0.8, robot_y, 0.11])
                
                # 设置球的 mocap 位置
                data = data.replace(
                    mocap_pos=data.mocap_pos.at[self._football_mocap_id, :].set(football_pos)
                )
            else:
                # 多个环境（已向量化，qpos 形状为 (n_envs, n_qpos)）
                n_envs = data.qpos.shape[0]
                # 获取每个环境实例的机器人根位置（前两个元素是 x, y）
                robot_x = data.qpos[:, 0]  # 机器人的 x 位置
                robot_y = data.qpos[:, 1]  # 机器人的 y 位置
                
                # 球的位置：机器人前方 0.8 米，高度 0.11 米
                football_pos = jnp.stack([
                    robot_x + 0.8,  # 机器人前方 0.8 米
                    robot_y,        # 与机器人相同的 y 位置
                    jnp.full(n_envs, 0.11)  # 高度 0.11 米
                ], axis=1)
                
                # 设置球的 mocap 位置（每个环境实例独立）
                data = data.replace(
                    mocap_pos=data.mocap_pos.at[:, self._football_mocap_id, :].set(football_pos)
                )
        
        return data, carry
