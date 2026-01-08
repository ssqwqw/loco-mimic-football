#!/usr/bin/env python3
"""
将 pkl 转换为 Loco-MuJoCo 的 npz 格式 (29自由度G1版本)

使用方法:
    python convert_pkl_to_npz_29dof.py
"""

import pickle
import numpy as np
import jax.numpy as jnp
import mujoco
from pathlib import Path

from loco_mujoco.trajectory import Trajectory, TrajectoryInfo, TrajectoryModel, TrajectoryData
from loco_mujoco.environments import LocoEnv


def quaternion_xyzw_to_wxyz(quat_xyzw):
    """
    将四元数从 (x, y, z, w) 格式转换为 MuJoCo 的 (w, x, y, z) 格式
    
    Args:
        quat_xyzw: shape (N, 4) 或 (4,)，格式为 [x, y, z, w]
    
    Returns:
        quat_wxyz: shape (N, 4) 或 (4,)，格式为 [w, x, y, z]
    """
    if quat_xyzw.ndim == 1:
        return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    else:
        return np.column_stack([quat_xyzw[:, 3], quat_xyzw[:, 0], 
                                quat_xyzw[:, 1], quat_xyzw[:, 2]])


def load_and_convert_pkl(pkl_path: str, env_name: str = "UnitreeG1", xml_path: str = None):
    """
    加载 pkl 文件并转换为 Trajectory 对象
    
    Args:
        pkl_path: pkl 文件路径
        env_name: 环境名称
        xml_path: 可选的XML文件路径（如果指定，将使用29自由度XML）
    
    Returns:
        Trajectory 对象
    """
    
    print("=" * 70)
    print("LOADING PKL FILE")
    print("=" * 70)
    
    # 1. 加载 pkl 文件
    print(f"Loading: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # 2. 提取数据
    fps = data['fps']
    root_pos = data['root_pos']  # (N, 3)
    root_rot = data['root_rot']  # (N, 4) - [x, y, z, w]
    dof_pos = data['dof_pos']    # (N, 29) - 29自由度
    
    print(f"✓ Loaded successfully!")
    print(f"  - FPS: {fps}")
    print(f"  - Number of frames: {root_pos.shape[0]}")
    print(f"  - Root position shape: {root_pos.shape}")
    print(f"  - Root rotation shape: {root_rot.shape}")
    print(f"  - DOF position shape: {dof_pos.shape}")
    
    # 3. 转换四元数格式
    print("\nConverting quaternion format from [x,y,z,w] to [w,x,y,z]...")
    root_rot_wxyz = quaternion_xyzw_to_wxyz(root_rot)
    print(f"✓ Quaternion conversion complete")
    
    # 4. 获取环境信息
    print(f"\n" + "=" * 70)
    print(f"GETTING ENVIRONMENT INFO: {env_name}")
    print("=" * 70)
    
    env_cls = LocoEnv.registered_envs[env_name]
    
    # 如果指定了XML路径，使用它
    if xml_path is not None:
        print(f"Using custom XML: {xml_path}")
        env = env_cls(spec=xml_path)
    else:
        env = env_cls()
    
    model = env.get_model()
    
    expected_qpos_dim = model.nq
    expected_dof_dim = expected_qpos_dim - 7  # 减去 root (3 pos + 4 rot)
    
    print(f"✓ Environment configuration:")
    print(f"  - Expected total qpos dimension: {expected_qpos_dim}")
    print(f"  - Expected DOF dimension (excluding root): {expected_dof_dim}")
    print(f"  - Your DOF dimension: {dof_pos.shape[1]}")
    
    # 5. 维度检查
    if dof_pos.shape[1] != expected_dof_dim:
        print(f"\n⚠️  Dimension mismatch detected!")
        print(f"  Expected DOF: {expected_dof_dim}, Got: {dof_pos.shape[1]}")
        diff = dof_pos.shape[1] - expected_dof_dim
        
        if diff > 0:
            print(f"  Your data has {diff} extra DOF(s).")
            print(f"  Attempting to remove the last {diff} DOF(s)...")
            dof_pos = dof_pos[:, :expected_dof_dim]
            print(f"  ✓ Trimmed dof_pos to shape: {dof_pos.shape}")
        elif diff < 0:
            print(f"  Your data has {abs(diff)} fewer DOF(s).")
            print(f"  This cannot be automatically fixed.")
            raise ValueError(f"Cannot fix dimension mismatch: expected {expected_dof_dim}, got {dof_pos.shape[1]}")
    else:
        print(f"\n✓ DOF dimensions match!")
    
    # 6. 组合 qpos: [root_pos(3), root_rot(4), dof_pos(29)] = 36维
    qpos = np.concatenate([root_pos, root_rot_wxyz, dof_pos], axis=1)
    print(f"\n✓ Combined qpos shape: {qpos.shape}")
    
    # 7. 计算 qvel (使用有限差分)
    print("\nComputing qvel using finite differences...")
    dt = 1.0 / fps
    n_frames = qpos.shape[0]
    qvel = np.zeros((n_frames, qpos.shape[1]))
    
    if n_frames > 2:
        qvel[1:-1] = (qpos[2:] - qpos[:-2]) / (2 * dt)
    if n_frames > 1:
        qvel[0] = (qpos[1] - qpos[0]) / dt
        qvel[-1] = (qpos[-1] - qpos[-2]) / dt
    
    print(f"✓ Computed qvel shape: {qvel.shape}")
    
    # 8. 获取关节信息
    joint_names = []
    joint_types = []
    
    for i in range(model.njnt):
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        jnt_type = model.jnt_type[i]
        joint_names.append(jnt_name)
        joint_types.append(jnt_type)
    
    print(f"\n✓ Environment has {len(joint_names)} joints")
    
    # 最终维度检查
    if qpos.shape[1] != model.nq:
        raise ValueError(f"Dimension mismatch: expected {model.nq}, got {qpos.shape[1]}")
    else:
        print(f"\n✓ Dimensions match perfectly!")
    
    # 9. 创建 TrajectoryModel
    traj_model = TrajectoryModel(
        njnt=len(joint_names),
        jnt_type=jnp.array(joint_types)
    )
    
    # 10. 创建 TrajectoryInfo
    traj_info = TrajectoryInfo(
        joint_names=joint_names,
        model=traj_model,
        frequency=float(fps)
    )
    
    # 11. 创建 TrajectoryData
    traj_data = TrajectoryData(
        qpos=jnp.array(qpos),
        qvel=jnp.array(qvel),
        split_points=jnp.array([0, n_frames])
    )
    
    # 12. 创建初始 Trajectory
    trajectory = Trajectory(info=traj_info, data=traj_data)
    
    print("\n" + "=" * 70)
    print("EXTENDING TRAJECTORY WITH PHYSICS DATA")
    print("=" * 70)
    
    # 扩展轨迹以包含完整物理数据
    from loco_mujoco.datasets.humanoids.LAFAN1.load import extend_motion
    from loco_mujoco.smpl.retargeting import load_robot_conf_file
    
    print("Computing forward kinematics...")
    
    try:
        robot_conf = load_robot_conf_file(env_name)
        trajectory = extend_motion(
            env_name=env_name,
            robot_conf=robot_conf,
            traj=trajectory,
            replace_qvel_with_finite_diff=False
        )
        
        print("✓ Trajectory extended with complete physics data")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not extend trajectory: {e}")
        print("Trajectory will only contain qpos and qvel.")
    
    print("\n" + "=" * 70)
    print("TRAJECTORY CREATED SUCCESSFULLY")
    print("=" * 70)
    
    return trajectory


def main():
    """主函数"""
    
    # 配置路径
    input_pkl = "/home/user/loco-mujoco/examples/training_examples/football_training/datasets-pkl/football5.pkl"
    output_npz = "/home/user/loco-mujoco/examples/training_examples/football_training/datasets-npz/11-18-5.npz"
    env_name = "UnitreeG1"
    
    # 29自由度XML路径
    xml_path = "/home/user/loco-mujoco/loco_mujoco/models/unitree_g1/g1_29dof.xml"
    
    print("\n" + "=" * 70)
    print("PKL TO NPZ CONVERTER FOR 29-DOF G1")
    print("=" * 70)
    print(f"Input:  {input_pkl}")
    print(f"Output: {output_npz}")
    print(f"Env:    {env_name}")
    print(f"XML:    {xml_path}")
    print("=" * 70 + "\n")
    
    try:
        # 1. 加载并转换
        trajectory = load_and_convert_pkl(input_pkl, env_name, xml_path=xml_path)
        
        # 2. 保存为 npz
        print(f"\nSaving to: {output_npz}")
        Path(output_npz).parent.mkdir(parents=True, exist_ok=True)
        trajectory.save(output_npz)
        print(f"✓ Saved successfully!")
        
        # 3. 验证
        print("\n" + "=" * 70)
        print("VERIFYING SAVED FILE")
        print("=" * 70)
        
        loaded_traj = Trajectory.load(output_npz)
        print(f"✓ File loaded successfully!")
        print(f"  - Number of timesteps: {loaded_traj.data.qpos.shape[0]}")
        print(f"  - qpos dimension: {loaded_traj.data.qpos.shape[1]}")
        print(f"  - qvel dimension: {loaded_traj.data.qvel.shape[1]}")
        print(f"  - Frequency: {loaded_traj.info.frequency} Hz")
        print(f"  - Duration: {loaded_traj.data.qpos.shape[0] / loaded_traj.info.frequency:.2f} seconds")
        
        print("\n" + "=" * 70)
        print("CONVERSION COMPLETE! 🎉")
        print("=" * 70)
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("ERROR OCCURRED")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
