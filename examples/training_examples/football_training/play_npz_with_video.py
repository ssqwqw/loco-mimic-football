#!/usr/bin/env python3
"""
播放npz轨迹文件并生成演示视频

使用方法:
    python play_npz_with_video.py
"""

from pathlib import Path
import loco_mujoco
from loco_mujoco.task_factories import ImitationFactory, CustomDatasetConf
from loco_mujoco.trajectory import Trajectory


def play_trajectory_with_video(npz_path: str, 
                               output_dir: str = None,
                               video_name: str = None,
                               n_episodes: int = 1,
                               n_steps_per_episode: int = None,
                               render: bool = True,
                               record: bool = True,
                               compress: bool = True,
                               xml_path: str = None):
    """
    播放轨迹并生成视频
    
    Args:
        npz_path: npz轨迹文件路径
        output_dir: 视频输出目录（默认: ./LocoMuJoCo_recordings）
        video_name: 视频文件名（不含扩展名，默认: 使用npz文件名）
        n_episodes: 播放的episode数量
        n_steps_per_episode: 每个episode的步数（None表示播放完整轨迹）
        render: 是否渲染显示
        record: 是否录制视频
        compress: 是否压缩视频
        xml_path: XML文件路径（None表示使用默认的29自由度XML）
        
    注意:
        fps 会自动使用环境的 dt 计算，不需要手动指定
    """
    
    print("=" * 70)
    print("PLAYING TRAJECTORY AND GENERATING VIDEO")
    print("=" * 70)
    
    # 1. 加载轨迹
    print(f"\nLoading trajectory: {npz_path}")
    traj = Trajectory.load(npz_path)
    print(f"✓ Loaded successfully!")
    print(f"  - Number of frames: {traj.data.qpos.shape[0]}")
    print(f"  - qpos dimension: {traj.data.qpos.shape[1]}")
    print(f"  - Frequency: {traj.info.frequency} Hz")
    print(f"  - Duration: {traj.data.qpos.shape[0] / traj.info.frequency:.2f} seconds")
    
    # 2. 创建环境
    print(f"\nCreating environment...")
    # 如果没有指定XML路径，使用默认的29自由度XML
    if xml_path is None:
        xml_path = (loco_mujoco.PATH_TO_MODELS / "unitree_g1" / "g1_29dof.xml").as_posix()
    
    print(f"  Using XML: {xml_path}")
    env = ImitationFactory.make("UnitreeG1", 
                               spec=xml_path,
                               custom_dataset_conf=CustomDatasetConf(traj))
    print(f"✓ Environment created (29-DoF model)")
    
    # 3. 设置视频录制参数
    if record:
        # 如果没有指定视频名，使用npz文件名
        if video_name is None:
            video_name = Path(npz_path).stem
        
        # 如果没有指定输出目录，使用默认目录
        if output_dir is None:
            output_dir = "./LocoMuJoCo_recordings"
        
        # 注意：不要包含 'fps'，因为 play_trajectory 会自动使用环境的 dt 来计算 fps
        recorder_params = {
            "path": output_dir,
            "video_name": video_name,
            "compress": compress
        }
        
        print(f"\nVideo recording parameters:")
        print(f"  - Output directory: {output_dir}")
        print(f"  - Video name: {video_name}")
        print(f"  - FPS: 自动使用环境dt计算 (1/dt = {1/env.dt:.1f} Hz)")
        print(f"  - Compress: {compress}")
    else:
        recorder_params = None
    
    # 4. 如果没有指定步数，使用轨迹长度
    if n_steps_per_episode is None:
        n_steps_per_episode = traj.data.qpos.shape[0]
    
    print(f"\n" + "=" * 70)
    print("PLAYING TRAJECTORY")
    print("=" * 70)
    print(f"  - Episodes: {n_episodes}")
    print(f"  - Steps per episode: {n_steps_per_episode}")
    print(f"  - Render: {render}")
    print(f"  - Record: {record}")
    print("=" * 70 + "\n")
    
    # 5. 播放轨迹
    try:
        env.play_trajectory(
            n_episodes=n_episodes,
            n_steps_per_episode=n_steps_per_episode,
            render=render,
            record=record,
            recorder_params=recorder_params
        )
        
        if record:
            video_file = env.video_file_path
            print("\n" + "=" * 70)
            print("VIDEO GENERATED SUCCESSFULLY! 🎉")
            print("=" * 70)
            print(f"Video saved to: {video_file}")
            print("=" * 70)
            
            return video_file
        else:
            print("\n" + "=" * 70)
            print("TRAJECTORY PLAYBACK COMPLETE!")
            print("=" * 70)
            return None
            
    except KeyboardInterrupt:
        print("\n\nPlayback interrupted by user.")
        return None
    except Exception as e:
        print(f"\n\nError during playback: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数"""
    
    # ========== 配置参数 ==========
    # npz文件路径
    npz_path = "/home/user/loco-mujoco/examples/training_examples/football_training/datasets-npz/football_motion.npz"
    
    # XML文件路径（可选，None表示使用默认的29自由度XML）s
    xml_path = "/home/user/loco-mujoco/loco_mujoco/models/unitree_g1/g1_29dof.xml"
    # xml_path = None  # 也可以设置为None使用默认路径
    
    # 视频输出配置
    output_dir = "./LocoMuJoCo_recordings"  # 视频保存目录
    video_name = "g1_29dof_demo"  # 视频文件名（不含扩展名）
    
    # 播放配置
    n_episodes = 1  # 播放的episode数量
    n_steps_per_episode = None  # None表示播放完整轨迹，也可以指定具体步数
    render = True  # 是否显示渲染窗口
    record = True  # 是否录制视频
    
    # 视频参数
    # fps 参数已移除，因为 play_trajectory 会自动使用环境的 dt 来计算 fps
    compress = None  # 是否压缩视频（压缩后文件更小但需要时间）
    
    # ========== 执行播放和录制 ==========
    video_file = play_trajectory_with_video(
        npz_path=npz_path,
        output_dir=output_dir,
        video_name=video_name,
        n_episodes=n_episodes,
        n_steps_per_episode=n_steps_per_episode,
        render=render,
        record=record,
        compress=compress,
        xml_path=xml_path
    )
    
    if video_file:
        print(f"\n你可以使用以下命令查看视频:")
        print(f"  - Linux/Mac: open {video_file}")
        print(f"  - 或者直接双击文件打开")
    
    return 0


if __name__ == "__main__":
    exit(main())
