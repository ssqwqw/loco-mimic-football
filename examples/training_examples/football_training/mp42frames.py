import cv2
import os

def extract_frames(video_path, output_folder):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 读取视频
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("无法打开视频文件:", video_path)
        return

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 视频读取完毕

        # 保存帧图片
        frame_path = os.path.join(output_folder, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(frame_path, frame)

        frame_count += 1

    cap.release()
    print(f"完成！共提取 {frame_count} 帧图片到：{output_folder}")


if __name__ == "__main__":
    video_path = "/home/user/loco-mujoco/examples/training_examples/football_training/1.mp4"        # ← 这里换成你的mp4视频路径
    output_folder = "/home/user/loco-mujoco/examples/training_examples/football_training/frames2" # ← 输出帧图片文件夹
    extract_frames(video_path, output_folder)
