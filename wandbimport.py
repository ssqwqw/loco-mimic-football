import wandb
import pandas as pd
import matplotlib.pyplot as plt


# ==========【1. 配置信息：只需要改这里】==========
# 你的 W&B 用户名或 team 名（在网页 URL 上可以看到）
ENTITY = "s081100369-dudu"         # 例："alunaqi"

# 你的 W&B 项目名（也在 URL 上，比如 https://wandb.ai/ENTITY/PROJECT）
PROJECT = "11-25-footballtrain"       # 例："my_project"

# 你要导出的 run id（不是 run 的 name）
# 不知道的话，可以先把 RUN_ID 留空，脚本会帮你列出来
RUN_ID = "r2cvcxv0"                         # 例："abc123def456"

# 你想画的那个 metric 的名字
# 要和 W&B 网页上 y 轴显示的列名完全一致
# 先可以暂时填 "loss"，或者后面打印列名再决定
METRIC_NAME = "Validation Measures/discrete_frechet_distance/qpos"
# ==================================================


def list_runs(entity: str, project: str):
    """列出该 project 下所有 run 的 id 和 name，方便你拷贝 RUN_ID。"""
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    print(f"\n📋 项目 {entity}/{project} 下的所有 runs：")
    for r in runs:
        print(f"  run_id: {r.id:<15} | name: {r.name}")
    print("\n👉 请把你想使用的 run_id 填到脚本顶部的 RUN_ID 变量里，然后重新运行脚本。")


def main():
    # 如果 RUN_ID 还没填，就先列出所有 runs，帮你选
    if RUN_ID == "" or RUN_ID is None:
        print("⚠️ 你还没有设置 RUN_ID，先帮你列出这个项目下的所有 runs：")
        list_runs(ENTITY, PROJECT)
        return

    # ==== 2. 通过 API 连接到指定 run ====
    print(f"\n🔗 正在连接到 run: {ENTITY}/{PROJECT}/{RUN_ID}")
    api = wandb.Api()
    run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")

    # ==== 3. 下载 history 数据 ====
    print("⬇️ 正在下载 run.history()（所有 step 的 metric 数据）...")
    history = run.history(samples=None)  # samples=None 表示全部
    print("✅ 下载完成！")

    # 打印一下有哪些列，帮助你确认 METRIC_NAME 写得对不对
    print("\n📌 history 中的列名（可用的 metric）：")
    print(list(history.columns))

    if METRIC_NAME not in history.columns:
        print(f"\n❌ 找不到名为 '{METRIC_NAME}' 的列，请检查 METRIC_NAME 是否写对。")
        print("   参考上面打印出来的列名之一，例如：'loss', 'accuracy'，或者你在 W&B 中看到的完整路径名。")
        return

    # ==== 4. 导出成 CSV 文件 ====
    csv_name = f"wandb_history_{RUN_ID}.csv"
    history.to_csv(csv_name, index=False, encoding="utf-8")
    print(f"\n💾 已将 history 导出为 CSV 文件：{csv_name}")

    # ==== 5. 准备画图的数据 ====
    # x 轴：通常用 _step（训练 step），你也可以改为 _runtime（运行时间秒）
    if "_step" in history.columns:
        x = history["_step"]
        x_label = "Step"
    elif "_runtime" in history.columns:
        x = history["_runtime"]
        x_label = "Runtime (s)"
    else:
        # 如果真的都没有，就用行号代替
        x = history.index
        x_label = "Index"

    y = history[METRIC_NAME]

    # ==== 6. 用 Matplotlib 画图 ====
    plt.figure(figsize=(10, 5))

    # 原始曲线：实线
    plt.plot(x, y, linestyle="-", linewidth=1.5, label=f"{METRIC_NAME} (raw)")

    # 平滑曲线：移动平均，虚线
    y_smooth = y.rolling(window=10, min_periods=1).mean()
    plt.plot(x, y_smooth, linestyle="--", linewidth=2.0, label=f"{METRIC_NAME} (moving avg)")

    # 也可以试试点线：
    # plt.plot(x, y_smooth, linestyle=":", linewidth=2.0, label=f"{METRIC_NAME} (dotted)")

    plt.xlabel(x_label)
    plt.ylabel(METRIC_NAME)
    plt.title(f"{METRIC_NAME} vs {x_label}\n(from W&B run {RUN_ID})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    print("\n📈 弹出的窗口中会显示图像。关闭图像窗口后脚本结束。")
    plt.show()


if __name__ == "__main__":
    main()
