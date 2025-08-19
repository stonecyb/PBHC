#!/usr/bin/env bash
set -euo pipefail

# 1. 定义包含所有运动文件的目录
MOTION_DIR="motion_cmu_718/"

# 2. 遍历每个 .pkl 文件
for motion_file in "$MOTION_DIR"/*.pkl; do
  # 3. 提取文件名（带扩展名）
  filename=$(basename "$motion_file")
  # 4. 去掉扩展名，作为 experiment_name
  exp_name="${filename%.*}"

  # 5. 执行训练命令
  python humanoidverse/train_agent.py \
    +simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
    project_name=MotionTracking num_envs=4096 \
    +obs=motion_tracking/main \
    +robot=N2/N2 \
    +domain_rand=main_N2 \
    +rewards=motion_tracking/main \
    +robot.motion.motion_file="$motion_file" \
    experiment_name="$exp_name" \
    seed=1 \
    +device=cuda:0

  echo "Finished experiment: $exp_name"
done


# python humanoidverse/train_agent.py \
#   +simulator=isaacgym +exp=motion_tracking +terrain=terrain_locomotion_plane \
#   project_name=MotionTracking num_envs=4096 \
#   +obs=motion_tracking/main \
#   +robot=N2/N2 \
#   +domain_rand=main_N2 \
#   +rewards=motion_tracking/main \
#   +robot.motion.motion_file="motion_cmu_718/05_15_poses.pkl" \
#   experiment_name="05_15_poses" \
#   seed=1 \
#   +device=cuda:0
