# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
鱼眼深度预测精度评测脚本。

评测流程：
  1. 加载训练好的鱼眼模型，在 TartanGround 鱼眼验证集上推理
  2. 仅使用模型预测的深度（depth_along_ray，已乘以 metric_scaling_factor）
  3. 结合数据集中的 GT 内参（MEI 光线方向）和 GT 外参（c2w 位姿）
  4. 将每个视图的预测点云和 GT 点云统一变换到参考帧（view 0）坐标系
  5. 按 GT z-depth（沿光轴深度）每 5 米分 bin，计算各 bin 的深度精度指标
"""

import json       # 用于保存/加载 JSON 格式的评测结果
import logging    # Python 标准日志模块
import os         # 文件路径操作
import sys        # 用于重定向 stdout/stderr
import warnings   # 用于发出运行时警告
from collections import defaultdict  # 带默认值的字典，用于累积结果
from pathlib import Path             # 面向对象的文件路径操作

import hydra                          # Facebook 的配置管理框架，自动解析 YAML 配置
import numpy as np                    # 数值计算库
import torch                          # PyTorch 深度学习框架
import torch.backends.cudnn as cudnn  # cuDNN 后端配置（用于控制 benchmark 模式）
from omegaconf import DictConfig, OmegaConf  # Hydra 使用的配置对象类型

from mapanything.datasets import get_test_data_loader  # 构建测试数据加载器
from mapanything.models import init_model              # 根据配置字符串初始化模型
from mapanything.utils.geometry import inv, geotrf     # inv: 矩阵求逆; geotrf: 对3D点施加刚体变换
from mapanything.utils.image import rgb as denorm_rgb  # 将归一化图像张量还原为 [0,1] RGB
from mapanything.utils.misc import StreamToLogger      # 将 print 输出重定向到日志的工具

log = logging.getLogger(__name__)  # 获取当前模块的 logger 实例


# ============================================================================
#  场景分组：室内 / 室外
# ============================================================================

# TartanGround 验证集场景的室内/室外分类
# 室内 (indoor): 有明确建筑内部结构的封闭场景
# 室外 (outdoor): 开放空间、街道、自然环境等场景
INDOOR_SCENES = {
    "Restaurant",     # 餐厅内部
    "Hospital",       # 医院内部
    "Supermarket",    # 超市内部
}

OUTDOOR_SCENES = {
    "VictorianStreet",       # 维多利亚风格街道
    "SeasideTown",           # 海滨小镇
    "SeasonalForestSpring",  # 春季森林
    "ConstructionSite",      # 建筑工地
    "BrushifyMoon",          # 月球表面（室外开放环境）
    "WaterMillNight",        # 夜间水磨坊（室外）
    "AncientTowns",          # 古镇街道
}


def _get_scene_group(scene_name):
    """
    根据场景名称返回其所属分组（indoor / outdoor / unknown）。

    Args:
        scene_name: str，场景名称

    Returns:
        str: "indoor", "outdoor", 或 "unknown"
    """
    if scene_name in INDOOR_SCENES:
        return "indoor"
    elif scene_name in OUTDOOR_SCENES:
        return "outdoor"
    else:
        return "unknown"


# ============================================================================
#  PLY 可视化工具
# ============================================================================

def _save_ply(path, points, colors):
    """
    将点云保存为二进制 PLY 文件。

    Args:
        path:   str，输出文件路径
        points: (N, 3) float32 数组，3D 点坐标
        colors: (N, 3) uint8 数组，每个点的 RGB 颜色
    """
    n = len(points)
    # PLY 文件头：声明顶点数量和属性（xyz + rgb）
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    # 使用结构化 numpy 数组以确保二进制布局正确
    dtype = np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
    ])
    verts = np.empty(n, dtype=dtype)
    verts['x'] = points[:, 0]
    verts['y'] = points[:, 1]
    verts['z'] = points[:, 2]
    verts['r'] = colors[:, 0]
    verts['g'] = colors[:, 1]
    verts['b'] = colors[:, 2]
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(verts.tobytes())


def _error_to_color(errors, vmin=0.0, vmax=2.0):
    """
    将逐点误差映射为 RGB 颜色（蓝→绿→红 渐变）。

    误差 ≤ vmin 显示为蓝色（精度好），误差 ≥ vmax 显示为红色（精度差），
    中间值按线性插值过渡。

    Args:
        errors: (N,) float 数组，逐点误差值
        vmin:   float，颜色映射下界（映射为蓝色）
        vmax:   float，颜色映射上界（映射为红色）

    Returns:
        (N, 3) uint8 数组，RGB 颜色
    """
    # 归一化到 [0, 1]
    t = np.clip((errors - vmin) / max(vmax - vmin, 1e-8), 0.0, 1.0)  # (N,)
    # 蓝(0,0,255) → 绿(0,255,0) → 红(255,0,0)
    r = np.clip(2.0 * t - 1.0, 0.0, 1.0)         # t ∈ [0.5, 1] 时从 0 升到 1
    g = np.clip(1.0 - np.abs(2.0 * t - 1.0), 0.0, 1.0)  # t=0.5 时最大
    b_ch = np.clip(1.0 - 2.0 * t, 0.0, 1.0)       # t ∈ [0, 0.5] 时从 1 降到 0
    colors = np.stack([r, g, b_ch], axis=-1)        # (N, 3) float [0,1]
    return (colors * 255).astype(np.uint8)


def save_visualization_plys(
    pred_pts_ref, gt_pts_ref, gt_z_local, colors,
    scene_name, sample_idx, output_dir,
    delta_threshold=1.25,
):
    """
    将 GT 和预测点云保存为 RGB 彩色 PLY 文件，用于可视化对比。

    生成 4 个 PLY 文件：
      - *_gt_rgb.ply:         所有有效 GT 点，使用原始图像 RGB 颜色
      - *_pred_rgb.ply:       所有预测点，使用原始图像 RGB 颜色
      - *_pred_inlier_rgb.ply: 满足 delta<阈值 的预测点（inlier），使用原始图像 RGB 颜色
      - *_pred_error.ply:     所有预测点按 3D 误差着色（蓝→绿→红），展示误差空间分布

    Args:
        pred_pts_ref:    (N, 3) 参考帧下的预测 3D 点
        gt_pts_ref:      (N, 3) 参考帧下的 GT 3D 点
        gt_z_local:      (N,)   各视图自身相机坐标系的 GT z-depth
        colors:          (N, 3) uint8，每个像素的原始图像 RGB 颜色
        scene_name:      str，场景名称（用于文件命名）
        sample_idx:      int，样本序号（用于文件命名）
        output_dir:      str，输出目录路径
        delta_threshold: float，delta 阈值，默认 1.25
    """
    vis_dir = os.path.join(output_dir, "vis_ply")
    os.makedirs(vis_dir, exist_ok=True)

    prefix = f"{scene_name}_s{sample_idx:04d}"

    # 计算参考帧 z-depth
    pred_z = pred_pts_ref[:, 2]  # (N,)
    gt_z = gt_pts_ref[:, 2]      # (N,)

    # 有效掩码：GT 和预测 z-depth 均为正值
    valid = (gt_z > 0) & (pred_z > 0)

    # ---- 1. 保存 GT RGB 彩色点云 ----
    gt_path = os.path.join(vis_dir, f"{prefix}_gt_rgb.ply")
    _save_ply(gt_path, gt_pts_ref.astype(np.float32), colors)
    print(f"  Saved GT RGB point cloud ({len(gt_pts_ref)} pts) -> {gt_path}")

    # ---- 2. 保存预测 RGB 彩色点云（全部有效点） ----
    pred_path = os.path.join(vis_dir, f"{prefix}_pred_rgb.ply")
    _save_ply(pred_path, pred_pts_ref.astype(np.float32), colors)
    print(f"  Saved pred RGB point cloud ({len(pred_pts_ref)} pts) -> {pred_path}")

    if valid.sum() == 0:
        print(f"  No valid points for inlier/error visualization in {prefix}")
        return

    # 计算逐点 delta 比值
    ratio = np.maximum(pred_z[valid] / gt_z[valid], gt_z[valid] / pred_z[valid])  # (N_valid,)
    # 满足 delta < 阈值的点为 inlier
    inlier_mask_in_valid = ratio < delta_threshold
    # 将 inlier 掩码映射回全局索引
    valid_indices = np.where(valid)[0]
    inlier_indices = valid_indices[inlier_mask_in_valid]

    # ---- 3. 保存 inlier 预测点云（原始 RGB 颜色，仅保留精度好的点） ----
    inlier_pts = pred_pts_ref[inlier_indices]
    inlier_colors = colors[inlier_indices]
    inlier_path = os.path.join(vis_dir, f"{prefix}_pred_inlier_rgb.ply")
    _save_ply(inlier_path, inlier_pts.astype(np.float32), inlier_colors)
    inlier_pct = len(inlier_pts) / valid.sum() * 100.0
    print(f"  Saved inlier pred RGB cloud ({len(inlier_pts)} pts, {inlier_pct:.1f}%) -> {inlier_path}")

    # ---- 4. 保存按误差着色的预测点云（蓝→绿→红） ----
    # 逐点 3D 欧氏误差
    errors_3d = np.linalg.norm(pred_pts_ref - gt_pts_ref, axis=-1)  # (N,)
    error_colors = _error_to_color(errors_3d, vmin=0.0, vmax=2.0)
    error_path = os.path.join(vis_dir, f"{prefix}_pred_error.ply")
    _save_ply(error_path, pred_pts_ref.astype(np.float32), error_colors)
    print(f"  Saved error-colored pred cloud ({len(pred_pts_ref)} pts) -> {error_path}")


def compute_depth_metrics(pred_z, gt_z):
    """
    计算标准单目深度评估指标。

    给定预测和真值的 z-depth 数组（均为正值），计算三个常用指标：
      - Abs Rel: 平均相对绝对误差 = mean(|pred - gt| / gt)
      - RMSE: 均方根误差 = sqrt(mean((pred - gt)^2))
      - delta < 1.25: 阈值准确率 = 满足 max(pred/gt, gt/pred) < 1.25 的像素比例 (%)

    ══════════════════════════════════════════════════════════════════════
    各指标的物理含义：
    ══════════════════════════════════════════════════════════════════════

    1. Abs Rel (Absolute Relative Error，绝对相对误差)
       公式: mean( |pred_z - gt_z| / gt_z )
       物理含义: 衡量预测深度相对于真实深度的偏差比例。
         - 值为 0.10 表示平均每个像素的深度预测误差是真实深度的 10%。
         - 例如真实深度 10m 处，Abs Rel=0.10 意味着平均预测误差约 1m。
         - 该指标与深度值的绝对大小无关（归一化后的误差），适合跨距离对比。
         - 越小越好，理想值为 0。

    2. RMSE (Root Mean Square Error，均方根误差)
       公式: sqrt( mean( (pred_z - gt_z)^2 ) )
       物理含义: 衡量预测深度与真实深度之间的绝对偏差，单位为米。
         - 值为 0.50 表示预测的 z-depth 与真值之间的均方根偏差约 0.5 米。
         - 对大误差更敏感（因为取了平方），少量严重偏差的点会显著拉高 RMSE。
         - 可以直观理解为"典型的深度预测误差有多少米"。
         - 越小越好，理想值为 0。

    3. delta < 1.25 (Threshold Accuracy，阈值准确率)
       公式: 100% × mean( max(pred_z/gt_z, gt_z/pred_z) < 1.25 )
       物理含义: 满足"预测深度与真实深度之比在 [1/1.25, 1.25] 范围内"的像素比例。
         - 即预测值不超过真值的 ±25% 偏差的点的百分比。
         - 值为 95.0 表示 95% 的像素深度预测误差在 ±25% 以内。
         - 这是一个鲁棒的准确率度量，不受极端异常值影响。
         - 越大越好，理想值为 100%。

    ══════════════════════════════════════════════════════════════════════

    Args:
        pred_z: (N,) numpy 数组，预测的 z-depth 值（必须 > 0）
        gt_z:   (N,) numpy 数组，真值的 z-depth 值（必须 > 0）

    Returns:
        dict: 包含 abs_rel, rmse, delta_125 三个指标的字典
    """
    if len(gt_z) == 0:  # 如果没有有效像素，返回 NaN
        return {"abs_rel": float("nan"), "rmse": float("nan"), "delta_125": float("nan")}

    # Abs Rel: 逐像素计算 |pred - gt| / gt，然后取均值
    # 物理意义：预测深度相对于真实深度的平均偏差比例（无量纲）
    abs_rel = np.mean(np.abs(pred_z - gt_z) / gt_z)

    # RMSE: 逐像素计算 (pred - gt)^2，取均值后开方
    # 物理意义：预测深度与真实深度之间的均方根偏差（单位：米）
    rmse = np.sqrt(np.mean((pred_z - gt_z) ** 2))

    # delta < 1.25: 计算 max(pred/gt, gt/pred)，统计小于 1.25 的比例
    # 物理意义：深度预测误差在 ±25% 以内的像素占比（百分比）
    ratio = np.maximum(pred_z / gt_z, gt_z / pred_z)  # (N,) 取逐元素最大值
    delta_125 = np.mean(ratio < 1.25) * 100.0          # 转换为百分比

    return {"abs_rel": float(abs_rel), "rmse": float(rmse), "delta_125": float(delta_125)}


def compute_pointcloud_metrics(pred_pts, gt_pts):
    """
    计算 3D 点云误差指标（作为深度指标的辅助参考）。

    对参考帧下的预测点和 GT 点，逐点计算欧氏距离误差，
    然后报告均值和中位数。

    ══════════════════════════════════════════════════════════════════════
    各指标的物理含义：
    ══════════════════════════════════════════════════════════════════════

    4. Mean 3D Error (平均 3D 欧氏误差)
       公式: mean( ||pred_pt - gt_pt||_2 )
       物理含义: 预测 3D 点与真实 3D 点之间欧氏距离的平均值，单位为米。
         - 与 RMSE 不同，这里衡量的是完整三维空间中的定位误差，
           不仅包含沿光轴（z）方向的偏差，也包含横向（x, y）偏差。
         - 值为 0.30 表示平均每个点在 3D 空间中偏离真值 0.3 米。
         - 如果模型深度预测准确但光线方向有偏差，3D 误差可能大于 z-depth 的 RMSE。
         - 越小越好，理想值为 0。

    5. Median 3D Error (中位 3D 欧氏误差)
       公式: median( ||pred_pt - gt_pt||_2 )
       物理含义: 与 Mean 3D Error 类似，但取中位数而非均值。
         - 中位数对极端异常值更鲁棒：即使少量点误差极大，中位数也不会被拉高。
         - 可以理解为"一半以上的点的 3D 定位误差不超过此值"。
         - 当 mean_3d_error 远大于 median_3d_error 时，说明存在少量大误差的异常点。
         - 越小越好，理想值为 0。

    ══════════════════════════════════════════════════════════════════════

    Args:
        pred_pts: (N, 3) numpy 数组，参考帧下的预测 3D 点
        gt_pts:   (N, 3) numpy 数组，参考帧下的 GT 3D 点

    Returns:
        dict: 包含 mean_3d_error（均值）和 median_3d_error（中位数）的字典
    """
    if len(gt_pts) == 0:  # 无有效点时返回 NaN
        return {"mean_3d_error": float("nan"), "median_3d_error": float("nan")}

    # 逐点计算欧氏距离: ||pred_pt - gt_pt||_2，形状 (N,)
    # 物理意义：每个预测 3D 点在参考帧中偏离真值的空间距离（单位：米）
    errors = np.linalg.norm(pred_pts - gt_pts, axis=-1)
    return {
        "mean_3d_error": float(np.mean(errors)),    # 所有点的平均 3D 定位误差（米）
        "median_3d_error": float(np.median(errors)), # 所有点的中位 3D 定位误差（米）
    }


def reconstruct_and_evaluate(batch, preds, distance_bin_size=5, max_distance=50):
    """
    核心评测函数：用预测深度 + GT 内参/外参重建参考帧点云，按距离分 bin 评估精度。

    处理流程（对 batch 中每个样本的每个视图）：
      1. 取 GT ray_directions_cam（由 MEI 内参计算的单位光线方向）
      2. 取模型预测的 depth_along_ray（已乘以 metric_scaling_factor 的径向深度）
      3. 在相机坐标系重建点云: pts_cam = ray_dirs * depth
      4. 用 GT 位姿将点云变换到参考帧（view 0 的相机坐标系）
      5. 按 GT z-depth（各视图自身相机坐标系下的光轴深度）分 bin
      6. 在每个 bin 内计算深度指标和 3D 点云指标

    Args:
        batch:  list[dict]，长度 = n_views，每个 dict 是一个视图的数据
                包含 ray_directions_cam, depth_along_ray, valid_mask, camera_pose 等
        preds:  list[dict]，长度 = n_views，每个 dict 是模型对该视图的预测
                包含 depth_along_ray（已乘以 scale）等
        distance_bin_size: int，距离 bin 宽度（米），默认 5
        max_distance: int，最大距离（米），超过此值归入最后一个 bin，默认 50

    Returns:
        tuple: (results, pointcloud_data)
          - results: list[dict]，长度 = batch_size，每个元素是 {bin_label: {指标: 值}}
          - pointcloud_data: list[dict]，长度 = batch_size，每个元素包含:
              - "pred_pts_ref": (N, 3) 参考帧下的预测点
              - "gt_pts_ref":   (N, 3) 参考帧下的 GT 点
              - "gt_z_local":   (N,) 各视图相机坐标系的 GT z-depth（用于分 bin）
              - "colors":       (N, 3) uint8 RGB 颜色（从原始图像反归一化得到）
    """
    n_views = len(batch)                             # 多视图数量（例如 4）
    batch_size = batch[0]["camera_pose"].shape[0]    # 当前 batch 的样本数

    # 计算参考帧（view 0）的 world-to-camera 变换矩阵
    # batch[0]["camera_pose"] 是 view 0 的 c2w (camera-to-world)，求逆得到 w2c
    w2c_ref = inv(batch[0]["camera_pose"])  # (B, 4, 4)

    # 构建距离 bin 的边界列表: [0, 5, 10, 15, ..., max_distance]
    bin_edges = list(range(0, max_distance + 1, distance_bin_size))

    results = []          # 存储 batch 内每个样本的评测结果
    pointcloud_data = []  # 存储 batch 内每个样本的点云数据（用于可视化）

    for b in range(batch_size):  # 遍历 batch 中的每个样本
        # 用于跨视图累积所有有效像素的数据
        all_gt_z_local = []    # GT z-depth（各视图自身相机坐标系），用于距离分 bin
        all_pred_pts_ref = []  # 预测点在参考帧下的 3D 坐标
        all_gt_pts_ref = []    # GT 点在参考帧下的 3D 坐标
        all_colors = []        # 各像素的 RGB 颜色（从原始图像提取）

        for v in range(n_views):  # 遍历每个视图
            # ---- 从数据集获取 GT 量 ----
            gt_ray_dirs_cam = batch[v]["ray_directions_cam"][b]  # (H, W, 3) 由 MEI 内参计算的单位光线方向
            gt_depth = batch[v]["depth_along_ray"][b]            # (H, W, 1) GT 径向深度（ray-depth）
            valid_mask = batch[v]["valid_mask"][b]                # (H, W)   有效像素掩码（排除天空、无效深度等）
            c2w_v = batch[v]["camera_pose"][b]                   # (4, 4)   该视图的 camera-to-world 位姿

            # ---- 从原始图像提取 RGB 颜色（反归一化） ----
            # batch[v]["img"] 形状为 (B, 3, H, W)，经过了 DINOv2 归一化
            # denorm_rgb 将其还原为 [0, 1] 范围的 float RGB，形状 (H, W, 3)
            norm_type = batch[v]["data_norm_type"][b]             # 该视图使用的归一化类型
            img_rgb = denorm_rgb(batch[v]["img"][b], norm_type)   # (H, W, 3) float [0,1]
            img_rgb_u8 = (img_rgb * 255).clip(0, 255).astype(np.uint8)  # 转为 uint8

            # ---- 从模型预测获取深度和非模糊掩码 ----
            # depth_along_ray 在模型 forward 中已经乘以了 metric_scaling_factor
            pred_depth = preds[v]["depth_along_ray"][b]          # (H, W, 1) 预测的径向深度

            # 数据集中的 GT non_ambiguous_mask 参与过滤（如果存在），排除天空等模糊区域
            if "non_ambiguous_mask" in batch[v]:
                gt_non_ambig = batch[v]["non_ambiguous_mask"][b]   # (H, W)
                valid_mask = valid_mask & (gt_non_ambig > 0.5)

            # ---- 在相机坐标系下重建 3D 点 ----
            # 点 = 单位光线方向 × 径向深度（MEI 鱼眼模型的标准参数化）
            gt_pts_cam = gt_ray_dirs_cam * gt_depth              # (H, W, 3) GT 相机坐标系点云
            pred_pts_cam = gt_ray_dirs_cam * pred_depth          # (H, W, 3) 预测相机坐标系点云

            # GT z-depth: 相机坐标系下沿光轴方向的深度（z 分量），用于距离分 bin
            gt_z_local = gt_pts_cam[..., 2]                      # (H, W)

            # ---- 变换到参考帧坐标系 ----
            # ref_from_cam = w2c_ref @ c2w_v：从视图 v 的相机坐标系到 view 0 相机坐标系的变换
            ref_from_cam = w2c_ref[b] @ c2w_v                   # (4, 4)

            # geotrf 对 3D 点施加 4×4 刚体变换（R @ pts + t），
            # 需要 unsqueeze(0) 添加 batch 维度以匹配 geotrf 的输入要求
            gt_pts_ref = geotrf(ref_from_cam.unsqueeze(0), gt_pts_cam.unsqueeze(0))[0]      # (H, W, 3)
            pred_pts_ref = geotrf(ref_from_cam.unsqueeze(0), pred_pts_cam.unsqueeze(0))[0]  # (H, W, 3)

            # ---- 提取有效像素并转为 numpy ----
            mask_np = valid_mask.cpu().numpy().astype(bool)        # (H, W) 布尔掩码
            gt_z_flat = gt_z_local.cpu().numpy()[mask_np]          # (N_valid,) 有效像素的 GT z-depth
            pred_ref_flat = pred_pts_ref.cpu().numpy()[mask_np]    # (N_valid, 3) 有效像素的预测参考帧点
            gt_ref_flat = gt_pts_ref.cpu().numpy()[mask_np]        # (N_valid, 3) 有效像素的 GT 参考帧点
            colors_flat = img_rgb_u8[mask_np]                      # (N_valid, 3) 有效像素的 RGB 颜色

            # 过滤掉 z-depth <= 0 的点（无效/背面的点）
            pos_mask = gt_z_flat > 0
            all_gt_z_local.append(gt_z_flat[pos_mask])
            all_pred_pts_ref.append(pred_ref_flat[pos_mask])
            all_gt_pts_ref.append(gt_ref_flat[pos_mask])
            all_colors.append(colors_flat[pos_mask])

        # 将所有视图的有效像素拼接在一起
        all_gt_z_local = np.concatenate(all_gt_z_local, axis=0)    # (N_total,) 所有有效点的 GT z-depth
        all_pred_pts_ref = np.concatenate(all_pred_pts_ref, axis=0) # (N_total, 3) 所有预测点（参考帧）
        all_gt_pts_ref = np.concatenate(all_gt_pts_ref, axis=0)     # (N_total, 3) 所有 GT 点（参考帧）
        all_colors = np.concatenate(all_colors, axis=0)             # (N_total, 3) 所有有效点的 RGB 颜色

        # 提取参考帧下的 z-depth 分量，用于深度指标计算
        pred_z_ref = all_pred_pts_ref[:, 2]  # (N_total,) 预测的参考帧 z-depth
        gt_z_ref = all_gt_pts_ref[:, 2]      # (N_total,) GT 的参考帧 z-depth

        # ---- 按距离分 bin 计算指标 ----
        per_bin_results = {}
        for i in range(len(bin_edges)):
            # 确定当前 bin 的上下界和标签
            if i < len(bin_edges) - 1:
                lo, hi = bin_edges[i], bin_edges[i + 1]  # 例如 lo=0, hi=5
                bin_label = f"{lo}-{hi}m"                 # 例如 "0-5m"
            else:
                lo = bin_edges[i]                          # 最后一个 bin: [max_distance, +∞)
                bin_label = f"{lo}m+"                      # 例如 "50m+"

            # 构建当前 bin 的布尔掩码（基于 GT z-depth 分 bin）
            bin_mask = (all_gt_z_local >= lo)
            if i < len(bin_edges) - 1:
                bin_mask = bin_mask & (all_gt_z_local < hi)  # 左闭右开区间 [lo, hi)

            n_pts = int(bin_mask.sum())  # 当前 bin 内的有效点数
            if n_pts == 0:  # 如果当前 bin 没有点，填充 NaN
                per_bin_results[bin_label] = {
                    "n_points": 0,
                    "abs_rel": float("nan"),
                    "rmse": float("nan"),
                    "delta_125": float("nan"),
                    "mean_3d_error": float("nan"),
                    "median_3d_error": float("nan"),
                }
                continue

            # 取出当前 bin 内的参考帧 z-depth
            bin_pred_z = pred_z_ref[bin_mask]  # 预测 z-depth
            bin_gt_z = gt_z_ref[bin_mask]      # GT z-depth
            # 确保预测和 GT 的 z-depth 均为正值（用于比值指标）
            valid_z = (bin_gt_z > 0) & (bin_pred_z > 0)
            # 计算深度指标: Abs Rel, RMSE, delta < 1.25
            depth_metrics = compute_depth_metrics(
                bin_pred_z[valid_z], bin_gt_z[valid_z]
            )

            # 计算 3D 点云指标: 平均/中位数欧氏距离误差
            pc_metrics = compute_pointcloud_metrics(
                all_pred_pts_ref[bin_mask], all_gt_pts_ref[bin_mask]
            )

            # 合并当前 bin 的所有指标
            per_bin_results[bin_label] = {
                "n_points": n_pts,   # 该 bin 内的点数
                **depth_metrics,     # abs_rel, rmse, delta_125
                **pc_metrics,        # mean_3d_error, median_3d_error
            }

        # ---- 计算全局（所有 bin 合并）指标 ----
        valid_all = (gt_z_ref > 0) & (pred_z_ref > 0)  # 全局正值掩码
        overall_depth = compute_depth_metrics(pred_z_ref[valid_all], gt_z_ref[valid_all])
        overall_pc = compute_pointcloud_metrics(all_pred_pts_ref, all_gt_pts_ref)
        per_bin_results["overall"] = {
            "n_points": int(valid_all.sum()),
            **overall_depth,
            **overall_pc,
        }

        results.append(per_bin_results)  # 将当前样本的结果加入列表

        # 保存点云数据供后续可视化使用
        pointcloud_data.append({
            "pred_pts_ref": all_pred_pts_ref,  # (N, 3) 参考帧预测点
            "gt_pts_ref": all_gt_pts_ref,      # (N, 3) 参考帧 GT 点
            "gt_z_local": all_gt_z_local,      # (N,)   各视图相机坐标系 GT z-depth
            "colors": all_colors,              # (N, 3) uint8 原始图像 RGB 颜色
        })

    return results, pointcloud_data  # 同时返回评测结果和点云数据


def build_dataset(dataset, batch_size, num_workers):
    """
    构建测试数据加载器。

    将 Hydra 配置中的 dataset_str 字符串传给 get_test_data_loader，
    后者会通过 eval() 实例化对应的 Dataset 类，并包装为 DataLoader。

    Args:
        dataset: str，数据集定义字符串（例如 "TartanGroundWAI(split='val', ...)"）
        batch_size: int，每个 batch 的样本数
        num_workers: int，数据加载的工作线程数

    Returns:
        DataLoader: 配置好的 PyTorch 数据加载器
    """
    print("Building data loader for dataset: ", dataset)
    loader = get_test_data_loader(
        dataset,
        batch_size=batch_size,   # 每 batch 的多视图集合数
        num_workers=num_workers,  # 并行数据加载线程数
        pin_mem=True,             # 使用 pinned memory 加速 CPU→GPU 传输
        shuffle=False,            # 评测时不打乱顺序，保证可复现
        drop_last=False,          # 不丢弃最后一个不完整的 batch
    )
    print("Dataset length: ", len(loader))  # 打印 batch 总数
    return loader


@torch.no_grad()  # 评测时关闭梯度计算，节省显存和加速
def benchmark(args):
    """
    评测主函数：加载模型和数据，逐 batch 推理并计算分 bin 指标，保存结果。

    流程：
      1. 创建输出目录、设置随机种子、配置混合精度
      2. 构建测试数据集的 DataLoader
      3. 初始化模型并加载预训练权重
      4. 遍历每个 batch：推理 → 重建点云 → 计算指标
      5. 按场景保存详细结果、按数据集保存聚合结果
      6. 打印结果摘要表格

    Args:
        args: OmegaConf DictConfig，包含所有 Hydra 配置参数
    """
    # 打印并创建输出目录
    print("Output Directory: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)  # 递归创建目录

    # 打印脚本路径和完整配置（方便调试）
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    # 设置计算设备（优先使用 GPU）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # 固定随机种子，保证评测结果可复现
    seed = args.seed
    torch.manual_seed(seed)       # PyTorch 随机种子
    np.random.seed(seed)          # NumPy 随机种子
    # 关闭 cuDNN benchmark 模式（输入尺寸固定时可开启加速，变尺寸时应关闭）
    cudnn.benchmark = not args.disable_cudnn_benchmark

    # 配置混合精度推理的浮点类型
    if args.amp:  # 如果启用了自动混合精度
        if args.amp_dtype == "fp16":
            amp_dtype = torch.float16           # 半精度
        elif args.amp_dtype == "bf16":
            if torch.cuda.is_bf16_supported():  # 检查硬件是否支持 bf16
                amp_dtype = torch.bfloat16
            else:
                warnings.warn("bf16 not supported, falling back to fp16.")
                amp_dtype = torch.float16
        elif args.amp_dtype == "fp32":
            amp_dtype = torch.float32           # 全精度（实际不做混合精度）
    else:
        amp_dtype = torch.float32               # 不使用混合精度

    # 从配置中读取距离分 bin 参数
    distance_bin_size = args.get("distance_bin_size", 5)   # 每个 bin 的宽度（米），默认 5
    max_distance = args.get("max_distance", 50)             # 最大距离（米），默认 50

    # 可视化参数
    save_vis = args.get("save_vis", True)                   # 是否保存可视化 PLY 文件
    vis_delta_threshold = args.get("vis_delta_threshold", 1.25)  # inlier 判定阈值（delta < 此值为 inlier）
    vis_max_samples = args.get("vis_max_samples", 5)        # 每个场景最多保存多少个样本的 PLY
    vis_sample_counter = defaultdict(int)                    # 每个场景已保存的样本计数器

    # ---- 构建测试数据集 ----
    # test_dataset 可能包含多个数据集（用 "+" 连接），逐个构建 DataLoader
    print("Building test dataset {:s}".format(args.dataset.test_dataset))
    data_loaders = {
        # key: 数据集名称（取 "(" 前面的部分，如 "TartanGroundWAI"）
        # value: 对应的 DataLoader
        dataset.split("(")[0]: build_dataset(
            dataset, args.batch_size, args.dataset.num_workers
        )
        for dataset in args.dataset.test_dataset.split("+")  # 按 "+" 拆分多个数据集
        if "(" in dataset                                      # 过滤空字符串
    }

    # ---- 加载模型 ----
    # init_model 根据 model_str（如 "fisheyemapanything"）和 model_config 创建模型实例
    model = init_model(
        args.model.model_str, args.model.model_config, torch_hub_force_reload=False
    )
    model.to(device)  # 将模型参数移到 GPU

    # 加载预训练权重
    if args.model.pretrained:
        print("Loading pretrained: ", args.model.pretrained)
        ckpt = torch.load(
            args.model.pretrained, map_location=device, weights_only=False  # 加载 checkpoint
        )
        # 加载模型权重，strict=False 允许忽略不匹配的 key
        print(model.load_state_dict(ckpt["model"], strict=False))
        del ckpt  # 释放 checkpoint 占用的内存

    # ---- 配置推理时的几何输入 ----
    # 通过 _configure_geometric_input_config 确定性地开/关各几何模态，
    # 取代训练时的随机概率，确保评测结果可复现。
    use_calibration = args.get("use_calibration", False)  # 是否向模型提供 GT 内参（ray_directions_cam）
    use_pose = args.get("use_pose", False)                # 是否向模型提供 GT 位姿
    use_depth_scale = args.get("use_depth_scale", False)  # 是否向模型提供深度的度量尺度信息
    use_pose_scale = args.get("use_pose_scale", False)    # 是否向模型提供位姿的度量尺度信息

    if hasattr(model, "_configure_geometric_input_config"):
        model._configure_geometric_input_config(
            use_calibration=use_calibration,
            use_depth=False,        # 评测时不提供 GT 深度（否则就是用答案做预测）
            use_pose=use_pose,
            use_depth_scale=use_depth_scale,
            use_pose_scale=use_pose_scale,
        )
        print(f"Geometric input config: calibration={use_calibration}, "
              f"pose={use_pose}, depth_scale={use_depth_scale}, pose_scale={use_pose_scale}")
    else:
        print("Model does not support geometric input configuration, using defaults.")

    # 用于存储所有数据集的聚合评测结果
    per_dataset_results = {}

    # ---- 遍历每个评测数据集 ----
    for benchmark_dataset_name, data_loader in data_loaders.items():
        print("Benchmarking dataset: ", benchmark_dataset_name)
        data_loader.dataset.set_epoch(0)  # 设置 epoch（影响确定性采样的种子）

        # 按场景（scene）累积所有样本的评测结果
        # key: 场景名（如 "VictorianStreet"），value: list[per_sample_binned_results]
        per_scene_results = defaultdict(list)

        # ---- 遍历每个 batch ----
        for batch in data_loader:
            n_views = len(batch)  # 每个样本包含的视图数量
            # 移除 idx 中的前两个元素（dataset 和 aspect ratio 索引），只保留 view 索引
            for view in batch:
                view["idx"] = view["idx"][2:]

            # 将 batch 数据传输到 GPU（跳过字符串类型和不需要的 key）
            ignore_keys = {
                "depthmap",        # 原始 depthmap，不再使用（已用 depth_along_ray 代替）
                "dataset",         # 数据集名称字符串
                "label",           # 场景名称字符串
                "instance",        # 帧标识字符串
                "idx",             # 索引元组
                "true_shape",      # 原始图像尺寸
                "rng",             # 随机数状态
                "data_norm_type",  # 数据归一化类型字符串
            }
            for view in batch:
                for name in view.keys():
                    if name in ignore_keys:  # 跳过字符串等无法传到 GPU 的字段
                        continue
                    view[name] = view[name].to(device, non_blocking=True)  # 异步传到 GPU

            # ---- 模型推理 ----
            # 使用混合精度上下文进行前向推理
            with torch.autocast("cuda", enabled=bool(args.amp), dtype=amp_dtype):
                preds = model(batch)  # preds: list[dict]，每个视图一个预测字典

            # ---- 重建点云并计算分 bin 指标 ----
            sample_results, sample_pc_data = reconstruct_and_evaluate(
                batch, preds,
                distance_bin_size=distance_bin_size,
                max_distance=max_distance,
            )

            # 将每个样本的结果按场景名称归类，并可选保存 PLY 可视化
            batch_size = batch[0]["img"].shape[0]
            for b_idx in range(batch_size):
                scene = batch[0]["label"][b_idx]                 # 场景名称来自 view 0
                per_scene_results[scene].append(sample_results[b_idx])

                # ---- 保存可视化 PLY 文件 ----
                if save_vis and vis_sample_counter[scene] < vis_max_samples:
                    pc = sample_pc_data[b_idx]
                    save_visualization_plys(
                        pred_pts_ref=pc["pred_pts_ref"],
                        gt_pts_ref=pc["gt_pts_ref"],
                        gt_z_local=pc["gt_z_local"],
                        colors=pc["colors"],
                        scene_name=scene,
                        sample_idx=vis_sample_counter[scene],
                        output_dir=args.output_dir,
                        delta_threshold=vis_delta_threshold,
                    )
                    vis_sample_counter[scene] += 1

        # ---- 保存逐场景详细结果 ----
        per_scene_path = os.path.join(
            args.output_dir, f"{benchmark_dataset_name}_per_scene_results.json"
        )
        with open(per_scene_path, "w") as f:
            json.dump(dict(per_scene_results), f, indent=4)  # 转为普通 dict 再序列化
        print(f"Saved per-scene results to {per_scene_path}")

        # ---- 跨场景聚合：对每个 bin 的每个指标取所有样本的均值 ----
        aggregated = _aggregate_binned_results(per_scene_results)

        # 保存聚合结果
        agg_path = os.path.join(
            args.output_dir, f"{benchmark_dataset_name}_aggregated.json"
        )
        with open(agg_path, "w") as f:
            json.dump(aggregated, f, indent=4)

        # 打印全局聚合结果表格
        print(f"\n{'='*60}")
        print(f"[All] Aggregated results for {benchmark_dataset_name}")
        print(f"{'='*60}")
        _print_binned_results(aggregated)

        # ---- 按室内/室外分组聚合 ----
        grouped_results = _aggregate_by_scene_group(per_scene_results)

        # 保存分组结果
        grouped_path = os.path.join(
            args.output_dir, f"{benchmark_dataset_name}_grouped.json"
        )
        with open(grouped_path, "w") as f:
            json.dump(grouped_results, f, indent=4)

        # 打印各分组结果表格
        for group_name, group_agg in grouped_results.items():
            # 收集该分组包含的场景名
            group_scenes = [s for s in per_scene_results if _get_scene_group(s) == group_name]
            print(f"\n{'='*60}")
            print(f"[{group_name}] scenes: {', '.join(group_scenes)}")
            print(f"{'='*60}")
            _print_binned_results(group_agg)

        # 存入跨数据集结果字典（包含全局 + 分组）
        per_dataset_results[benchmark_dataset_name] = {
            "all": aggregated,
            **{f"group_{k}": v for k, v in grouped_results.items()},
        }

    # ---- 保存所有数据集的汇总结果 ----
    overall_path = os.path.join(args.output_dir, "per_dataset_results.json")
    with open(overall_path, "w") as f:
        json.dump(per_dataset_results, f, indent=4)
    print(f"\nAll results saved to {args.output_dir}")


def _aggregate_by_scene_group(per_scene_results):
    """
    按室内/室外分组聚合评测结果。

    将 per_scene_results 中的场景按照 INDOOR_SCENES / OUTDOOR_SCENES 分类，
    分别调用 _aggregate_binned_results 计算各组的聚合指标。

    Args:
        per_scene_results: defaultdict(list)，
            key = 场景名，value = list[dict]

    Returns:
        dict: {"indoor": {bin: metrics}, "outdoor": {bin: metrics}, ...}
              仅包含有数据的分组
    """
    # 按分组收集场景结果
    grouped = defaultdict(lambda: defaultdict(list))
    for scene_name, scene_samples in per_scene_results.items():
        group = _get_scene_group(scene_name)
        grouped[group][scene_name] = scene_samples

    # 对每个分组分别聚合
    result = {}
    for group_name in ["indoor", "outdoor", "unknown"]:
        if group_name not in grouped:
            continue
        group_agg = _aggregate_binned_results(grouped[group_name])
        result[group_name] = group_agg

    return result


def _aggregate_binned_results(per_scene_results):
    """
    将所有场景、所有样本的分 bin 结果聚合为单一的全局统计。

    遍历每个场景下的每个样本的每个 bin 的每个指标值，
    收集到一个二级字典中（bin_label -> metric_name -> list[values]），
    最后对 n_points 求和、对其他指标求均值。跳过 NaN 值。

    Args:
        per_scene_results: defaultdict(list)，
            key = 场景名，value = list[dict]，每个 dict 是一个样本的 {bin: {metric: val}}

    Returns:
        dict: {bin_label: {metric_name: aggregated_value, ...}, ...}，按 bin 排序
    """
    # 二级默认字典: bin_label -> metric_name -> list[float]
    bin_accumulators = defaultdict(lambda: defaultdict(list))

    # 遍历所有场景的所有样本
    for scene_samples in per_scene_results.values():
        for sample in scene_samples:
            for bin_label, metrics in sample.items():
                for metric_name, value in metrics.items():
                    if not np.isnan(value):  # 跳过 NaN（该 bin 无有效点的情况）
                        bin_accumulators[bin_label][metric_name].append(value)

    # 对收集到的值进行聚合
    aggregated = {}
    for bin_label in sorted(bin_accumulators.keys(), key=_bin_sort_key):  # 按距离排序
        agg = {}
        for metric_name, values in bin_accumulators[bin_label].items():
            if metric_name == "n_points":
                agg[metric_name] = int(np.sum(values))   # 点数：求和
            else:
                agg[metric_name] = float(np.mean(values)) # 其他指标：求均值
        aggregated[bin_label] = agg

    return aggregated


def _bin_sort_key(label):
    """
    将 bin 标签转为数值用于排序。

    排序规则: '0-5m' → 0, '5-10m' → 5, '50m+' → 50, 'overall' → inf

    Args:
        label: str，bin 标签字符串

    Returns:
        float: 用于排序的数值
    """
    if label == "overall":          # "overall" 总是排在最后
        return float("inf")
    try:
        # 取 "-" 前面的数字部分，例如 "10-15m" → "10"，"50m+" → "50"
        return int(label.split("-")[0].replace("m+", "").replace("m", ""))
    except ValueError:
        return float("inf")


def _print_binned_results(aggregated):
    """
    以格式化表格打印分 bin 聚合结果。

    输出列: Bin | N_pts | AbsRel | RMSE | d<1.25 | 3D_mean | 3D_med

    Args:
        aggregated: dict，{bin_label: {metric: value, ...}} 聚合结果字典
    """
    # 表头
    header = f"{'Bin':<12} {'N_pts':>10} {'AbsRel':>10} {'RMSE':>10} {'d<1.25':>10} {'3D_mean':>10} {'3D_med':>10}"
    print(header)
    print("-" * len(header))  # 分隔线
    # 按距离顺序遍历每个 bin
    for bin_label in sorted(aggregated.keys(), key=_bin_sort_key):
        m = aggregated[bin_label]
        print(
            f"{bin_label:<12} "                                          # bin 标签，左对齐
            f"{m.get('n_points', 0):>10} "                               # 有效点数
            f"{m.get('abs_rel', float('nan')):>10.4f} "                  # Abs Rel（4 位小数）
            f"{m.get('rmse', float('nan')):>10.4f} "                     # RMSE（4 位小数）
            f"{m.get('delta_125', float('nan')):>10.2f} "                # delta<1.25 百分比（2 位小数）
            f"{m.get('mean_3d_error', float('nan')):>10.4f} "            # 平均 3D 误差（4 位小数）
            f"{m.get('median_3d_error', float('nan')):>10.4f}"           # 中位 3D 误差（4 位小数）
        )


@hydra.main(
    version_base=None,                    # 使用 Hydra 默认版本行为
    config_path="../../configs",          # 配置文件搜索路径（相对于本脚本）
    config_name="fisheye_depth_benchmark" # 默认加载的配置文件名（不含 .yaml）
)
def execute_benchmarking(cfg: DictConfig):
    """
    Hydra 入口函数：解析配置、重定向日志、启动评测。

    Hydra 会自动从命令行和 YAML 文件解析配置，合并后传入 cfg。
    本函数将 stdout/stderr 重定向到 Python logging，然后调用 benchmark()。

    Args:
        cfg: DictConfig，Hydra 解析后的完整配置
    """
    # 将 DictConfig 转为可编辑的结构化配置（允许运行时修改）
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))
    # 将 stdout 和 stderr 重定向到日志系统，使 print() 输出也被记录到日志文件
    sys.stdout = StreamToLogger(log, logging.INFO)
    sys.stderr = StreamToLogger(log, logging.ERROR)
    # 启动评测主流程
    benchmark(cfg)


if __name__ == "__main__":
    execute_benchmarking()  # 脚本直接运行时的入口
