#!/bin/bash

# 逐 epoch 评测所有 checkpoint，找出深度精度最优的模型。
# 使用 GT 内参 (use_calibration=true) 作为模型输入。

export HYDRA_FULL_ERROR=1

# for epoch in 1 2 3 4 5 6 7 8 9 10; do
#     echo "========================================"
#     echo "Evaluating checkpoint-${epoch}.pth"
#     echo "========================================"

#     bash bash_scripts/benchmark/fisheye_depth.sh \
#         use_calibration=true \
#         model.pretrained='${root_experiments_dir}/mapanything/training/mapa_fisheye_04201413/checkpoint-'"${epoch}"'.pth' \
#         hydra.run.dir='${root_experiments_dir}/mapanything/benchmarking/fisheye_depth/epoch_'"${epoch}"

#     echo "Finished epoch ${epoch}"
#     echo ""
# done

# ---- 汇总所有 epoch 的评测结果 ----
# 从 machine config 获取实际路径（与 Hydra 的 ${root_experiments_dir} 对应）
RESULTS_BASE_DIR="/drobotics-ailab/bohao.zhang/work_dirs/mapanything/benchmarking/fisheye_depth"

echo ""
echo "============================================================"
echo "Summary: All Epochs Aggregated Results"
echo "============================================================"

python3 -c "
import json, os, sys

base_dir = '${RESULTS_BASE_DIR}'
epochs = list(range(1, 11))

# 收集所有 epoch 的结果
all_results = {}
for ep in epochs:
    path = os.path.join(base_dir, f'epoch_{ep}', 'TartanGroundWAI_aggregated.json')
    if not os.path.exists(path):
        continue
    with open(path) as f:
        all_results[ep] = json.load(f)

if not all_results:
    print('No results found.')
    sys.exit(0)

# 获取所有 bin 标签（从第一个有效结果中读取）
first = next(iter(all_results.values()))
bin_labels = list(first.keys())

# 列定义: (指标名, 表头显示名, 数据格式, 列宽)
columns = [
    ('n_points',        'N_pts',    'd',   12),
    ('abs_rel',         'AbsRel',   '.4f', 12),
    ('rmse',            'RMSE',     '.4f', 12),
    ('delta_125',       'd<1.25',   '.2f', 12),
    ('mean_3d_error',   '3D_mean',  '.4f', 12),
    ('median_3d_error', '3D_med',   '.4f', 12),
]

# 构建表头
header = f\"{'Epoch':<8}\"
for _, col_name, _, w in columns:
    header += f'{col_name:>{w}}'
print()

# 逐 bin 打印所有 epoch 的指标
for bin_label in bin_labels:
    print(f'--- {bin_label} ---')
    print(header)
    print('-' * len(header))
    for ep in sorted(all_results.keys()):
        data = all_results[ep].get(bin_label, {})
        row = f'EP-{ep:<5}'
        for key, _, fmt, w in columns:
            val = data.get(key, float('nan'))
            if fmt == 'd':
                row += f'{int(val):>{w}}'
            else:
                row += f'{val:>{w}{fmt}}'
        print(row)
    print()

# 打印 overall 的最优 epoch
if 'overall' in first:
    print('========== Best Epoch by Overall Metrics ==========')
    best_abs_rel = min(all_results.items(), key=lambda x: x[1].get('overall', {}).get('abs_rel', 999))
    best_delta = max(all_results.items(), key=lambda x: x[1].get('overall', {}).get('delta_125', 0))
    best_rmse = min(all_results.items(), key=lambda x: x[1].get('overall', {}).get('rmse', 999))
    print(f'  Best Abs Rel:   EP-{best_abs_rel[0]}  ({best_abs_rel[1][\"overall\"][\"abs_rel\"]:.4f})')
    print(f'  Best RMSE:      EP-{best_rmse[0]}  ({best_rmse[1][\"overall\"][\"rmse\"]:.4f})')
    print(f'  Best delta<1.25: EP-{best_delta[0]}  ({best_delta[1][\"overall\"][\"delta_125\"]:.2f}%)')
"
