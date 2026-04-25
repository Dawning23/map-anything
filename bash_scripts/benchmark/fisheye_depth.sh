#!/bin/bash

# Benchmark fisheye depth prediction on TartanGround fisheye validation set.
#
# Geometric input conditioning modes (set via command line override):
#   默认 (images only):  use_calibration=false use_pose=false
#   + GT 内参:           use_calibration=true
#   + GT 内参+位姿:      use_calibration=true use_pose=true use_pose_scale=true
#
# Example:
#   bash fisheye_depth.sh                          # images only
#   bash fisheye_depth.sh use_calibration=true     # with GT ray_dirs

export HYDRA_FULL_ERROR=1

python3 \
    benchmarking/fisheye_depth/benchmark.py \
    machine=huoshan \
    dataset=tartan_ground_fisheye \
    dataset.num_workers=12 \
    batch_size=10 \
    model=fisheyemapanything_dino_init_small \
    model.pretrained='${root_experiments_dir}/mapanything/training/mapa_fisheye_04201413/checkpoint-best.pth' \
    hydra.run.dir='${root_experiments_dir}/mapanything/benchmarking/fisheye_depth/mapa_fisheye' \
    "$@"
