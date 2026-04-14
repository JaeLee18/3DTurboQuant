#!/bin/bash
# Train all 8 NeRF Synthetic scenes with white background
PYTHON=/mnt/ssd1/conda_envs/gs_compression/bin/python
DATA_ROOT=data/nerf_synthetic
cd /mnt/ssd1/idea/TurboQuant/gaussian-splatting

SCENES="lego chair drums ficus hotdog materials mic ship"

for scene in $SCENES; do
    echo "=========================================="
    echo "Training $scene"
    echo "=========================================="

    if [ -f "output/${scene}_wb/point_cloud/iteration_30000/point_cloud.ply" ]; then
        echo "  Already trained, skipping."
        continue
    fi

    $PYTHON train.py \
        -s ${DATA_ROOT}/${scene} \
        -m output/${scene}_wb \
        --white_background \
        --iterations 30000 \
        --eval \
        --save_iterations 1000 3000 7000 15000 30000 \
        --disable_viewer

    echo "  Done training $scene"
done

echo "All scenes trained."
