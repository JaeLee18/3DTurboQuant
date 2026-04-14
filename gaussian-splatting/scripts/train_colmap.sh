#!/bin/bash
# Train all COLMAP scenes: MipNeRF360 (9), Tanks&Temples (2), Deep Blending (2)
# No --white_background (these are real-world scenes)
PYTHON=/mnt/ssd1/conda_envs/gs_compression/bin/python
cd /mnt/ssd1/idea/TurboQuant/gaussian-splatting

# MipNeRF360
for scene in bicycle bonsai counter flowers garden kitchen room stump treehill; do
    echo "=========================================="
    echo "Training 360/$scene"
    echo "=========================================="
    if [ -f "output/${scene}/point_cloud/iteration_30000/point_cloud.ply" ]; then
        echo "  Already done, skipping"
        continue
    fi
    $PYTHON train.py -s data/360_v2/${scene} -m output/${scene} \
        --eval --iterations 30000 --save_iterations 7000 30000 \
        --test_iterations 7000 30000 --disable_viewer
    echo "  Done training $scene"
done

# Tanks & Temples
for scene in truck train; do
    echo "=========================================="
    echo "Training tandt/$scene"
    echo "=========================================="
    if [ -f "output/${scene}/point_cloud/iteration_30000/point_cloud.ply" ]; then
        echo "  Already done, skipping"
        continue
    fi
    $PYTHON train.py -s data/tandt/${scene} -m output/${scene} \
        --eval --iterations 30000 --save_iterations 7000 30000 \
        --test_iterations 7000 30000 --disable_viewer
    echo "  Done training $scene"
done

# Deep Blending
for scene in playroom drjohnson; do
    echo "=========================================="
    echo "Training db/$scene"
    echo "=========================================="
    if [ -f "output/${scene}/point_cloud/iteration_30000/point_cloud.ply" ]; then
        echo "  Already done, skipping"
        continue
    fi
    $PYTHON train.py -s data/db/${scene} -m output/${scene} \
        --eval --iterations 30000 --save_iterations 7000 30000 \
        --test_iterations 7000 30000 --disable_viewer
    echo "  Done training $scene"
done

echo "All COLMAP scenes trained."
