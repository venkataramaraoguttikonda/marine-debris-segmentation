#!/bin/bash

# Run SegFormer training, prediction, evaluation, and visualization

python src/main_segformer.py --train --epochs 100 --batch_size 8 --lr 8e-5

python src/main_segformer.py --predict

python src/main_segformer.py --evaluate

python src/main_segformer.py --visualize

echo "All steps completed successfully."