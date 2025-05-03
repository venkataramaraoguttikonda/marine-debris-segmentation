#!/bin/bash

# Run SegFormer training, prediction, evaluation, and visualization

echo "Step 1: Training SegFormer model..."
python src/main_segformer.py --train --epochs 2 --batch_size 8 --lr 8e-5

echo "Step 2: Running inference on test set..."
python src/main_segformer.py --predict

echo "Step 3: Evaluating predictions..."
python src/main_segformer.py --evaluate

echo "Step 4: Visualizing predictions..."
python src/main_segformer.py --visualize

echo "All steps completed successfully."