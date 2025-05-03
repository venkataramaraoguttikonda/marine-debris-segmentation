#!/bin/bash

# Run UNet training, prediction, evaluation, and visualization

echo "Step 1: Training UNet model..."
python src/main_unet.py --train --model unet --epochs 100 --batch_size 8

echo "Step 2: Running inference on test set..."
python src/main_unet.py --predict --model unet

echo "Step 3: Evaluating predictions..."
python src/main_unet.py --evaluate --model unet

echo "Step 4: Visualizing predictions..."
python src/main_unet.py --visualize --model unet

echo "All steps completed successfully."