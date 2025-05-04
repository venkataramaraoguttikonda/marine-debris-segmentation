#!/bin/bash

# Run UNet training, prediction, evaluation, and visualization

python src/main_unet.py --train --model unet --epochs 100 --batch_size 8

python src/main_unet.py --predict --model unet

python src/main_unet.py --evaluate --model unet

python src/main_unet.py --visualize --model unet

echo "All steps completed successfully."