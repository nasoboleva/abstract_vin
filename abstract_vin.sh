#!/bin/bash
source ~/anaconda3/bin/activate pytorch
cd abstract_vin/
python generate_dataset.py --dim 2 --size 32
python train.py --dim 2 --net Abstraction_VIN --size 32
python visualize.py --dim 2 --size 32 --num 20
