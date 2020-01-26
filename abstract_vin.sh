#!/bin/bash
source ~/anaconda3/bin/activate pytorch
cd abstract_vin/
#python generate_dataset.py --dim 2 --size 64 --exp_name low_dense_128_left --if_ours 1 --max_obs_num 6 --min_obs_num 1
#python train.py --dim 2 --net Abstraction_VIN --size 64 --exp_name low_dense_128_left
#python test.py --dim 2 --net Abstraction_VIN --size 32 --exp_name low_dense_64
python visualize.py --dim 2 --size 32 --num 30 --exp_name low_dense_64_left
