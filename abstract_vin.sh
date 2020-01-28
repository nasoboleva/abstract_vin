#!/bin/bash
source ~/anaconda3/bin/activate pytorch
cd abstract_vin/
python generate_dataset.py --dim 2 --size 128 --exp_name ours_64_free_ --if_ours 1 --max_obs_num 6 --min_obs_num 1 --num_grids 7000 --type training
--paths_per_grid 7
python generate_dataset.py --dim 2 --size 128 --exp_name ours_64_f4_f4_f4_free_ --if_ours 1 --max_obs_num 6 --min_obs_num 1 --num_grids 5000 --type evaluation --paths_per_grid 1
python generate_dataset.py --dim 2 --size 128 --exp_name ours_64_free_ --if_ours 1 --max_obs_num 6 --min_obs_num 1 --num_grids 5000 --type validation --paths_per_grid 1
python train.py --dim 2 --net Abstraction_VIN --size 128 --exp_name ours_64_free_
#python test.py --dim 2 --net Abstraction_VIN --size 32 --exp_name ours_64
python visualize.py --dim 2 --size 128 --num 30 --exp_name ours_64_free_
