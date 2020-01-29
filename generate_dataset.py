import numpy as np
import torch
import torch.nn as nn
import random
import time
from multiprocessing import Pool
import argparse
import cv2
import os
import glob
import math

from utils import path_to_action_vectors, extract_path
from astar import astar_planner_2d, astar_planner_3d, dijkstra_planner_2d, dijkstra_planner_3d


# class for environment representation (containing obstacle map, start and goal pose)
class LabeledExample:
    def __init__(self, size):
        self.size = size                    # grid world size
        self.start = (size // 2, size // 2) # start position (center of map)
        self.map = torch.zeros(size, size)  # obstacle map
        self.number_of_obstacles = 0
        self.goal = ()

        # only for 3D motion planning
        self.orientation = 0                # start orientation
        self.goal_orientation = 0
        self.num_orientations = 16          # number of discrete orientations
        self.rotation_step_size = 2*np.pi/self.num_orientations # difference between two adjacent discrete orientations
        self.leg_x = 2                      # distance between robot base and wheel (x-coordinate)
        self.leg_y = 2                      # distance between robot base and wheel (y-coordinate)


    def add_random_rectangular_obstacles(self, number, max_obs_height, max_obs_width, if_ours=False, from_image=None):
        if from_image:
            image = cv2.imread(from_image).mean(-1)
            im_size = image.shape[0]
            if im_size == self.size:
                self.x_crop = 0
                self.y_crop = 0
            else:
                self.x_crop = np.random.choice(np.arange(0, im_size - self.size))
                self.y_crop = np.random.choice(np.arange(0, im_size - self.size))
            self.map = image[self.y_crop:self.y_crop+self.size, self.y_crop:self.y_crop+self.size]
            print(self.map.shape, np.unique(self.map))

            self.y_start = np.random.choice(np.arange(self.size // 4, self.size //2 ))

        elif if_ours:
            self.y_start = np.random.choice(np.arange(self.size // 4, self.size //2 ))
            while self.number_of_obstacles < number:
                dencity = 0.2
                indent = 3
                max_proportion = math.floor(math.sqrt((self.size // 4) ** 2 * dencity // 6))
                proportion = np.random.choice(np.arange(1, max_proportion + 1))
                obst_width = 2 * proportion
                obst_height = 3 * proportion
                x = np.random.choice(np.arange(self.size // 2 + indent, self.size // 4 * 3 - indent - obst_height))
                y = np.random.choice(np.arange(self.y_start, self.y_start + self.size // 4 - obst_height))
                if self.number_of_obstacles % 2 != 0:
                    for dx in range(obst_width):
                        for dy in range(obst_height):
                            self.map[y + dy][x + dx] = 1
                else:
                    for dx in range(obst_height):
                        for dy in range(obst_width):
                            self.map[y + dy][x + dx] = 1
                #self.map[:self.size // 4, :] = 1
                #self.map[-self.size // 4:, :] = 1
                #self.map[self.size // 4:-self.size // 4, :self.size // 2] = 1
                self.number_of_obstacles += 1
        else:
            while self.number_of_obstacles < number:
                # sample random position
                x = np.random.randint(0, high=self.size)
                y = np.random.randint(0, high=self.size)

                # sample random size
                height = np.random.randint(1, high=max_obs_height)
                width = np.random.randint(1, high=max_obs_width)

                # ensure that obstacle lies completly within map
                if x+width >= self.size:
                    width = self.size-x
                if y+height >= self.size:
                    height = self.size-y

                # add obstacle
                self.number_of_obstacles += 1
                self.map[x:x+width, y:y+height] = 1


# generates one environment with multiple paths
# Input:    -idx: index of generated map
#           -use_dijkstra: use Dijkstra instead of A* to compute all paths in parallel
# Output: Occupancy map and list of expert paths
def generate_map_with_trajectories(idx, from_images=None, use_dijkstra=True):
    # reset random seed
    np.random.seed()
    # get global parameters
    size, min_number_of_obstacles, max_number_of_obstacles, max_obs_height, max_obs_width, num_paths_per_grid, for_3d, if_ours = parameter
    if from_images:
        path_images = glob.glob(from_images + '*.png')
    failed = True
    while failed:
        failed = False

        # create empty map of doubled size (to allow cropping patches of the desired size for each position on the expert paths)
        example = LabeledExample(2*size)
        path_list = []
        action_list = []

        # add random obstacles
        obsts = np.random.choice(np.arange(1, max_number_of_obstacles + 1))

        #number_of_obstacles = np.random.randint(4*min_number_of_obstacles, high=4*max_number_of_obstacles)
        if from_images:
            example.add_random_rectangular_obstacles(obsts, max_obs_height, max_obs_width, if_ours, np.random.choice(path_images))
        else:
            example.add_random_rectangular_obstacles(obsts, max_obs_height, max_obs_width, if_ours)

        # set start position at the center of the map
        example.start = (size, size)
        # make sure that the start position does not contain any obstacles
        if for_3d:
            example.map[size-example.leg_x:size+example.leg_x+1,size-example.leg_y:size+example.leg_y+1] = 0
        else:
            if if_ours:

                if example.map[example.start[0], example.start[1]] != 0:
                    failed = True
                    continue
            else:
                example.map[size,size] = 0

        num_paths = 0
        counter = 0
        # Use Dijkstra
        if use_dijkstra:
            if for_3d:
                # sample start orientation
                example.orientation = np.random.randint(0,high=example.num_orientations)
                distances, predecessors = dijkstra_planner_3d(example)
            else:

                distances, predecessors = dijkstra_planner_2d(example, if_ours)

            while num_paths < num_paths_per_grid:
                counter += 1
                # sample random goal state
                if if_ours:
                    indent = 3
                    x = np.random.randint(example.y_start, high= example.y_start + example.size // 4)
                    y = np.random.randint(size // 2 * 3 - indent, high=  size // 2 * 3)
                    if example.map[x, y] != 0:
                        failed = True
                        break
                else:
                    x = np.random.randint(size//2+example.leg_x, high= size//2 + size-example.leg_x)
                    y = np.random.randint(size//2+example.leg_y, high= size//2 + size-example.leg_y)
                if for_3d:
                    # sample random goal orientation
                    theta = np.random.randint(0,high=example.num_orientations)

                    # check if valid path exists:
                    if distances[x,y,theta] != float('Inf'):
                        # get optimal path
                        optimal_path = list(reversed(extract_path((size,size,example.orientation),(x,y,theta),predecessors)))
                        # only allow paths with at least two different poses
                        if len(optimal_path) <= 1:
                            # if not enough paths were generated: start again with empty map
                            if counter > 20:
                                failed = True
                                break
                            continue
                        optimal_path = torch.stack(optimal_path, dim=0)
                        action_sequence = path_to_action_vectors(optimal_path, dim=3)
                        action_sequence = torch.stack(action_sequence, dim=0).float()
                        path_list.append(optimal_path)
                        action_list.append(action_sequence)
                        num_paths +=1
                    else:
                        # if not enough paths were generated: start again with empty map
                        if counter > 20:
                            failed = True
                            break
                else:
                    # check if valid path exists:
                    if distances[x,y] != float('Inf'):
                        #print(x, y)
                        #plt.imshow(example.map)
                        # get optimal path
                        optimal_path = list(reversed(extract_path((size,size),(x,y),predecessors, dim=2)))
                        if len(optimal_path) <= 1:
                            # if not enough paths were generated: start again with empty map
                            if counter > 20:
                                failed = True
                                break
                            continue
                        optimal_path = torch.stack(optimal_path, dim=0)
                        action_sequence = path_to_action_vectors(optimal_path, dim=2)
                        action_sequence = torch.stack(action_sequence, dim=0).float()
                        path_list.append(optimal_path)
                        action_list.append(action_sequence)
                        num_paths +=1
                    else:
                        # if not enough paths were generated: start again with empty map
                        if counter > 20:
                            failed = True
                            break
        # Use A* planner
        else:
            while num_paths < num_paths_per_grid:
                counter += 1
                # add random goal
                if if_ours:
                    x = np.random.randint(1, high= size*2 - 1)
                    y = np.random.randint( size* 2-1 - 3, high=  size*2-1)
                else:
                    x = np.random.randint(size//2+example.leg_x, high= size//2 + size-example.leg_x)
                    y = np.random.randint(size//2+example.leg_y, high= size//2 + size-example.leg_y)
                example.goal = (x,y)
                # ensure that goal position is free
                if if_ours:
                    if example.map[x,y] != 0:
                        continue
                else:
                    example.map[x,y] = 0
                if for_3d:
                    # sample random start and goal orientations
                    example.orientation = np.random.randint(0,high=example.num_orientations)
                    example.goal_orientation = np.random.randint(0,high=example.num_orientations)
                    optimal_path = list(reversed(astar_planner_3d(example)))
                else:
                    optimal_path = list(reversed(astar_planner_2d(example)))
                if len(optimal_path) <= 1:
                    if counter > 20:
                        failed = True
                        break
                    continue

                optimal_path = torch.stack(optimal_path, dim=0)

                if for_3d:
                    action_sequence = path_to_action_vectors(optimal_path, dim=3)
                    action_sequence = torch.stack(action_sequence, dim=0).float()
                else:
                    action_sequence = path_to_action_vectors(optimal_path, dim=2)
                    action_sequence = torch.stack(action_sequence, dim=0).float()

                path_list.append(optimal_path)
                action_list.append(action_sequence)
                num_paths +=1

    return example.map, path_list, action_list, example.y_start

def generate_data(number_of_examples, size, min_number_of_obstacles,
                  max_number_of_obstacles, max_obs_height=2, max_obs_width=2,
                  num_path_per_grid=7, for_3d=False, data='training',
                  num_workers=4, make_images=True, exp_name='1', if_ours=False,
                  from_images=None):
    if data=='training':
        print('Generating Training Data')
    elif data=='validation':
        print('Generating Validation Data')
    elif data=='evaluation':
        print('Generating Evaluation Data')
    print('Workers: ', num_workers)

    # make global parameter available for each worker
    global parameter
    parameter = (size, min_number_of_obstacles, max_number_of_obstacles,
                 max_obs_height, max_obs_width, num_path_per_grid, for_3d,
                 if_ours)

    inputs = []
    paths = []
    actions = []

    maps = []

    for ex in range(number_of_examples):
        maps += [generate_map_with_trajectories(ex, from_images=from_images)]
    #pool = Pool(processes=num_workers)
    #maps = pool.map(generate_map_with_trajectories, range(number_of_examples))
    #pool.close()

    if data=='training':
        if for_3d:
            os.makedirs('data_sets/3D/' + str(exp_name) + '_imgs', exist_ok=True)
            os.makedirs('data_sets/3D/' + str(exp_name), exist_ok=True)
            file_path = 'data_sets/3D/' + str(exp_name) + '_imgs/trainingset_'+str(size)
            file_path_vin = 'data_sets/3D/' + str(exp_name) + '/trainingset_'+str(size)
        else:
            os.makedirs('data_sets/2D/' + str(exp_name) + '_imgs', exist_ok=True)
            os.makedirs('data_sets/2D/' + str(exp_name), exist_ok=True)
            file_path = 'data_sets/2D/' + str(exp_name) + '_imgs/trainingset_'+str(size)
            file_path_vin = 'data_sets/2D/' + str(exp_name) + '/trainingset_'+str(size)
    elif data=='validation':
        if for_3d:
            os.makedirs('data_sets/3D/' + str(exp_name)+ '_imgs', exist_ok=True)
            os.makedirs('data_sets/3D/' + str(exp_name), exist_ok=True)
            file_path = 'data_sets/3D/' + str(exp_name) + '_imgs/validationset_'+str(size)
            file_path_vin = 'data_sets/3D/' + str(exp_name) + '/validationset_'+str(size)
        else:
            os.makedirs('data_sets/2D/' + str(exp_name) + '_imgs', exist_ok=True)
            os.makedirs('data_sets/2D/' + str(exp_name), exist_ok=True)
            file_path = 'data_sets/2D/' + str(exp_name) + '_imgs/validationset_'+str(size)
            file_path_vin = 'data_sets/2D/' + str(exp_name) + '/validationset_'+str(size)
    elif data=='evaluation':
        if for_3d:
            os.makedirs('data_sets/3D/' + str(exp_name) + '_imgs', exist_ok=True)
            os.makedirs('data_sets/3D/' + str(exp_name), exist_ok=True)
            file_path = 'data_sets/3D/' + str(exp_name) + '_imgs/evaluationset_'+str(size)
            file_path_vin = 'data_sets/3D/' + str(exp_name)  + '/evaluationset_'+str(size)
        else:
            os.makedirs('data_sets/2D/' + str(exp_name) + '_imgs', exist_ok=True)
            os.makedirs('data_sets/2D/' + str(exp_name), exist_ok=True)
            file_path = 'data_sets/2D/' + str(exp_name) + '_imgs/evaluationset_'+str(size)
            file_path_vin = 'data_sets/2D/' + str(exp_name) + '/evaluationset_'+str(size)

    for i, (map, path_list, action_list, y_start) in enumerate(maps):
        if make_images:
            if from_images:
                map_img = map.copy()
            else:
                map_img = map.numpy().copy()
            map_img[map_img == 0] = 2
            for j, path in enumerate(path_list):
                curr_map_img = map_img.copy()
                start, goal = path[0].numpy(), path[-1].numpy()
                curr_map_img[start[0], start[1]] = 0
                curr_map_img[goal[0], goal[1]] = 0

                cv2.imwrite(file_path + '_' + str(i) + '_' + str(j) + 'full.png',
                            curr_map_img.astype('uint8'))
                cv2.imwrite(file_path + '_' + str(i) + '_' + str(j) + '.png',
                            curr_map_img[y_start:y_start + size //2, size:size+size//2].astype('uint8'))
                for path_cell in path[1:-1]:
                    curr_map_img[path_cell[0], path_cell[1]] = 0
                cv2.imwrite(file_path + '_' + str(i) + '_' + str(j) + '_logfull.png',
                            curr_map_img.astype('uint8'))
                cv2.imwrite(file_path + '_' + str(i) + '_' + str(j) + '_log.png',
                            curr_map_img[y_start:y_start + size // 2, size:size+size//2].astype('uint8'))

        inputs.append(map)
        paths.append(path_list)
        actions.append(action_list)

    torch.save({'inputs':inputs, 'paths':paths, 'actions':actions}, file_path_vin +'.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dim',
        type=int,
        default=2,
        help='Dimensionality of the grid: either 2 for the 2D grid world task or 3 for 3D locomotion planning.')
    parser.add_argument('--size', type=int, default=32, help='Size of grid world')
    parser.add_argument(
        '--max_obs_num',
        type=int,
        help='Maximum number of obstacles.')
    parser.add_argument(
        '--min_obs_num',
        type=int,
        help='Minimum number of obstacles.')
    parser.add_argument(
        '--max_obs_height',
        type=int,
        help='Maximum height of obstacles.')
    parser.add_argument(
        '--max_obs_width',
        type=int,
        help='Maximum width of obstacles.')
    parser.add_argument(
        '--paths_per_grid',
        type=int,
        default=7,
        help='Number of expert paths for each grid.')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of workers for parallel grid creation.')
    parser.add_argument(
        '--type',
        type=str,
        default='all',
        help='Type of dataset. One of [training, validation, evaluation, all]. Default: all.')
    parser.add_argument(
        '--num_grids',
        type=int,
        help='Number of different grid worlds to be created. Only if --type is set to all, default values will be used:5000 for training and 715 each for validation and evaluation.')
    parser.add_argument(
        '--make_images',
        type=bool,
        default=True,
        help='True, if create images for GAN-finder.')
    parser.add_argument(
        '--exp_name',
        type=str,
        help='Name of the data and results folder.')
    parser.add_argument(
        '--if_ours',
        type=bool,
        help='Name of the data and results folder.')
    parser.add_argument(
        '--from_images',
        type=str,
        help='folder with map images')
    param = parser.parse_args()

    # set size/dim dependent default values
    if param.max_obs_num is None:
        if param.dim == 2:
            param.max_obs_num = 144
        else:
            param.max_obs_num = 21

    if param.min_obs_num is None:
        if param.dim == 2:
            param.min_obs_num = 71
        else:
            param.min_obs_num = 6

    if param.max_obs_height is None:
        if param.dim == 2:
            param.max_obs_height = 3*param.size//32
        else:
            param.max_obs_height = 5*param.size//32

    if param.max_obs_width is None:
        param.max_obs_width = param.max_obs_height

    if param.num_grids is None:
        if param.type == 'training':
            param.num_grids = 5000
        else:
            param.num_grids = 715

    if param.make_images is None:
        param.make_images = True

    if param.dim==3:
        for_3d = True
    else:
        for_3d = False

    # generate data
    if param.type == 'training':
        generate_data(param.num_grids, param.size, param.min_obs_num,
                      param.max_obs_num, param.max_obs_height,
                      param.max_obs_width, param.paths_per_grid,
                      for_3d=for_3d, data='training',
                      num_workers=param.num_workers, make_images=param.make_images,
                      exp_name=param.exp_name, if_ours=param.if_ours, from_images=param.from_images)
    elif param.type == 'validation':
        generate_data(param.num_grids, param.size, param.min_obs_num,
                      param.max_obs_num, param.max_obs_height,
                      param.max_obs_width, param.paths_per_grid,
                      for_3d=for_3d, data='validation',
                      num_workers=param.num_workers, make_images=param.make_images,
                      exp_name=param.exp_name, if_ours=param.if_ours, from_images=param.from_images)
    elif param.type == 'evaluation':
        generate_data(param.num_grids, param.size, param.min_obs_num,
                      param.max_obs_num, param.max_obs_height,
                      param.max_obs_width, param.paths_per_grid,
                      for_3d=for_3d, data='evaluation',
                      num_workers=param.num_workers, make_images=param.make_images,
                      exp_name=param.exp_name, if_ours=param.if_ours, from_images=param.from_images)
    elif param.type == 'all':
        generate_data(5000, param.size, param.min_obs_num, param.max_obs_num,
                      param.max_obs_height, param.max_obs_width,
                      param.paths_per_grid, for_3d=for_3d, data='training',
                      num_workers=param.num_workers, make_images=param.make_images,
                      exp_name=param.exp_name, if_ours=param.if_ours, from_images=param.from_images)
        generate_data(715, param.size, param.min_obs_num, param.max_obs_num,
                      param.max_obs_height, param.max_obs_width,
                      param.paths_per_grid, for_3d=for_3d, data='validation',
                      num_workers=param.num_workers, make_images=param.make_images,
                      exp_name=param.exp_name, if_ours=param.if_ours, from_images=param.from_images)
        generate_data(715, param.size, param.min_obs_num, param.max_obs_num,
                      param.max_obs_height, param.max_obs_width,
                      param.paths_per_grid, for_3d=for_3d, data='evaluation',
                      num_workers=param.num_workers, make_images=param.make_images,
                      exp_name=param.exp_name, if_ours=param.if_ours, from_images=param.from_images)
    else:
        print('Invalid dataset type.')
