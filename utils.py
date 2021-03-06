from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
from math import sqrt, cos, sin
import numpy as np
import os

actions = {(0,0):0, (-1,-1):1,(-1,0):2,(-1,1):3,(0,-1):4,(0,1):5,(1,-1):6,(1,0):7,(1,1):8}
actions_3d = {(0,0,0):0, (-1,-1,0):1,(-1,0,0):2,(-1,1,0):3,(0,-1,0):4,(0,1,0):5,(1,-1,0):6,(1,0,0):7,(1,1,0):8, (0,0,-1):9, (0,0,1):10}
actions_to_moves = {0:(0,0), 1:(-1,-1),2:(-1,0),3:(-1,1),4:(0,-1),5:(0,1),6:(1,-1),7:(1,0),8:(1,1)}
actions_to_moves_3d = {0:(0,0,0), 1:(-1,-1,0),2:(-1,0,0),3:(-1,1,0),4:(0,-1,0),5:(0,1,0),6:(1,-1,0),7:(1,0,0),8:(1,1,0), 9:(0,0,-1), 10:(0,0,1)}

# extract path from predecessor list generated by Dijkstra's algorithm
# Input:    -start: start pose
#           -goal: goal pose
#           -predecessor: array containing predecessor (on optimal path) for each state
#           -dim: 2 or 3 (for 2D grid world or 3D locomotion planning task)
def extract_path(start, goal, predecessor, dim=3):
    path = []
    current = goal

    if dim == 2:
        while (current[0] != start[0] or current[1] != start[1]):
            new_list = list(map(int, [current[0],current[1]]))
            path.append(torch.tensor(new_list))
            current = (predecessor[0,current[0],current[1]],predecessor[1,current[0],current[1]])

        start_list = list(map(int, [start[0],start[1]]))
        path.append(torch.tensor(start_list))
    elif dim == 3:
        while (current[0] != start[0] or current[1] != start[1] or current[2] != start[2]):
            path.append(torch.tensor([current[0],current[1],current[2]]))
            current = (predecessor[0,current[0],current[1],current[2]],predecessor[1,current[0],current[1],current[2]],predecessor[2,current[0],current[1],current[2]])

        path.append(torch.tensor([start[0],start[1],start[2]]))
    else:
        print('Dimensionality %d is not supported.' %dim)

    return path

# generate footprint masks for each discrete orientation and each abstraction level
# Input:    -leg_x, leg_y: absolute values of distance between wheel and robot base position
#           -num_orientations: number of discrete orientations
# Output: one list for each abstraction level,
#         each list contains one binary kernel where the cells containing wheels are marked with '1'
def calculate_local_footprints_mulitlayer(leg_x, leg_y, num_orientations):
    # base position in local robot coordinates
    base1 = torch.tensor([2,2])
    base2 = torch.tensor([1,1])
    base3 = torch.tensor([1,1])

    local_footprints_1 = []
    local_footprints_2 = []
    local_footprints_3 = []

    # wheel positions in local robot coordinates
    local_fl = torch.tensor([leg_x,leg_y]).float()
    local_fr = torch.tensor([leg_x,-leg_y]).float()
    local_bl = torch.tensor([-leg_x,leg_y]).float()
    local_br = torch.tensor([-leg_x,-leg_y]).float()

    # compute global wheel positions for each possible orientation
    for orientation in range(num_orientations):
        theta = orientation*2*np.pi/num_orientations
        # rotation matrix
        rotation = torch.tensor([[cos(-theta), sin(-theta)],[-sin(-theta), cos(-theta)]])

        # global coordinates
        fl_1 = base1 + torch.mv(rotation,local_fl).long()
        fr_1 = base1 + torch.mv(rotation,local_fr).long()
        bl_1 = base1 + torch.mv(rotation,local_bl).long()
        br_1 = base1 + torch.mv(rotation,local_br).long()

        fl_2 = base2 + torch.mv(rotation,local_fl).long()//2
        fr_2 = base2 + torch.mv(rotation,local_fr).long()//2
        bl_2 = base2 + torch.mv(rotation,local_bl).long()//2
        br_2 = base2 + torch.mv(rotation,local_br).long()//2

        fl_3 = base3 + torch.mv(rotation,local_fl).long()//4
        fr_3 = base3 + torch.mv(rotation,local_fr).long()//4
        bl_3 = base3 + torch.mv(rotation,local_bl).long()//4
        br_3 = base3 + torch.mv(rotation,local_br).long()//4

        # footprint masks for Level-1
        footprint_1 = torch.zeros(5,5)
        footprint_1[fl_1[0],fl_1[1]] = 1
        footprint_1[fr_1[0],fr_1[1]] = 1
        footprint_1[bl_1[0],bl_1[1]] = 1
        footprint_1[br_1[0],br_1[1]] = 1

        local_footprints_1.append(footprint_1)

        # footprint masks for Level-2
        footprint_2 = torch.zeros(3,3)
        footprint_2[fl_2[0],fl_2[1]] = 1
        footprint_2[fr_2[0],fr_2[1]] = 1
        footprint_2[bl_2[0],bl_2[1]] = 1
        footprint_2[br_2[0],br_2[1]] = 1

        local_footprints_2.append(footprint_2)

        # footprint masks for Level-3
        footprint_3 = torch.zeros(3,3)
        footprint_3[fl_3[0],fl_3[1]] = 1
        footprint_3[fr_3[0],fr_3[1]] = 1
        footprint_3[bl_3[0],bl_3[1]] = 1
        footprint_3[br_3[0],br_3[1]] = 1

        local_footprints_3.append(footprint_3)
    local_footprints_1 = torch.stack(local_footprints_1, dim=0)
    local_footprints_2 = torch.stack(local_footprints_2, dim=0)
    local_footprints_3 = torch.stack(local_footprints_3, dim=0)

    return local_footprints_1, local_footprints_2, local_footprints_3

# returns the current wheel positions (for the 3D task)
# Input:    -pose: robot base position and orientation
#           -rotation_step_size: difference between two adjacent discretized orientations
#           -leg_x, leg_y: absolute values of distance between wheel and robot base position
# Output: wheel coordinates (float values)
def get_wheel_coord(pose, rotation_step_size, leg_x, leg_y):
    base = torch.tensor([pose[0],pose[1]]).float()
    theta = pose[2].item()*rotation_step_size

    # local wheel coordinates
    local_fl = torch.tensor([leg_x,leg_y]).float()
    local_fr = torch.tensor([leg_x,-leg_y]).float()
    local_bl = torch.tensor([-leg_x,leg_y]).float()
    local_br = torch.tensor([-leg_x,-leg_y]).float()

    # global coordinates
    rotation = torch.tensor([[cos(-theta), sin(-theta)],[-sin(-theta), cos(-theta)]])
    fl = base + torch.mv(rotation,local_fl)
    fr = base + torch.mv(rotation,local_fr)
    bl = base + torch.mv(rotation,local_bl)
    br = base + torch.mv(rotation,local_br)

    return (fl,fr,bl,br)

# returns difference between old and new pose as vector
# Input:    -action_vector: action probabilities (output from network)
#           -dim: task dimensionality (2 or 3)
def get_action(action_vector, dim=2):
    action = action_vector.argmax(0).item()
    if dim == 2:
        return torch.tensor([actions_to_moves[action][0],actions_to_moves[action][1]])
    elif dim == 3:
        return torch.tensor([actions_to_moves_3d[action][0],actions_to_moves_3d[action][1],actions_to_moves_3d[action][2]])
    else:
        print("Error: Invalid search space dimension.")
        return 1

# extract action probabilities from path
# Input:    -path
#           -dim: task dimensionality (2 or 3)
#           -num_orientations: number of discrete orientations
# Output:   list of one-hot vectors, encoding action for each step
def path_to_action_vectors(path, dim=2, num_orientations=16):
    action_sequence = []

    if dim == 2:
        for i in range(len(path)-1):
            # get action index
            action = actions[((path[i+1][0]-path[i][0]).item(),(path[i+1][1]-path[i][1]).item())]

            # one-hot vector
            action_vector = torch.zeros(9)
            action_vector[action] = 1
            action_sequence.append(action_vector)

        # add 'stop' action
        action_vector = torch.zeros(9)
        action_vector[0] = 1
        action_sequence.append(action_vector)
    elif dim == 3:
        for i in range(len(path)-1):
            rotation_index = (path[i+1][2]-path[i][2]).item()
            if rotation_index == -num_orientations+1:
                rotation_index = 1
            elif rotation_index == num_orientations-1:
                rotation_index = -1

            # get action index
            action = actions_3d[((path[i+1][0]-path[i][0]).item(),(path[i+1][1]-path[i][1]).item(),rotation_index)]

            # one-hot vector
            action_vector = torch.zeros(11)
            action_vector[action] = 1
            action_sequence.append(action_vector)

        # add 'stop' action
        action_vector = torch.zeros(11)
        action_vector[0] = 1
        action_sequence.append(action_vector)
    else:
        print("Error: Undefined search space dimension.")
    return action_sequence

# calculates path length
# Input:    -path
#           -dim: task dimensionality (2 or 3)
#           -rotation_step_size: difference between two adjacent discretized orientations
#           -leg_x, leg_y: absolute values of distance between wheel and robot base position
def get_path_length(path, dim=2, leg_x=2, leg_y=2, rotation_step_size=2*np.pi/16.):
    length = 0.

    for i in range(1,path.size(0)):
        if dim==2:
            length += torch.dist(path[i].float(),path[i-1].float()).item()
        if dim==3:
            # add distance between base positions
            length += torch.dist(path[i,0:2].float(),path[i-1,0:2].float()).item()
            # distance between orientations
            if path[i,2] != path[i-1,2]:
                # add cost for one rotation step
                length += np.sqrt(leg_x**2+leg_y**2)*rotation_step_size

    return length

# plots paths for 2D grid world task
# Input:    -map: occupancy map of environment
#           -goal: goal position
#           -opt_path: expert path
#           -path_list: list of paths predicted by the different networks
#           -path_labels: list of names of the corresponding networks
def visualize_2d(map, goal, opt_path, path_list, path_labels, number, exp_name="1"):
    colours = ['b', 'r', 'orange']

    opt_path = np.array(opt_path)
    centre = (map.size()[0]//2,map.size()[1]//2)
    fig, ax = plt.subplots()

    # plot occupancy map
    implot = plt.imshow(map, cmap='Greys',interpolation='none')

    # plot network paths
    for i in range(len(path_list)):
        if path_list[i] is None:
            continue
        path = path_list[i]
        path = path.numpy()
        path = np.array(path)
        ax.plot(path[:, 1]+i*0.1, path[:, 0]+i*0.1, c=colours[i], label=path_labels[i])

    # plot optimal path
    ax.plot(opt_path[:, 1]-0.1, opt_path[:, 0]-0.1, c='black', label='Optimal Path')
    ax.plot(centre[1], centre[0], '-o', c='c', label='Start')
    ax.plot(goal[1], goal[0], '-s', c='c',label='Goal')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

    # Put a legend below current axis
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=3)

    for label in legend.get_texts():
        label.set_fontsize('small')  # the legend text size
    for label in legend.get_lines():
        label.set_linewidth(1)  # the legend line width

    plt.xlim(map.size()[0]//4-0.5, map.size()[0]//4-0.5+map.size()[0]//2)
    plt.ylim(map.size()[1]//4-0.5, map.size()[1]//4-0.5+map.size()[1]//2)

    os.makedirs('./visualization/' + exp_name + '/', exist_ok=True)
    plt.savefig('./visualization/' + exp_name + '/img_' + str(number) + '.png')

    #plt.draw()
    #plt.waitforbuttonpress(0)
    #plt.close(fig)

# plots paths for 3D locomotion planning task
# Input:    -map: occupancy map of environment
#           -goal: goal position
#           -opt_path: expert path
#           -net_path: path predicted by network
#           -rotation_step_size: difference between two adjacent discretized orientations
#           -leg_x, leg_y: absolute values of distance between wheel and robot base position
def visualize_3d(map, goal, opt_path, net_path,leg_x=2, leg_y=2, rotation_step_size=2*np.pi/16.):
    centre = (map.size()[0]//2,map.size()[1]//2)
    fig, ax = plt.subplots()

    # plot occupancy map
    implot = plt.imshow(map, cmap='Greys',interpolation='none')


    # plot network path:
        # get wheel trajectories
    wheel_trajectories = get_wheel_trajectories(net_path, leg_x, leg_y, rotation_step_size)
    wheel_trajectories = wheel_trajectories.numpy()

        # plot base trajectory
    net_path = net_path.numpy()
    net_path = np.array(net_path)
    ax.plot(net_path[:, 1], net_path[:, 0], c='blue', label='Abstraction VIN (base)')

        # plot wheel trajectories
    ax.plot(wheel_trajectories[:,0,1], wheel_trajectories[:,0,0], c='darkblue', label='Abstraction VIN (fl)', linestyle='dotted')
    ax.plot(wheel_trajectories[:,1,1], wheel_trajectories[:,1,0], c='darkblue', label='Abstraction VIN (fr)', linestyle='dashed')
    ax.plot(wheel_trajectories[:,2,1], wheel_trajectories[:,2,0], c='lightblue', label='Abstraction VIN (bl)', linestyle='dotted')
    ax.plot(wheel_trajectories[:,3,1], wheel_trajectories[:,3,0], c='lightblue', label='Abstraction VIN (br)', linestyle='dashed')

    # plot optimal path:
        # get wheel trajectories
    wheel_trajectories = get_wheel_trajectories(opt_path, leg_x, leg_y, rotation_step_size).numpy()

        # plot base trajectory
    opt_path = np.array(opt_path)
    ax.plot(opt_path[:, 1]-0.1, opt_path[:, 0]-0.1, c='green', label='Optimal Path (base)')

        # plot wheel trajectories
    ax.plot(wheel_trajectories[:,0,1]+0.1, wheel_trajectories[:,0,0]+0.1, linestyle='dotted', c='darkgreen', label='Optimal Path (fl)')
    ax.plot(wheel_trajectories[:,1,1]+0.1, wheel_trajectories[:,1,0]+0.1, linestyle='dashed', c='darkgreen', label='Optimal Path (fr)')
    ax.plot(wheel_trajectories[:,2,1]+0.1, wheel_trajectories[:,2,0]+0.1, linestyle='dotted', c='lightgreen', label='Optimal Path (bl)')
    ax.plot(wheel_trajectories[:,3,1]+0.1, wheel_trajectories[:,3,0]+0.1, linestyle='dashed', c='lightgreen', label='Optimal Path (br)')

    ax.plot(centre[1], centre[0], '-o', c='c', label='Start')
    ax.plot(goal[1], goal[0], '-s', c='c',label='Goal')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

    # Put a legend below current axis
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=3)

    for label in legend.get_texts():
        label.set_fontsize('small')  # the legend text size
    for label in legend.get_lines():
        label.set_linewidth(1)  # the legend line width

    plt.xlim(map.size()[0]//4-0.5, map.size()[0]//4-0.5+map.size()[0]//2)
    plt.ylim(map.size()[1]//4-0.5, map.size()[1]//4-0.5+map.size()[1]//2)

    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)

# returns trajectories of the four robot wheels
def get_wheel_trajectories(base_locations, leg_x, leg_y, rotation_step_size):
    # wheel positions in local coordinates
    local_fl = torch.tensor([leg_x,leg_y]).float()
    local_fr = torch.tensor([leg_x,-leg_y]).float()
    local_bl = torch.tensor([-leg_x,leg_y]).float()
    local_br = torch.tensor([-leg_x,-leg_y]).float()

    leg_positions = torch.zeros(base_locations.size()[0],4,2)
    for i in range(base_locations.size()[0]):
        base = base_locations[i, 0:2].float()
        theta = base_locations[i,2].item()*rotation_step_size

        # global coordinates
        rotation = torch.tensor([[cos(-theta), sin(-theta)],[-sin(-theta), cos(-theta)]])
        leg_positions[i,0,:] = base + torch.mv(rotation,local_fl)
        leg_positions[i,1,:] = base + torch.mv(rotation,local_fr)
        leg_positions[i,2,:] = base + torch.mv(rotation,local_bl)
        leg_positions[i,3,:] = base + torch.mv(rotation,local_br)

    return leg_positions
