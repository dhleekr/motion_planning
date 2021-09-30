from examples.doo import generate_obstacles
from pybullet_tools.utils import GREY, Point, connect, create_box, disconnect, joints_from_names, joint_from_name, load_model, set_joint_positions, set_point, wait_if_gui
import pybullet as p
import pybullet_data
import random
import numpy as np
from pybullet_tools.pr2_utils import PR2_URDF, DRAKE_PR2_URDF,PR2_GROUPS
import os
import pandas as pd

connect(use_gui=True)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane = p.loadURDF("plane.urdf")

floor_extent = 7
wall_side = 0.1
wall1 = create_box(floor_extent + wall_side, wall_side, 1.2, color=GREY)
set_point(wall1, Point(y=floor_extent/2., z=wall_side/2.))
wall2 = create_box(floor_extent + wall_side, wall_side, 1.2, color=GREY)
set_point(wall2, Point(y=-floor_extent/2., z=wall_side/2.))
wall3 = create_box(wall_side, floor_extent + wall_side, 1.2, color=GREY)
set_point(wall3, Point(x=floor_extent/2., z=wall_side/2.))
wall4 = create_box(wall_side, floor_extent + wall_side, 1.2, color=GREY)
set_point(wall4, Point(x=-floor_extent/2., z=wall_side/2.))
walls = [wall1, wall2, wall3, wall4]

# path = "./savebullet/0817/"
# filedir = os.listdir(path)
# file = "0_2_3(epoch=7000).bullet"
# print('#'*25)
# print("File : ",file)
# print('#'*25)

# filea = file.split('_')
# idx = int(filea[0])
# num_obstacles = int(filea[1].split('.')[0])
obstacles = generate_obstacles(num_obstacles=5, walls=walls)
obstacles.append(plane)

pr2_urdf = DRAKE_PR2_URDF
pr2 = load_model(pr2_urdf, fixed_base=True)
base_joints = [joint_from_name(pr2, name) for name in PR2_GROUPS['base']]
left_joints = joints_from_names(pr2, PR2_GROUPS['left_arm'])
right_joints = joints_from_names(pr2, PR2_GROUPS['right_arm'])
set_joint_positions(pr2, left_joints, [np.pi/2, -np.pi/3, 0, 0, 0, 0, 0])
set_joint_positions(pr2, right_joints, [-np.pi/2, -np.pi/3, 0, 0, 0, 0, 0])

kappa = np.array(pd.read_csv('./dataset/kkkk.csv', header=None))

p.restoreState(fileName="./savebullet/0817/0_3_5(0817).bullet")

for i in  [18671, 11207, 11844,  5153, 14168, 19616, 12516,  8834, 16060,  3492]:
    x, y = kappa[i][:2]
    set_joint_positions(pr2, base_joints, (x, y, 0))
    wait_if_gui('finish?')

disconnect()