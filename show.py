from examples.test_pr2_motion import SLEEP
import pybullet as p
import pybullet_data
import time
import numpy as np
import random
import pandas as pd

from pybullet_tools.pr2_utils import PR2_URDF, DRAKE_PR2_URDF,PR2_GROUPS, get_disabled_collisions
from pybullet_tools.utils import Euler, GREY, LockRenderer, Point, connect, disconnect, joint_from_name, joints_from_names, \
    load_pybullet, pairwise_collisions, plan_joint_motion, quat_from_euler, set_joint_positions, set_point, set_quat, wait_if_gui, load_model, create_box


def generate_obstacles(num_obstacles, walls):
    obstacles = walls
    idx = 0
    pos_list = [(-2.5, 0, 0), (-0.8, 0, 0), (0.8, -0.4, 0), (1.6, -1.7, 0)]
    ori_list = [0, 0, np.pi/2, np.pi/2]
    while True:
        table = load_pybullet("models/table_collision/table.urdf", fixed_base=True)
        set_quat(table, quat_from_euler(Euler(yaw=ori_list[idx])))
        set_point(table, pos_list[idx])
        
        collision = pairwise_collisions(table, obstacles)
        if not collision:
            obstacles.append(table)
            idx += 1
        else:
            p.removeBody(table)
        
        if len(obstacles) == num_obstacles+4:
            return obstacles


def planning(pr2, base_start, base_goal, kappa, phi, obstacles=[]):
    disabled_collisions = get_disabled_collisions(pr2)
    base_joints = [joint_from_name(pr2, name) for name in PR2_GROUPS['base']]
    set_joint_positions(pr2, base_joints, base_start)
    base_joints = base_joints[:2]
    base_goal = base_goal[:len(base_joints)]
    with LockRenderer(lock=False):
        base_path, state = plan_joint_motion(pr2, base_joints, base_goal, kappa, phi, obstacles=obstacles,
                                      disabled_collisions=disabled_collisions)

    if base_path is None:
        print('Unable to find a base path')
        if state is not None:
            return state
        return

    for bq in base_path:
        set_joint_positions(pr2, base_joints, bq)
        kappa.append(bq)
        phi.append(0)
    
    if state is not None:
        return state

        
def main(index, num_obstacles, use_pr2_drake=True):
    connect(use_gui=True)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane = p.loadURDF("plane.urdf")

    """
    environment setting (wall + obs)
    """
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

    # num_obstacles = random.randint(1, 5)
    obstacles = generate_obstacles(num_obstacles=num_obstacles, walls=walls)
    obstacles.append(plane)

    """
    PR2 setting
    """
    pr2_urdf = DRAKE_PR2_URDF if use_pr2_drake else PR2_URDF
    pr2 = load_model(pr2_urdf, fixed_base=True)
    left_joints = joints_from_names(pr2, PR2_GROUPS['left_arm'])
    right_joints = joints_from_names(pr2, PR2_GROUPS['right_arm'])
    set_joint_positions(pr2, left_joints, [np.pi/2, -np.pi/3, 0, 0, 0, 0, 0])
    set_joint_positions(pr2, right_joints, [-np.pi/2, -np.pi/3, 0, 0, 0, 0, 0])

    """
    Planning
    """
    kappa = []
    phi = []
    states = []
    epoch = 0
    wait_if_gui('start?')
    while epoch < 1:
        base_start = (2.5, 2.5, 0)
        base_goal = (-2.5, -2.5, 0)        
        state = planning(pr2, base_start, base_goal, kappa, phi, obstacles=obstacles)
        
        epoch += 1



if __name__== '__main__':
    
    print('\n')
    print('#'*30)
    print('Num obstacles : {}'.format(4))
    print('#'*30)
    print('\n')
    main(1, 3)
    wait_if_gui('Finish')
    disconnect()
    