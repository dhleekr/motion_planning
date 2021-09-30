"""

Path planning Sample Code with Randomized Rapidly-Exploring Random Trees (RRT)

author: AtsushiSakai(@Atsushi_twi)

"""

import math
import random
import os

import matplotlib.pyplot as plt
import numpy as np
import cv2

show_animation = False


class RRT:
    """
    Class for RRT planning
    """

    class Node:
        """
        RRT Node
        """

        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

    def __init__(self,
                 rand_area,
                 expand_dis=3,
                 path_resolution=0.5,
                 goal_sample_rate=5,
                 max_iter=1000):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]

        """
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.node_list = []

        obstacle_list = self.generate_obstacles()
        self.obstacle_list = obstacle_list
        self.start, self.end, self.img = self.generate_points(self.obstacle_list)
        
        

    def planning(self, animation=False, idx=0, create_dataset=False):
        """
        rrt path planning

        animation: flag for animation on or off
        """
        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_collision(new_node, self.obstacle_list):
                self.node_list.append(new_node)

            if animation and i % 5 == 0:
                # self.draw_graph(rnd_node)
                pass

            if self.calc_dist_to_goal(self.node_list[-1].x,
                                      self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end,
                                        self.expand_dis)
                if self.check_collision(final_node, self.obstacle_list):
                    print('Iterator : {}'.format(i+1))
                    if create_dataset:
                        directory = './dataset/{}/'.format(i+1)
                        try:
                            if not os.path.exists(directory):
                                os.makedirs(directory)
                        except OSError:
                            print ('Error: Creating directory. ' +  directory)

                        cv2.imwrite('./dataset/{}/{}.jpg'.format(i+1,idx), self.img)
                    return self.generate_final_course(len(self.node_list) - 1)
                    # return True
                    
            if i == self.max_iter-1:
                print('#################################################################')
                print('Iterator : {}'.format(i+1))
                print('#################################################################')
                if create_dataset:
                    directory = './dataset/{}/'.format(i+1)
                    try:
                        if not os.path.exists(directory):
                            os.makedirs(directory)
                    except OSError:
                        print ('Error: Creating directory. ' +  directory)
                    cv2.imwrite('./dataset/{}/{}.jpg'.format(i+1,idx), self.img)
                
        return None  # cannot find path

    def steer(self, from_node, to_node, extend_length=float("inf")):

        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.x = to_node.x
            new_node.y = to_node.y

        new_node.parent = from_node

        return new_node
    
    def generate_obstacles(self):
        obstacle_list = []
        obstacle_num = random.randint(1, 4)
        for i in range(obstacle_num):
            obstacle_list.append([random.uniform(0, 20), random.uniform(0, 20), random.uniform(1, 7)])

        return obstacle_list

    def generate_points(self, obstacle_list):
        start = self.Node(random.uniform(0, 20), random.uniform(0, 20))

        while not self.generate_points_check(start, obstacle_list): # collision
            start = self.Node(random.uniform(0, 20), random.uniform(0, 20))
        
        goal = self.Node(random.uniform(0, 20), random.uniform(0, 20))

        while not self.generate_points_check(goal, obstacle_list): # collision
            goal = self.Node(random.uniform(0, 20), random.uniform(0, 20))

        # Image 저장
        img = np.full((500, 500, 3), 255, np.uint8)
        for ox, oy, size in obstacle_list:
            ox = int(25*ox)
            oy = int(25*oy)
            size = int(25*size)
            # print(ox, oy, size, type(ox))
            img = cv2.circle(img, (ox, oy), size, (0, 0, 0), -1)

        img = cv2.circle(img, (int(25*start.x), int(25*start.y)), 8, (255,0,0), -1)
        img = cv2.circle(img, (int(25*goal.x), int(25*goal.y)), 8, (255,0,0), -1)
        
        return start, goal, img
    
    def generate_points_check(self, node, obstacle_list):
        for ox, oy, size in obstacle_list:
            d = math.sqrt((ox - node.x)**2 + (oy - node.y)**2)
            if d <= size:
                return False # 충돌 생김

        return True  # safe

    def generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        return path

    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand))
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y)
        return rnd

    def draw_graph(self, rnd=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        for (ox, oy, size) in self.obstacle_list:
            self.plot_circle(ox, oy, size)

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis("equal")
        plt.axis([0, 20, 0, 20])
        plt.grid(True)
        plt.pause(0.01)

    @staticmethod
    def plot_circle(x, y, size, color="-b"):  # pragma: no cover
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)

    @staticmethod
    def obstacles_space(x, y, size):
        img = np.full((20, 20), 255, dtype=np.int32)
        img = cv2.circle(img, (x, y), size, (255, 0, 0), -1)

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2
                 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    @staticmethod
    def check_collision(node, obstacleList):

        if node is None:
            return False

        for (ox, oy, size) in obstacleList:
            dx_list = [ox - x for x in node.path_x]
            dy_list = [oy - y for y in node.path_y]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]

            if min(d_list) <= size**2:
                return False  # collision

        return True  # safe

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta


def main(idx):
    print("start " + __file__)

    # Set Initial parameters
    rrt = RRT(rand_area=[0, 20])
    path = rrt.planning(animation=show_animation, idx=idx, create_dataset=True)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

        # Draw final path
        if show_animation:
            rrt.draw_graph()
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
            plt.grid(True)
            plt.pause(0.01)  # Need for Mac
            plt.show()


if __name__ == '__main__':
    for i in range(1000):
        main(i)