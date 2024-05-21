import random
import numpy as np
import matplotlib.pyplot as plt

OBJECT_NUM = 10  # number of object
OBJECT_TYPE = 5  # number of obj type
MAX_SIZE = 20  # max size of object

# assume container has size {30, 30, 30}
CONTAINER_SIZE = 50

# define a class called object
class Object:
    def __init__(self):
        self.H = 0
        self.W = 0
        self.L = 0
        self.type = 0

    def set_value(self, h, w, l, t):
        self.H = h
        self.W = w
        self.L = l
        self.type = t

    def show_volume(self):
        return self.H * self.W * self.L

    def show_type(self):
        return self.type

    def show_H(self):
        return self.H

    def show_W(self):
        return self.W

    def show_L(self):
        return self.L

    def set_position(self, position):
        self.position = position


def decide_position(dp, obj, j, k, capacity):
    if (j, k) not in dp:
        dp[(j, k)] = []

    grid = np.full((capacity[0]+1, capacity[1]+1), 0)

    for i in dp[(j, k)]:
        for l in range(i[0], i[2]+1):
            for m in range(i[1], i[3]+1):
                grid[l][m] = 1

    for row in range(j, obj[0] - 1, -1):
        for col in range(k, obj[1] - 1, -1):
            if all(grid[row - i][col - m] == 0 for i in range(obj[0]+1) for m in range(obj[1]+1)):
                dp[(j, k)].append((row - obj[0], col - obj[1], row, col, obj[2]))
                return
            elif all(grid[row - m][col - i] == 0 for m in range(obj[1]+1) for i in range(obj[0]+1)):
                dp[(j, k)].append((row - obj[1], col - obj[0], row, col, obj[2]))
                return

    return

def bin_packing_2D(objects, capacity):
    n = len(objects)
    dp = {}

    dp[(CONTAINER_SIZE, CONTAINER_SIZE)] = [(CONTAINER_SIZE-objects[0][0], CONTAINER_SIZE-objects[0][1], CONTAINER_SIZE, CONTAINER_SIZE, objects[0][2])]

    for i in range(1, n):
        decide_position(dp, objects[i], CONTAINER_SIZE, CONTAINER_SIZE, capacity)

    solution = []

    j = capacity[0]
    k = capacity[1]

    for obj in dp[(j, k)]:
        solution.append(obj)

    return solution

def visualize_bin_packing_V2(solution, capacity):
    plt.figure()
    plt.fill([0, capacity[0], capacity[0], 0], [0, 0, capacity[1], capacity[1]], 'lightgray')

    color = plt.cm.rainbow(np.linspace(0, 1, len(solution)))  
    color_index = 0  

    for obj in solution:
        plt.fill([obj[0], obj[2], obj[2], obj[0]], [obj[1], obj[1], obj[3], obj[3]], color=color[color_index])
        plt.plot([obj[0], obj[2], obj[2], obj[0], obj[0]], [obj[1], obj[1], obj[3], obj[3], obj[1]], 'black')
        color_index = (color_index + 1) % len(solution)  

        obj_center_x = (obj[0] + obj[2]) / 2
        obj_center_y = (obj[1] + obj[3]) / 2
        plt.text(obj_center_x, obj_center_y, str(obj[4]), ha='center', va='center', color='black')

    plt.xlim(0, capacity[0])
    plt.ylim(0, capacity[1])
    plt.gca().set_aspect('equal', adjustable='box')

    plt.show()

if __name__ == "__main__":
    random.seed()
    obj = []

    for _ in range(OBJECT_NUM):
        new_obj = Object()
        h = random.randint(1, MAX_SIZE)
        w = random.randint(1, MAX_SIZE)
        l = random.randint(1, MAX_SIZE)
        t = random.randint(0, OBJECT_TYPE - 1)
        new_obj.set_value(h, w, l, t)
        obj.append(new_obj)

    obj.sort(key=lambda x: x.show_type())

    index_end = 0
    for i in range(OBJECT_TYPE):
        index_start = index_end

        for j in range(OBJECT_NUM):
            if obj[j].show_type() == i:
                index_end += 1

        obj[index_start:index_end] = sorted(obj[index_start:index_end], key=lambda x:x.show_volume(), reverse=True)

        index_start = index_end

    for i in range(OBJECT_NUM):
        print(f"object {i} : H:{obj[i].show_H()}, W:{obj[i].show_W()}, L:{obj[i].show_L()}, tpye:{obj[i].show_type()}" )

    objects = []

    for i in obj:
      objects.append([i.show_W(), i.show_L(), i.show_type()])

    capacity = (CONTAINER_SIZE, CONTAINER_SIZE)

    solution = bin_packing_2D(objects, capacity)
    for x in solution:
        print(x)
    visualize_bin_packing_V2(solution, capacity)

