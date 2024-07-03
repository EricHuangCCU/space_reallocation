import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
from matplotlib import pyplot as plt

OBJECT_NUM = 10  # number of object
OBJECT_TYPE = 5  # number of obj type
MAX_SIZE = 20  # max size of object

# assume container has size {30, 30, 30}
CONTAINER_SIZE = 300

# define a class called object
class Object:
    def __init__(self):
        self.H = 0
        self.W = 0
        self.L = 0
        self.type = 0
        self.num = 0

    def set_value(self, h, w, l, t, num):
        self.H = h
        self.W = w
        self.L = l
        self.type = t
        self.num = num

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
    
    def show_NUM(self):
        return self.num
    
    def set_position(self, position):
        self.position = position

def box(position1, position2, color, ax):
    x1, y1, z1 = position1
    x2, y2, z2 = position2

    width = int(abs(x2 - x1))
    depth = int(abs(y2 - y1))
    height = int(abs(z2 - z1))

    x = np.arange(x1, x2 + 1)
    y = np.arange(y1, y2 + 1)
    z = np.arange(z1, z2 + 1)

    X0, Y0 = np.meshgrid(x, y)
    Z0 = np.full((depth + 1, width + 1), z1)

    X1 = X0
    Y1 = Y0
    Z1 = Z0 + height

    X2, Z2 = np.meshgrid(x, z)
    Y2 = np.full((height + 1, width + 1), y1)

    X3 = X2
    Y3 = Y2 + depth
    Z3 = Z2

    Y4, Z4 = np.meshgrid(y, z)
    X4 = np.full((height + 1, depth + 1), x1)

    X5 = X4 + width
    Y5 = Y4
    Z5 = Z4

    surfaces = [
        [X0, Y0, Z0],
        [X1, Y1, Z1],
        [X2, Y2, Z2],
        [X3, Y3, Z3],
        [X4, Y4, Z4],
        [X5, Y5, Z5]
    ]

    for X, Y, Z in surfaces:
        ax.plot_surface(X, Y, Z, color=color)

def decide_position(rec, obj, j, k, capacity):
    if (j, k) not in rec:
        rec[(j, k)] = []

    grid = np.full((capacity[0]+1, capacity[1]+1), 0)

    #先將已被佔用的空間標示為1
    for i in rec[(j, k)]:
        for l in range(i[0], i[2]+1):
            for m in range(i[1], i[3]+1):
                grid[l][m] = 1

    #找到夠放的位置就加入
    for row in range(j, obj[0] - 1, -1):
        for col in range(k, obj[1] - 1, -1):

            #橫的放或是直的放
            if all(grid[row - i][col - m] == 0 for i in range(obj[0]+1) for m in range(obj[1]+1)):
                rec[(j, k)].append((row - obj[0], col - obj[1], row, col, obj[2], obj[3]))
                return
            elif all(grid[row - m][col - i] == 0 for m in range(obj[1]+1) for i in range(obj[0]+1)):
                rec[(j, k)].append((row - obj[1], col - obj[0], row, col, obj[2], obj[3]))
                return

    return

def bin_packing(objects, capacity):
    n = len(objects)

    #用dict儲存物件位置資訊
    rec = {}

    rec[(CONTAINER_SIZE, CONTAINER_SIZE)] = [(CONTAINER_SIZE-objects[0][0], CONTAINER_SIZE-objects[0][1], CONTAINER_SIZE, CONTAINER_SIZE, objects[0][2], objects[0][3])]

    for i in range(1, n):
        decide_position(rec, objects[i], CONTAINER_SIZE, CONTAINER_SIZE, capacity)

    solution = []

    j = capacity[0]
    k = capacity[1]

    for obj in rec[(j, k)]:
        solution.append(obj)

    return solution

def visualize_bin_packing(solution):
    color = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'cyan', 'gray']

    

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for rec in solution:
        #num = random.randint(0, 9)
        box([rec[0], rec[1],0], [rec[2], rec[3], rec[4]], color[int(rec[5])], ax)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()

def object_detect(modelName, image_path):
    # 初始化Mediapipe Objectron模型
    mp_drawing = mp.solutions.drawing_utils
    mp_objectron = mp.solutions.objectron

    # 選擇要檢測的物體類型
    objectron = mp_objectron.Objectron(static_image_mode=True,
                                    max_num_objects = OBJECT_NUM,
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5,
                                    model_name = modelName)  # 'Shoe', 'Chair', 'Cup', or 'Camera'
    
    image = cv2.imread(image_path)

    # 檢查是否成功讀取圖片
    if image is None:
        raise FileNotFoundError(f"Image at {image_path} not found.")
    
    # 將BGR圖像轉換為RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 檢測物體並估計3D姿態
    results = objectron.process(image_rgb)
    
    obj = []
    # 繪製檢測到的物體及其3D姿態
    if results.detected_objects:
        for i, detected_object in enumerate(results.detected_objects):
            # 繪製3D邊界框
            mp_drawing.draw_landmarks(
                image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
            mp_drawing.draw_axis(image, detected_object.rotation, detected_object.translation)

            # 計算長、寬、高
            landmarks_3d = detected_object.landmarks_3d.landmark
            x_coords = [landmark.x for landmark in landmarks_3d]
            y_coords = [landmark.y for landmark in landmarks_3d]
            z_coords = [landmark.z for landmark in landmarks_3d]

            new_obj = Object()
            l = int((max(x_coords) - min(x_coords) ) * 100)
            w = int((max(y_coords) - min(y_coords)) * 100)
            h = int((max(z_coords) - min(z_coords)) * 100)
            t = 0

            print(f"object {i}: L:{l} W:{w} H:{h}")

            
            merge = False
            for ob in obj:
                #如果有(長、寬)、(寬、高)、(長、高)任一種都相同或較小，就把兩樣物件做合併
                if(h<=ob.show_H() and w<=ob.show_W() and t==ob.show_type()):
                    new_obj.set_value(h, w, l+ob.show_L(), OBJECT_TYPE - 1, ob.show_NUM)
                    merge = True
                elif(l<=ob.show_L() and w<=ob.show_W() and t==ob.show_type()):
                    new_obj.set_value(h+ob.show_H(), w, l, OBJECT_TYPE - 1, ob.show_NUM)
                    merge = True
                elif(l<=ob.show_L() and h<=ob.show_H() and t==ob.show_type()):
                    new_obj.set_value(h, w+ob.show_H(), l, OBJECT_TYPE - 1, ob.show_NUM)
                    merge = True

                if merge == True :
                    obj.remove(ob)
                    break
            
            if merge == False:
                new_obj.set_value(h, w, l, t, i)

            obj.append(new_obj)


        return obj




if __name__ == "__main__":
    random.seed()
    obj = []

    modelName = 'Chair'
    image_path = "Chairs.jpg"

    obj = object_detect(modelName, image_path)


    #將物件照著type做排序
    obj.sort(key=lambda x: x.show_type())

    index_end = 0
    for i in range(OBJECT_TYPE):
        index_start = index_end

        for j in range(len(obj)):
            if obj[j].show_type() == i:
                index_end += 1

        #將各類型的物件照著體積大小排序
        #obj[index_start:index_end] = sorted(obj[index_start:index_end], key=lambda x:x.show_volume(), reverse=True)

        #將各類型的物件照著高做排序
        obj[index_start:index_end] = sorted(obj[index_start:index_end], key=lambda x:x.show_H(), reverse=True)

        index_start = index_end

    for i in range(len(obj)):
        print(f"object {obj[i].show_NUM()} : H:{obj[i].show_H()}, W:{obj[i].show_W()}, L:{obj[i].show_L()}, tpye:{obj[i].show_type()}" )

    objects = []

    for i in obj:
      objects.append([i.show_W(), i.show_L(), i.show_H(), i.show_type()])

    capacity = (CONTAINER_SIZE, CONTAINER_SIZE)

    solution = bin_packing(objects, capacity)
    for x in solution:
        print(x)
    #visualize_bin_packing_V2(solution, capacity)
    visualize_bin_packing(solution)

