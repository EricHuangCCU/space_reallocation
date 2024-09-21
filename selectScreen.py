from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

CONTAINER_SIZE = 100
OBJECT_TYPE = 5

# 初始化 MediaPipe 物件偵測模型
mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils

# 設置攝像頭
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# 初始化模型，只需一次
objectron_models = {
    'shoe': mp_objectron.Objectron(static_image_mode=False, max_num_objects=5, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_name='Shoe'),
    'cup': mp_objectron.Objectron(static_image_mode=False, max_num_objects=5, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_name='Cup'),
    'chair': mp_objectron.Objectron(static_image_mode=False, max_num_objects=5, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_name='Chair'),
    'camera': mp_objectron.Objectron(static_image_mode=False, max_num_objects=5, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_name='Camera')
}


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



class Ui_Form(object):

    def __init__(self):
        self.type = None
        self.objects = []

    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1240, 765)
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(18)
        Form.setFont(font)

        self.horizontalLayoutWidget = QtWidgets.QWidget(Form)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 1220, 660))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")

        self.cam = QtWidgets.QGraphicsView(self.horizontalLayoutWidget)
        self.cam.setObjectName("cam")
        self.horizontalLayout.addWidget(self.cam)
        self.scene = QtWidgets.QGraphicsScene()
        self.cam.setScene(self.scene)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.result = QtWidgets.QGraphicsView(self.horizontalLayoutWidget)
        self.result.setObjectName("result")
        self.horizontalLayout.addWidget(self.result)

        self.L_showCurType = QtWidgets.QLabel(Form)
        self.L_showCurType.setGeometry(QtCore.QRect(110, 690, 220, 60))
        self.L_showCurType.setObjectName("L_showCurType")

        self.CB_selType = QtWidgets.QComboBox(Form)
        self.CB_selType.setGeometry(QtCore.QRect(300, 700, 140, 40))
        self.CB_selType.setObjectName("CB_selType")

        self.BUT_ok = QtWidgets.QPushButton(Form)
        self.BUT_ok.setGeometry(QtCore.QRect(478, 700, 140, 40))
        self.BUT_ok.setObjectName("BUT_exit")

        self.BUT_exit = QtWidgets.QPushButton(Form)
        self.BUT_exit.setGeometry(QtCore.QRect(1040, 700, 140, 40))
        self.BUT_exit.setObjectName("BUT_exit")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)


    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))

        self.CB_selType.addItem("shoe")
        self.CB_selType.addItem("cup")
        self.CB_selType.addItem("chair")
        self.CB_selType.addItem("camera")
        self.CB_selType.currentIndexChanged.connect(self.change_type)

        self.L_showCurType.setText(_translate("Form", f"current type: {self.CB_selType.currentText()}"))

        self.BUT_ok.setText(_translate("Form", "OK"))
        self.BUT_ok.clicked.connect(self.obj_check) 

        self.BUT_exit.setText(_translate("Form", "exit"))
        self.BUT_exit.clicked.connect(QtWidgets.qApp.quit)  # 正確退出應用


    def box(self, position1, position2, color, ax):
        x1, y1, z1 = position1
        x2, y2, z2 = position2

        width = abs(x2 - x1)
        depth = abs(y2 - y1)
        height = abs(z2 - z1)

        x = np.linspace(x1, x2, width + 1)
        y = np.linspace(y1, y2, depth + 1)
        z = np.linspace(z1, z2, height + 1)

        # Create the surfaces by ensuring correct meshgrid alignment
        X0, Y0 = np.meshgrid(x, y)  # Front and back surfaces
        Z0 = np.full_like(X0, z1)
        Z1 = np.full_like(X0, z2)

        X2, Z2 = np.meshgrid(x, z)  # Left and right surfaces
        Y2 = np.full_like(X2, y1)
        Y3 = np.full_like(X2, y2)

        Y4, Z4 = np.meshgrid(y, z)  # Top and bottom surfaces
        X4 = np.full_like(Y4, x1)
        X5 = np.full_like(Y4, x2)

        surfaces = [
            (X0, Y0, Z0),  # Front
            (X0, Y0, Z1),  # Back
            (X2, Y2, Z2),  # Left
            (X2, Y3, Z2),  # Right
            (X4, Y4, Z4),  # Top
            (X5, Y4, Z4)   # Bottom
        ]

        # Plot each surface ensuring that shapes match
        for X, Y, Z in surfaces:
            ax.plot_surface(X, Y, Z, color=color)


    def visualize_bin_packing_3D(self, solution):
        color = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'cyan', 'gray']

        

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for rec in solution:
            #num = random.randint(0, 9)
            self.box([rec[0], rec[1],0], [rec[2], rec[3], rec[4]], color[int(rec[5])], ax)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        # plt.show()

        # canvas = FigureCanvas(fig)
        # scene = QtWidgets.QGraphicsScene()
        # scene.addWidget(canvas)
        # QGraphicsView.setScene(scene)

        # 將 Matplotlib 圖像嵌入到 QGraphicsView 中
        canvas = FigureCanvas(fig)  # 使用 FigureCanvas
        canvas.draw()  # 畫出圖像

        # 創建一個場景
        scene = QtWidgets.QGraphicsScene()
        
        # 將 Matplotlib 畫布嵌入到場景
        scene.addWidget(canvas)

        # 設置到 result QGraphicsView 上顯示
        self.result.setScene(scene)

        # 自動調整以適應視窗大小
        self.result.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)


    def decide_position(self, rec, obj, j, k, capacity):
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

    def bin_packing_3D(self, objects, capacity):
        n = len(objects)

        #用dict儲存物件位置資訊
        rec = {}

        rec[(CONTAINER_SIZE, CONTAINER_SIZE)] = [(CONTAINER_SIZE-objects[0][0], CONTAINER_SIZE-objects[0][1], CONTAINER_SIZE, CONTAINER_SIZE, objects[0][2], objects[0][3])]

        for i in range(1, n):
            self.decide_position(rec, objects[i], CONTAINER_SIZE, CONTAINER_SIZE, capacity)

        solution = []

        j = capacity[0]
        k = capacity[1]

        for obj in rec[(j, k)]:
            solution.append(obj)

        return solution


    
    def show_result(self):
        #將物件照著type做排序
        self.objects.sort(key=lambda x: x.show_type())

        index_end = 0
        for i in range(OBJECT_TYPE):
            index_start = index_end

            for j in range(len(self.objects)):
                if self.objects[j].show_type() == i:
                    index_end += 1

            #將各類型的物件照著高做排序
            self.objects[index_start:index_end] = sorted(self.objects[index_start:index_end], key=lambda x:x.show_H(), reverse=True)

            index_start = index_end

        for i in range(len(self.objects)):
            print(f"object {i} : H:{self.objects[i].show_H()}, W:{self.objects[i].show_W()}, L:{self.objects[i].show_L()}, tpye:{self.objects[i].show_type()}" )


        Objects = []

        for i in self.objects:
            Objects.append([i.show_W(), i.show_L(), i.show_H(), i.show_type()])

        capacity = (CONTAINER_SIZE, CONTAINER_SIZE)

        solution = self.bin_packing_3D(Objects, capacity)
        for x in solution:
            print(x)
        #visualize_bin_packing_V2(solution, capacity)
        # visualize_bin_packing_3D(solution)
        self.visualize_bin_packing_3D(solution)
            

    def obj_check(self):
        # 獲取畫面上掃描到的物件長寬高並輸出
        ret, frame = cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if hasattr(self, 'current_objectron'):
                results = self.current_objectron.process(rgb_frame)
                if results.detected_objects:
                    for detected_object in results.detected_objects:
                        # 計算長、寬、高
                        landmarks_3d = detected_object.landmarks_3d.landmark
                        x_coords = [landmark.x for landmark in landmarks_3d]
                        y_coords = [landmark.y for landmark in landmarks_3d]
                        z_coords = [landmark.z for landmark in landmarks_3d]

                        l = round(max(x_coords) - min(x_coords)*100)
                        w = round(max(y_coords) - min(y_coords)*100)
                        h = round(max(z_coords) - min(z_coords)*100)

                        print(f"l: {l} w: {w}  h: {h}")


                        # 將物件加入List
                        if self.type == "shoe":
                            t = 0
                        elif self.type == "cup":
                            t = 1
                        elif self.type == "chair":
                            t = 2
                        elif self.type == "camera":
                            t = 3

                        new_obj = Object()

                        merge = False
                        for ob in self.objects:
                            #如果有(長、寬)、(寬、高)、(長、高)任一種都相同或較小，就把兩樣物件做合併
                            if(h<=ob.show_H() and w<=ob.show_W() and t==ob.show_type()):
                                new_obj.set_value(h, w, l+ob.show_L(), OBJECT_TYPE - 1)
                                merge = True
                            elif(l<=ob.show_L() and w<=ob.show_W() and t==ob.show_type()):
                                new_obj.set_value(h+ob.show_H(), w, l, OBJECT_TYPE - 1)
                                merge = True
                            elif(l<=ob.show_L() and h<=ob.show_H() and t==ob.show_type()):
                                new_obj.set_value(h, w+ob.show_H(), l, OBJECT_TYPE - 1)
                                merge = True

                            if merge == True :
                                self.objects.remove(ob)
                                break
                        
                        if merge == False:
                            new_obj.set_value(h, w, l, t)

                        self.objects.append(new_obj)

                        self.show_result()

                        # print("add obj success")
                   
                else:
                    print("No object detected")
            else:
                print("Objectron model not initialized")
        else:
            print("Failed to capture frame")

    # 更新相機畫面
    def update_frame(self):
        ret, frame = cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if hasattr(self, 'current_objectron'):
                results = self.current_objectron.process(rgb_frame)
                if results.detected_objects:
                    for detected_object in results.detected_objects:
                        mp_drawing.draw_landmarks(rgb_frame, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                        mp_drawing.draw_axis(rgb_frame, detected_object.rotation, detected_object.translation)

            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QtGui.QImage(rgb_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qt_image)
            self.scene.clear()
            self.scene.addPixmap(pixmap)
            self.cam.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    # 更換type
    def change_type(self):
        self.type = self.CB_selType.currentText()

        print(self.type)
        self.L_showCurType.setText(f"current type: {self.type}")
        self.scene.clear()
        self.current_objectron = objectron_models[self.type]


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
