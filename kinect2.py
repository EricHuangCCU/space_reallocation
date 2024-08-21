import cv2
import numpy as np
from pykinect2 import PyKinectV2, PyKinectRuntime

# 初始化Kinect
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color)

# 設置深度影像和RGB影像的解析度
depth_width = kinect.depth_frame_desc.Width
depth_height = kinect.depth_frame_desc.Height
color_width = kinect.color_frame_desc.Width
color_height = kinect.color_frame_desc.Height

# 深度閾值（根據實際情况调整）
MIN_DEPTH = 500  # 最小深度（單位：毫米）
MAX_DEPTH = 4500 # 最大深度（單位：毫米）

# 輪廓面積閾值（用来去除小的noise）
MIN_CONTOUR_AREA = 500  # 最小輪廓面積（像素）

while True:
    # 讀取RGB影像
    if kinect.has_new_color_frame():
        color_frame = kinect.get_last_color_frame()
        color_frame = color_frame.reshape((color_height, color_width, 4))
        color_frame = color_frame[:, :, :3]  # 去除alpha通道 

        # 讀取深度影像
        if kinect.has_new_depth_frame():
            depth_frame = kinect.get_last_depth_frame()
            depth_frame = depth_frame.reshape((depth_height, depth_width))
            
            # 去除無效深度值
            depth_frame = np.where((depth_frame >= MIN_DEPTH) & (depth_frame <= MAX_DEPTH), depth_frame, 0)

            # 深度影像轉換為 8 bit影像
            depth_frame_8bit = np.uint8(depth_frame.clip(0, 4500) / 4500 * 255)

            # 高斯模糊减少noise
            blurred = cv2.GaussianBlur(depth_frame_8bit, (5, 5), 0)

            # 使用Canny邊緣檢測算法
            edges = cv2.Canny(blurred, 50, 150)

            # 查找輪廓
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 遍歷每個輪廓並產生立方體
            for contour in contours:
                if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # 獲取輪廓的深度訊息
                    depth_region = depth_frame[y:y+h, x:x+w]
                    avg_depth = np.mean(depth_region)
                    
                    
                    depth_scale = 0.5 
                    depth_in_meters = avg_depth * 0.001  # 深度轉換為meter
                    box_thickness = int(depth_in_meters * depth_scale)
                    
                    cv2.rectangle(depth_frame_8bit, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    
                    cv2.line(depth_frame_8bit, (x, y), (x + box_thickness, y), (0, 255, 0), 2)
                    cv2.line(depth_frame_8bit, (x, y), (x, y + box_thickness), (0, 255, 0), 2)
                    cv2.line(depth_frame_8bit, (x + w, y), (x + w + box_thickness, y), (0, 255, 0), 2)
                    cv2.line(depth_frame_8bit, (x + w, y), (x + w, y + box_thickness), (0, 255, 0), 2)
                    cv2.line(depth_frame_8bit, (x, y + h), (x + box_thickness, y + h), (0, 255, 0), 2)
                    cv2.line(depth_frame_8bit, (x, y + h), (x, y + h + box_thickness), (0, 255, 0), 2)
                    cv2.line(depth_frame_8bit, (x + w, y + h), (x + w + box_thickness, y + h), (0, 255, 0), 2)
                    cv2.line(depth_frame_8bit, (x + w, y + h), (x + w, y + h + box_thickness), (0, 255, 0), 2)

            cv2.imshow('RGB Image with 3D Cuboids',depth_frame_8bit)

        # 按下 'q' 鍵退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

kinect.close()
cv2.destroyAllWindows()
