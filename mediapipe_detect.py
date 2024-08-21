import cv2
import mediapipe as mp

# 初始化 MediaPipe 物件偵測模型
mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils

# 設置攝像頭
cap = cv2.VideoCapture(0)

with mp_objectron.Objectron(static_image_mode=False,
                            max_num_objects=5,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5,
                            model_name='Chair') as objectron:  # 可以改成 'Cup', 'Chair', 'Camera', etc.
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # 轉換顏色格式
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 偵測物體
        results = objectron.process(image)

        # 將顏色轉回 BGR 以便顯示
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 如果檢測到物體，繪製結果
        if results.detected_objects:
            for detected_object in results.detected_objects:
                mp_drawing.draw_landmarks(image, 
                                          detected_object.landmarks_2d, 
                                          mp_objectron.BOX_CONNECTIONS)
                mp_drawing.draw_axis(image, 
                                     detected_object.rotation, 
                                     detected_object.translation)
        
        # 顯示結果
        cv2.imshow('MediaPipe Objectron', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
