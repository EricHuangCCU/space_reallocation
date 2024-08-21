import numpy as np
import open3d as o3d
from pykinect2 import PyKinectRuntime, PyKinectV2
import matplotlib.pyplot as plt

# 初始化Kinect v2
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)

# 設置Kinect v2深度影像的尺寸
depth_width = 512
depth_height = 424

try:
    while True:
        if kinect.has_new_depth_frame():
            depth_frame = kinect.get_last_depth_frame()
            depth_frame = depth_frame.reshape((depth_height, depth_width))

            # 將深度影像轉換為點雲
            points = []
            for y in range(depth_frame.shape[0]):
                for x in range(depth_frame.shape[1]):
                    z = depth_frame[y, x]
                    if z == 0:  # 排除無效深度
                        continue
                    points.append([x, y, z])

            points = np.array(points)
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points)

            # 平面分割
            plane_model, inliers = point_cloud.segment_plane(distance_threshold=0.01,
                                                             ransac_n=3,
                                                             num_iterations=1000)

            # 提取平面上的點
            plane_cloud = point_cloud.select_by_index(inliers)

            # 剩餘的點
            non_plane_cloud = point_cloud.select_by_index(inliers, invert=True)

            # DBSCAN聚類
            labels = np.array(non_plane_cloud.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))

            # 獲取不同聚類的點
            max_label = labels.max()
            colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
            colors[labels < 0] = 0
            non_plane_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])

            # 顯示結果：平面和聚類點雲
            o3d.visualization.draw_geometries([plane_cloud, non_plane_cloud])

finally:
    # 關閉Kinect
    kinect.close()
