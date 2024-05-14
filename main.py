import bdb
import os
import platform

if platform.system() == "Linux":
    os.environ["WEBOTS_HOME"] = "/usr/local/webots"


from controller import Robot, Supervisor, GPS, Compass, Lidar, LidarPoint

from utils import warp_robot, move_robot_to, bresenham
from transformations import get_translation, create_tf_matrix
from plane import Plane

from tqdm import tqdm
import numpy as np
import math
import pdb

import open3d as o3d
#import pyransac3d as pyrsc

from sklearn.cluster import DBSCAN, KMeans


def main() -> None:
    supervisor: Supervisor = Supervisor()

    timestep: int = int(supervisor.getBasicTimeStep())  # in ms

    lidar: Lidar = supervisor.getDevice('lidar')
    lidar.enable(timestep)
    lidar.enablePointCloud()


    gps: GPS = supervisor.getDevice('gps')
    gps.enable(timestep)
    supervisor.step()
    gps_readings: [float] = gps.getValues()
    robot_position: (float, float) = (gps_readings[0], gps_readings[1])

    compass: Compass = supervisor.getDevice('compass')
    compass.enable(timestep)
    compass_readings: [float] = compass.getValues()
    robot_orientation: float = math.atan2(compass_readings[0], compass_readings[1])

    warp_robot(supervisor, "EPUCK", robot_position)

    data = []



    #x = np.repeat(np.arange(0.5, 2, 0.5), 7)
    #y = np.arange(0.2, 1.6, 0.2)
    #full_y = np.concatenate((y, y[::-1], y))

    #x = np.arange(0, 1, 0.2)
    #pdb.set_trace()
    x = np.repeat(np.arange(0, 2, 0.1),  10)
    y = np.arange(0, 2, 0.1)
    full_y = np.concatenate((y, y[::-1], y, y[::-1], y, y[::-1], y, y[::-1], y, y[::-1]))

    moves = [(x[i], full_y[i]) for i in range(len(x))]
    print(f"number of moves: {len(moves)}")

    #pdb.set_trace()
    read = False
    filename = "point_clouds/square_2.npy"
    if read:
        #while len(moves) != 0:
        for new_position in tqdm(moves):
            supervisor.step()
            step_distance: float = 1
            #new_position: (float, float) = (robot_position[0] + step_distance,
            #                                robot_position[1] + step_distance)
            #new_position = moves.pop(0)
            print(new_position)
            move_robot_to(supervisor, robot_position, robot_orientation, new_position, 0.1, math.pi)

            gps_readings = gps.getValues()
            robot_position = (gps_readings[0], gps_readings[1])
            compass_readings = compass.getValues()
            robot_orientation = math.atan2(compass_readings[0], compass_readings[1])

            pcd = lidar.getPointCloud()

            #data_tmp = np.array([[point.x, point.y, 0] for point in pcd if math.isfinite(point.x) and math.isfinite(point.y)])


            robot_tf: np.ndarray = create_tf_matrix((robot_position[0], robot_position[1], 0.0), robot_orientation)
            data_tmp = np.array([[point.x, point.y, 0, 1] for point in pcd if math.isfinite(point.x) and math.isfinite(point.y) and math.isfinite(point.z)])
            z_tmp = np.array([point.z for point in pcd if math.isfinite(point.x) and math.isfinite(point.y) and math.isfinite(point.z)])
            data_tmp = data_tmp.T

            #robot_tf[:2,:2] = 0
            #robot_tf[0, 0] = 1
            #robot_tf[1, 1] = 1

            tf_data = robot_tf @ data_tmp
            tf_data = tf_data.T
            tf_data = tf_data[:, :3]
            tf_data[:, 2] = z_tmp
            #pdb.set_trace()


            data += list(tf_data)

            print(len(data))


            #point_cloud = o3d.geometry.PointCloud()
            #point_cloud.points = o3d.utility.Vector3dVector(point_cloud_data)

            # Visualizing the point cloud
            #o3d.visualization.draw_plotly([point_cloud])

        data_arr = np.array(data)

        np.save(filename, data_arr)
    else:
        data_arr = np.load(filename)
    data_arr = data_arr[data_arr[:, -1] >= -0.02]

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(data_arr)
    o3d.visualization.draw_plotly([point_cloud])

    dbscan = DBSCAN(eps=0.1, min_samples=1)
    #kmeans = KMeans(n_clusters=1, random_state=0, n_init="auto", max_iter=1000)

    #pdb.set_trace()
    outliers_before = data_arr
    for orientation in ["vertical"]:
        print(orientation)
        inliers_plane = [0, 0, 0]
        while len(inliers_plane) > 0:
            plane = Plane()
            equation, inliers_plane = plane.fit(outliers_before, 0.02, minPoints=700, maxIteration=10000, orientation=orientation)
            inliers_plane = outliers_before[inliers_plane, :]
            clusters = dbscan.fit_predict(inliers_plane)
            #pdb.set_trace()
            #clusters = kmeans.fit_predict(inliers_plane)

            unique_clusters = set(clusters)
            print(len(unique_clusters))
            biggest_cluster_size = -1
            biggest_cluster_points = None
            for cluster_label in unique_clusters:
                #print("okdawokdaw")
                #pdb.set_trace()
                cluster_points = inliers_plane[clusters == cluster_label]
                if len(cluster_points) > biggest_cluster_size:
                    #print("ok")
                    biggest_cluster_size = len(cluster_points)
                    biggest_cluster_points = cluster_points
            #pdb.set_trace()
            print(len(biggest_cluster_points), len(inliers_plane))
            outliers_plane = np.array([point for point in tqdm(outliers_before) if point not in biggest_cluster_points])
            outliers_full = np.array([point for point in tqdm(data_arr) if point not in biggest_cluster_points])
            print(f"outliers_plane.shape -> {outliers_plane.shape}")
            print(f"inliers_plane.shape -> {biggest_cluster_points.shape}")
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(outliers_plane)
            o3d.visualization.draw_plotly([point_cloud])

            outliers_before = outliers_plane



            #pdb.set_trace()


    pdb.set_trace()

if __name__ == '__main__':
    main()














