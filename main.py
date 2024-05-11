import os
import platform

if platform.system() == "Linux":
    os.environ["WEBOTS_HOME"] = "/usr/local/webots"


from controller import Robot, Supervisor, GPS, Compass, Lidar, LidarPoint

from utils import warp_robot, move_robot_to, bresenham
from transformations import get_translation, create_tf_matrix


from tqdm import tqdm
import numpy as np
import math
import pdb

import open3d as o3d
import pyransac3d as pyrsc

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



    x = np.arange(0, 4, 0.3)
    y = np.arange(0, 4, 0.2)
    moves = [(tx, ty) for tx in x for ty in y]
    print(f"number of moves: {len(moves)}")
    #moves = [(0.10, 0.10)]

    while len(moves) != 0:
        supervisor.step()
        step_distance: float = 1
        #new_position: (float, float) = (robot_position[0] + step_distance,
        #                                robot_position[1] + step_distance)
        new_position = moves.pop(0)
        print(new_position)
        move_robot_to(supervisor, robot_position, robot_orientation, new_position, 1, math.pi)

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


    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(data_arr)
    o3d.visualization.draw_plotly([point_cloud])

    #cylinder = pyrsc.Cylinder()
    #center, axis, radius, inliers = cylinder.fit(data_arr, 0.5)

    # cuboid1 = pyrsc.Cuboid()
    # cuboid2 = pyrsc.Cuboid()
    #
    # best_eq1, inliers_cuboid1 = cuboid1.fit(data_arr, 0.005)
    #
    # print(f"number of inliers cuboid: {len(inliers_cuboid1)}")
    #
    #
    # inliers_cuboid1 = data_arr[inliers_cuboid1, :]
    # outliers_cuboid1 = np.array([point for point in tqdm(data_arr) if point not in inliers_cuboid1])
    # print(outliers_cuboid1.shape)
    #
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(inliers_cuboid1)
    # o3d.visualization.draw_plotly([point_cloud])
    #
    # #cylinder = pyrsc.Cylinder()
    # #center, axis, radius, inliers_cylinder = cylinder.fit(inliers_cuboid, 0.001)
    # #inliers_cylinder = inliers_cuboid[inliers_cylinder, :]
    #
    # best_eq2, inliers_cuboid2 = cuboid2.fit(outliers_cuboid1, 0.001)
    # #pdb.set_trace()
    # inliers_cuboid2 = outliers_cuboid1[inliers_cuboid2, :]
    # outliers_cuboid2 = np.array([point for point in tqdm(outliers_cuboid1) if point not in inliers_cuboid2])
    #
    # print(f"number of inliers cylinder : {len(inliers_cuboid2)}")
    #
    #
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(inliers_cuboid2)
    # o3d.visualization.draw_plotly([point_cloud])

    outliers_before = data_arr
    for i in range(9):
        plane = pyrsc.Plane()

        equation, inliers_plane = plane.fit(outliers_before, 0.00001, minPoints=2000, maxIteration=100000)
        inliers_plane = outliers_before[inliers_plane, :]
        outliers_plane = np.array([point for point in tqdm(outliers_before) if point not in inliers_plane])
        print(inliers_plane.shape)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(inliers_plane)
        o3d.visualization.draw_plotly([point_cloud])

        outliers_before = outliers_plane
    pdb.set_trace()

if __name__ == '__main__':
    main()














