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

import networkx as nx

def find_squares(edges):
    #breakpoint()
    # Create the graph
    G = nx.Graph()
    G.add_edges_from(edges)


    # List to store the squares
    squares = []


    # Iterate over each node and find squares
    for node in G.nodes():
        neighbors_2 = []
        square = []
        #breakpoint()
        # Find all 2-neighbors of the node (nodes with distance 2 from the current node)
        for neighbor in G.neighbors(node):
            for n in G.neighbors(neighbor):
                if n != node and n not in neighbors_2:
                    neighbors_2.append(n)
                    square.append(n)
        #breakpoint()

        square.append(node)
        #square.append(neighbors_2)
        # Check each pair of 2-neighbors to see if they form a square with the current node
        for n1 in neighbors_2:
            for n2 in G.neighbors(n1):
                if G.has_edge(n2, node):
                #for n2 in G.neighbors(node):
                #    if G.has_edge(n1, n2):
                    #square = sorted([node, n1, n2, list(set(G.neighbors(node)).intersection(G.neighbors(n1), G.neighbors(n2)))[0]])
                    square.append(n2)

        if sorted(square) not in squares:
            squares.append(sorted(square))

    return squares

def plane_intersect(a, b):
    """
    a, b   4-tuples/lists
           Ax + By +Cz + D = 0
           A,B,C,D in order

    output: 2 points on line of intersection, np.arrays, shape (3,)
    """
    a_vec, b_vec = np.array(a[:3]), np.array(b[:3])

    aXb_vec = np.cross(a_vec, b_vec)

    A = np.array([a_vec, b_vec, aXb_vec])
    d = np.array([-a[3], -b[3], 0.]).reshape(3,1)

    # could add np.linalg.det(A) == 0 test to prevent linalg.solve throwing error

    p_inter = np.linalg.solve(A, d).T

    return p_inter[0], (p_inter + aXb_vec)[0]




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

            data += list(tf_data)

            print(len(data))

        data_arr = np.array(data)

        np.save(filename, data_arr)
    else:
        data_arr = np.load(filename)
    data_arr = data_arr[data_arr[:, -1] >= -0.02]

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(data_arr)
    o3d.visualization.draw_plotly([point_cloud])


    planes = []

    outliers_before = data_arr
    for orientation in ["vertical"]:
        print(orientation)
        inliers_plane = [0, 0, 0]
        while len(inliers_plane) > 0:
            plane = Plane()
            equation, inliers_plane = plane.fit(outliers_before, 0.02, minPoints=700, maxIteration=10000, orientation=orientation)
            print(len(inliers_plane))
            if len(inliers_plane) <= 0:
                break
            outliers_plane = np.array([point for point in tqdm(outliers_before) if point not in inliers_plane])
            outliers_full = np.array([point for point in tqdm(data_arr) if point not in inliers_plane])
            print(f"outliers_plane.shape -> {outliers_plane.shape}")
            print(f"inliers_plane.shape -> {inliers_plane.shape}")
            #point_cloud = o3d.geometry.PointCloud()
            #point_cloud.points = o3d.utility.Vector3dVector(outliers_plane)
            #o3d.visualization.draw_plotly([point_cloud])

            outliers_before = outliers_plane

            planes.append((equation, inliers_plane))
    z = np.max(data_arr[:, 2])
    intersection_points = []
    intersection_edges = []
    #point_a, point_b = plane_intersect(planes[0][0], planes[1][0])
    # for a, plane_a in enumerate(planes):
    #     for b, plane_b in enumerate(planes):
    for a in range(len(planes)):
        for b in range(a, len(planes)):
            read = 0
            plane_a = planes[a]
            plane_b = planes[b]
            if plane_a != plane_b:
                point_a, point_b = plane_intersect(plane_a[0], plane_b[0])
                x, y = point_a[0], point_a[1]
                for point in plane_a[1]:
                    x2, y2 = point[0], point[1]
                    if np.sqrt((x-x2)**2 + (y-y2)**2) < 0.2:
                        read += 1
                        break
                for point in plane_b[1]:
                    x2, y2 = point[0], point[1]
                    if np.sqrt((x - x2) ** 2 + (y - y2) ** 2) < 0.2:
                        read += 1
                        break
                if read == 2:
                    print(a, b)
                    intersection_point = np.array([x, y, z])
                    intersection_points.append(intersection_point)
                    intersection_edges.append((a, b))
                    intersection_edges.append((b, a))


    intersection_points = np.array(intersection_points)

    squares = find_squares(intersection_edges)

    matrix = np.zeros((2000, 2000))
    matrix[0][0] = 1
    matrix[-1][0] = 1
    matrix[0][-1] = 1
    matrix[-1][-1] = 1


    breakpoint()


if __name__ == '__main__':
    main()














