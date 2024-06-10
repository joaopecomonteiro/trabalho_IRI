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
import pickle
import matplotlib.pyplot as plt
import open3d as o3d
import pandas as pd
#import pyransac3d as pyrsc
from skimage.draw import line
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







def find_all_cycles(graph):
    def dfs_cycle(start, current, visited, stack, cycles):
        visited[current] = True
        stack.append(current)

        for neighbor in graph.neighbors(current):
            if not visited[neighbor]:
                dfs_cycle(start, neighbor, visited, stack, cycles)
            elif neighbor == start and len(stack) > 2:
                cycle = stack[:] + [start]
                cycles.append(cycle)

        stack.pop()
        visited[current] = False

    cycles = []
    for node in graph.nodes():
        visited = {n: False for n in graph.nodes()}
        dfs_cycle(node, node, visited, [], cycles)

    # Remove duplicate cycles (considering rotations and reversed versions)
    unique_cycles = []
    for cycle in cycles:
        cycle = cycle[:-1]  # Remove the duplicate start/end node
        normalized_cycle = tuple(sorted(cycle))
        if normalized_cycle not in unique_cycles:
            unique_cycles.append(normalized_cycle)

    return [list(cycle) for cycle in unique_cycles]

def find_cycles(edges):
    # Create the graph
    G = nx.Graph()
    G.add_edges_from(edges)

    # Find all cycles in the graph
    cycles = find_all_cycles(G)

    return cycles



def detect_orientation_icp(source_points: np.ndarray, target_points: np.ndarray) -> float:
    def convert_to_3d(points_2d: np.ndarray) -> np.ndarray:
        return np.hstack((points_2d, np.zeros((points_2d.shape[0], 1))))

    source_points_3d = convert_to_3d(source_points)
    target_points_3d = convert_to_3d(target_points)

    source_pcd = o3d.geometry.PointCloud()
    target_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_points_3d)
    target_pcd.points = o3d.utility.Vector3dVector(target_points_3d)

    threshold = 0.02
    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    rotation_matrix = reg_p2p.transformation[:3, :3]

    # Ensure the rotation matrix is in the xy-plane
    angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    # Convert to counterclockwise from the x-axis
    if angle < 0:
        angle += 2 * np.pi

    return angle



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
    pcd_filename = "point_clouds/zzzmap_test_41.npy"
    shapes_filename = "worlds/custom_maps/zzzmap_test_4_shapes.pkl"
    csv_points = "worlds/custom_maps/zzzmap_test_4_points.csv"
    output_csv = "results/comparison_results.csv"

    data_arr = np.load(pcd_filename)
    data_arr = data_arr[data_arr[:, -1] >= -0.02]

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(data_arr)
    o3d.visualization.draw_plotly([point_cloud])

    source_points_df = pd.read_csv(csv_points, header=None, names=['x', 'y'])
    source_points = source_points_df[['x', 'y']].to_numpy()

    planes = []
    outliers_before = data_arr
    for orientation in ["vertical"]:
        print(orientation)
        inliers_plane = [0, 0, 0]
        while len(inliers_plane) > 0:
            plane = Plane()
            equation, inliers_plane = plane.fit(outliers_before, 0.02, minPoints=300, maxIteration=10000,
                                                orientation=orientation, random_seed=42)

            if len(inliers_plane) <= 0:
                break
            outliers_plane = np.array([point for point in outliers_before if point not in inliers_plane])
            outliers_full = np.array([point for point in data_arr if point not in inliers_plane])

            outliers_before = outliers_plane
            planes.append((equation, inliers_plane))

    z = np.max(data_arr[:, 2])
    intersection_points = []
    intersection_edges = []
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
                    if np.sqrt((x - x2) ** 2 + (y - y2) ** 2) < 0.1:
                        read += 1
                        break
                for point in plane_b[1]:
                    x2, y2 = point[0], point[1]
                    if np.sqrt((x - x2) ** 2 + (y - y2) ** 2) < 0.1:
                        read += 1
                        break
                if read == 2:
                    intersection_point = np.array([x, y, z])
                    intersection_points.append(intersection_point)
                    intersection_edges.append((a, b))

    edges = []
    for plane_idx, plane in enumerate(planes):
        plane_eq = plane[0]
        edge_points = []
        for point_idx, point in enumerate(intersection_points):
            dist = (plane_eq[0] * point[0] + plane_eq[1] * point[1] + plane_eq[2] * point[2] + plane_eq[3]) / np.sqrt(
                plane_eq[0] ** 2 + plane_eq[1] ** 2 + plane_eq[2] ** 2)
            if np.abs(dist) <= 0.0001:
                edge_points.append(point_idx)
        if len(edge_points) == 2:
            edges.append(edge_points)
    intersection_points = np.array(intersection_points)
    intersection_edges = edges

    shapes = find_cycles(intersection_edges)

    selected_points = []
    for cycle in shapes:
        if cycle:
            selected_points.append(intersection_points[cycle[0]])

    orientation_angles = []
    target_points_list = []
    for point in selected_points:
        target_points = point[:2].reshape(1, -1)
        orientation_angle = detect_orientation_icp(source_points, target_points)
        orientation_angles.append((point, orientation_angle))
        target_points_list.append(target_points)

    target_points_array = np.vstack(target_points_list)

    cycle_centers = []
    for cycle in shapes:
        cycle_points = intersection_points[cycle]
        center = np.mean(cycle_points, axis=0)
        cycle_centers.append(center)

    cycle_centers = np.array(cycle_centers)

    # Ensure all arrays have the same length
    num_points = len(intersection_points)
    num_centers = len(cycle_centers)
    num_angles = len(orientation_angles)

    repeated_centers = np.repeat(cycle_centers, num_points // num_centers + 1, axis=0)[:num_points]
    repeated_angles = np.repeat([angle for _, angle in orientation_angles], num_points // num_angles + 1)[:num_points]

    results_df = pd.DataFrame({
        'intersection_x': intersection_points[:, 0],
        'intersection_y': intersection_points[:, 1],
        'intersection_z': intersection_points[:, 2],
        'center_x': repeated_centers[:, 0],
        'center_y': repeated_centers[:, 1],
        'center_z': repeated_centers[:, 2],
        'angle_rad': repeated_angles,
        'angle_deg': np.degrees(repeated_angles)
    })

    results_df.to_csv(output_csv, index=False)

    # Visualize the points and their orientation angles
    plt.figure(figsize=(10, 10))
    plt.scatter(source_points[:, 0], source_points[:, 1], c='blue', label='Source Points')
    plt.scatter(intersection_points[:, 0], intersection_points[:, 1], c='green', label='Intersection Points')
    plt.scatter(target_points_array[:, 0], target_points_array[:, 1], c='red', label='Target Points')
    plt.scatter(cycle_centers[:, 0], cycle_centers[:, 1], c='purple', label='Cycle Centers', marker='x')

    for point, angle in orientation_angles:
        x, y = point[:2]
        plt.arrow(x, y, 0.1 * np.cos(angle), 0.1 * np.sin(angle), color='r', head_width=0.05)
        plt.text(x, y, f"{np.degrees(angle):.2f}Â°", color='red', fontsize=12)

    for center in cycle_centers:
        x, y = center[:2]
        plt.text(x, y - 0.05, f"({x:.2f}, {y:.2f})", color='purple', fontsize=12, ha='center', va='top')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Source Points, Intersection Points, and Target Points with Orientation Angles and Cycle Centers')
    plt.grid(True)
    plt.show()

    # Visualize the intersection edges and points
    matrix = np.zeros((530, 530))
    for edge in intersection_edges:
        point_a = np.round(intersection_points[edge[0]] * 100).astype(int) + 15
        point_b = np.round(intersection_points[edge[1]] * 100).astype(int) + 15

        rr, cc = line(point_a[0], point_a[1], point_b[0], point_b[1])
        matrix[rr, cc] = 1

    for point in intersection_points:
        point = np.round(point * 100).astype(int) + 15
        matrix[point[0], point[1]] = 2

    plt.imshow(np.rot90(matrix))
    plt.show()

    #with open(shapes_filename, 'rb') as f:
    #    gt_shapes = pickle.load(f)



if __name__ == '__main__':
    main()