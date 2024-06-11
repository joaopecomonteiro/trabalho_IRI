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


angles_dict = {
    'draw_triangle': 120,
    'draw_square': 90,
    'draw_pentagon': 72
}


def sort_vertices_by_angle(vertices, centroid):
    centroid_x, centroid_y = centroid
    sorted_vertices = sorted(vertices, key=lambda v: np.arctan2(v[1] - centroid_y, v[0] - centroid_x))
    return sorted_vertices


def is_cycle(vertices):
    G = nx.Graph()
    num_vertices = len(vertices)
    for i in range(num_vertices):
        G.add_edge(tuple(vertices[i]), tuple(vertices[(i + 1) % num_vertices]))
    if len(list(nx.cycle_basis(G))) == 1 and all(len(list(G.neighbors(node))) == 2 for node in G.nodes):
        return True
    return False


def classify_shape(vertices, centroid):
    num_vertices = len(vertices)

    sorted_vertices = sort_vertices_by_angle(vertices, centroid)

    if num_vertices == 2:
        return "Plane"

    if not is_cycle(vertices):
        return f"Unknown with {num_vertices} vertices"

    if num_vertices == 3:
        side_lengths = []
        for i in range(num_vertices):
            x1, y1 = sorted_vertices[i]
            x2, y2 = sorted_vertices[(i + 1) % num_vertices]
            side_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            side_lengths.append(side_length)
        if all(abs(length - side_lengths[0]) < 0.1 for length in side_lengths):
            return "Regular Triangle"
        else:
            return "Triangle"

    elif num_vertices == 4:
        side_lengths = []
        for i in range(num_vertices):
            x1, y1 = sorted_vertices[i]
            x2, y2 = sorted_vertices[(i + 1) % num_vertices]
            side_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            side_lengths.append(side_length)
        if all(abs(length - side_lengths[0]) < 0.1 for length in side_lengths):
            return "Square"
        elif (abs(side_lengths[0] - side_lengths[2]) < 0.1 and
              abs(side_lengths[1] - side_lengths[3]) < 0.1):
            return "Rectangle"
        else:
            return "Polygon with 4 vertices"

    elif num_vertices == 5:
        side_lengths = []
        for i in range(num_vertices):
            x1, y1 = sorted_vertices[i]
            x2, y2 = sorted_vertices[(i + 1) % num_vertices]
            side_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            side_lengths.append(side_length)
        if all(abs(length - side_lengths[0]) < 0.1 for length in side_lengths):
            return "Pentagon"
        else:
            return "Polygon with 5 vertices"

    else:
        return f"Polygon with {num_vertices} vertices"


def calculate_rotation_angle(rotated_vertices, type_shape):
    # Sort vertices based on their y-coordinate
    sorted_vertices = sorted(rotated_vertices, key=lambda vertex: vertex[1])

    # Get the first two points (lowest y-coordinate)
    p1, p2 = sorted_vertices[:2]

    # Calculate the angle of the line connecting p1 and p2 with respect to the x-axis
    angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

    # Convert angle to degrees and adjust according to shape
    rotation_angle = np.degrees(angle) % angles_dict[type_shape]

    return rotation_angle


def parse_vertices(vertices_str):
    vertices = vertices_str.replace("(", "").replace(")", "").split(";")
    return [tuple(map(int, v.split(","))) for v in vertices]


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
    d = np.array([-a[3], -b[3], 0.]).reshape(3, 1)

    # could add np.linalg.det(A) == 0 test to prevent linalg.solve throwing error

    p_inter = np.linalg.solve(A, d).T

    return p_inter[0], (p_inter + aXb_vec)[0]


def main() -> None:
    pcd_filename = "point_clouds/zmap_test_4_ana.npy"
    #shapes_filename = "worlds/custom_maps/zmap_test_4_shapes.pkl"
    csv_points = "worlds/custom_maps/zmap_test_4_points.csv"
    output_csv = "results/comparison_results_zmap_test_4.csv"
    ground_truths = pd.read_csv("ground_truth/zmap_test_4_shapes.csv")

    data_arr = np.load(pcd_filename)
    data_arr = data_arr[data_arr[:, -1] >= -0.02]

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(data_arr)
    o3d.visualization.draw_plotly([point_cloud])

    source_points_df = pd.read_csv(csv_points, header=None, names=['x', 'y'])
    source_points = source_points_df[['x', 'y']].to_numpy()

    planes = []
    inliers_all = []
    outliers_before = data_arr
    for orientation in ["vertical"]:
        #print(orientation)
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
            inliers_all.extend(inliers_plane)

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
            selected_points.append([intersection_points[idx] for idx in cycle])

    cycle_centers = []
    for cycle in shapes:
        cycle_points = intersection_points[cycle]
        center = np.mean(cycle_points, axis=0)
        cycle_centers.append(center)

    cycle_centers = np.array(cycle_centers)

    orientation_angles = []
    detected_shapes = []
    for points, center in zip(selected_points, cycle_centers):
        flat_points = [[point[0] / 1000, point[1] / 1000] for point in points]
        classification = classify_shape(flat_points, center[:2])

        shape_key = ''
        if classification == "Triangle":
            shape_key = 'draw_triangle'
        elif classification == "Square" or classification == "Rectangle":
            shape_key = 'draw_square'
        elif classification == "Regular Pentagon":
            shape_key = 'draw_pentagon'

        if shape_key:
            rotation_angle = calculate_rotation_angle(flat_points, shape_key)
            orientation_angles.append(rotation_angle)
            detected_shapes.append(classification)
        else:
            orientation_angles.append(None)
            detected_shapes.append("Unknown")

    # Ensure all arrays have the same length
    num_points = len(intersection_points)
    num_centers = len(cycle_centers)
    num_angles = len(orientation_angles)

    repeated_centers = np.repeat(cycle_centers, num_points // num_centers + 1, axis=0)[:num_points]
    repeated_angles = np.repeat(orientation_angles, num_points // num_angles + 1)[:num_points]

    results_df = pd.DataFrame({
        'intersection_x': intersection_points[:, 0],
        'intersection_y': intersection_points[:, 1],
        'intersection_z': intersection_points[:, 2],
        'center_x': repeated_centers[:, 0],
        'center_y': repeated_centers[:, 1],
        'center_z': repeated_centers[:, 2],
        'angle_rad': repeated_angles,
        'angle_deg': [angle if angle is not None else None for angle in repeated_angles]
    })

    results_df.to_csv(output_csv, index=False)

    # Visualize the points and their orientation angles
    plt.figure(figsize=(10, 10))
    plt.scatter(np.array(inliers_all)[:, 0], np.array(inliers_all)[:, 1], c='orange', label='Inliers')
    plt.scatter(source_points[:, 0], source_points[:, 1], c='blue', label='Source Points')
    plt.scatter(intersection_points[:, 0], intersection_points[:, 1], c='green', label='Intersection Points')

    for point, angle in zip(cycle_centers, orientation_angles):
        if angle is not None:
            x, y = point[:2]
            plt.text(x, y, f"{angle:.2f}°", color='red', fontsize=12)

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

    comparison_data = []
    for index, row in ground_truths.iterrows():
        shape = row['Shape']
        vertices = parse_vertices(row['Vertices'])
        gt_angle = row['Angle']
        gt_center = np.array(eval(row['Center']))

        # Find the closest detected shape
        detected_center_idx = np.argmin(np.linalg.norm(cycle_centers[:, :2] * 1000 - gt_center[:2], axis=1))
        detected_center = cycle_centers[detected_center_idx] * 1000
        detected_angle = orientation_angles[detected_center_idx]

        center_error = np.linalg.norm(gt_center[:2] - detected_center[:2])
        angle_error = np.abs(gt_angle - detected_angle) if detected_angle is not None else None

        comparison_data.append({
            'Ground Truth Shape': shape,
            'Detected Shape': detected_shapes[detected_center_idx],
            'Center Error': center_error,
            'Angle Error': angle_error
        })

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df)




if __name__ == '__main__':
    main()