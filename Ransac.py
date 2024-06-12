import os
from tqdm import tqdm
import numpy as np
import math
import pickle
import matplotlib.pyplot as plt
import open3d as o3d
import pandas as pd
from plane import Plane


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

def Ransac(data_arr, min_points=1000, threshold=0.05, max_iteration=1000):

    planes = []
    inliers_all = []
    outliers_before = data_arr
    for orientation in ["vertical"]:

        inliers_plane = [0, 0, 0]
        while len(inliers_plane) > 0:
            plane = Plane()
            total_points = len(outliers_before)
            equation, inliers_plane_ids = plane.fit(outliers_before, threshold, minPoints=min_points, maxIteration=max_iteration,
                                                orientation=orientation)

            if len(inliers_plane) <= 0:
                break

            all_indices = np.arange(outliers_before.shape[0])
            outliers_plane_indices = np.setdiff1d(all_indices, inliers_plane_ids)
            outliers_plane = outliers_before[outliers_plane_indices]

            inliers_plane = outliers_before[inliers_plane_ids]
            inliers_all.extend(inliers_plane)
            outliers_before = outliers_plane


            if len(equation) > 0:
                planes.append((equation, inliers_plane))
            print(f"len planes -> {len(planes)}, n points left -> {len(outliers_before)}")

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
                    if np.sqrt((x - x2) ** 2 + (y - y2) ** 2) < 0.2:
                        read += 1
                        break
                for point in plane_b[1]:
                    x2, y2 = point[0], point[1]
                    if np.sqrt((x - x2) ** 2 + (y - y2) ** 2) < 0.2:
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

    return planes, intersection_points, intersection_edges, inliers_all


if __name__ == '__main__':

    mapname = "star"
    pcd_filename = f"point_clouds/{mapname}.npy"

    data_arr = np.load(pcd_filename)
    data_arr = data_arr[data_arr[:, -1] >= 0]

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(data_arr)
    o3d.visualization.draw_plotly([point_cloud])

    planes, intersection_points, intersection_edges, inliers_all = Ransac(data_arr,
                                                                          min_points=400,
                                                                          threshold=0.01,
                                                                          max_iteration=50000)

    plt.figure(figsize=(10, 10))

    for edge in intersection_edges:
        point_a = intersection_points[edge[0]]
        point_b = intersection_points[edge[1]]
        plt.plot([point_a[0], point_b[0]], [point_a[1], point_b[1]], c="green", label='Ransac')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left')

    plt.show()
    #plt.scatter(intersection_points[:, 0], intersection_points[:, 1])
    #plt.show()

    breakpoint()


















