import os
import platform

if platform.system() == "Linux":
    os.environ["WEBOTS_HOME"] = "/usr/local/webots"


from controller import Robot, Supervisor, GPS, Compass, Lidar, LidarPoint

from utils import warp_robot, move_robot_to, bresenham
from transformations import get_translation, create_tf_matrix

from occupancy_grid import OccupancyGrid


import numpy as np
import math
import pdb

import open3d as o3d


class DeterministicOccupancyGrid(OccupancyGrid):
    def __init__(self, origin: (float, float), dimensions: (int, int), resolution: float):
        super().__init__(origin, dimensions, resolution)

        # Initialize the grid
        #self.occupancy_grid: np.ndarray  # TODO
        self.occupancy_grid = np.ones(shape=self.dimensions)*0.5
    def update_map(self, robot_tf: np.ndarray, lidar_points: [LidarPoint]) -> None:
        # Get the grid coord for the robot pose
        print("inside update_map")
        #pdb.set_trace()
        #robot_coord: (int, int)  # TODO
        robot_coord = self.real_to_grid_coords(get_translation(robot_tf))

        # Get the grid coords for the lidar points
        grid_lidar_coords: [(int, int)] = []
        for point in lidar_points:
            coords = robot_tf @ np.array([point.x, point.y, 0, 1])
            grid_point = self.real_to_grid_coords(coords)
            grid_lidar_coords.append(grid_point)
            bresenham_points = bresenham(robot_coord, grid_point)
            for bre_point in bresenham_points[1:-1]:
                self.update_cell(bre_point, False)
            self.update_cell(grid_point, True)
            #pass  # TODO
        # Set as free the cell of the robot's position
        self.update_cell(robot_coord, False)  # TODO

        # Set as free the cells leading up to the lidar points
        # TODO
        #for point in grid_lidar_coords:
        #    bresenham_points = bresenham(robot_coord, point)
        #    for point in bresenham_points:
        #        self.update_cell(point, False)
        #    self.update_cell(point, True)
        # Set as occupied the cells for the lidar points
        # TODO

    def update_cell(self, coords: (int, int), is_occupied: bool) -> None:
        if self.are_grid_coords_in_bounds(coords):
            # Update the grid cell
            #import pdb
            #pdb.set_trace()
            self.occupancy_grid[coords] = int(is_occupied)  # TODO




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

    moves = [(0.10, 0.10), (0.10, 0.40), (0.10, 0.70),
             (0.40, 0.70), (0.70, 0.70),
             (0.70, 0.40), (0.70, 0.10)]
    #moves = [(0.10, 0.10)]

    while len(moves) != 0:
        supervisor.step()
        step_distance: float = 1
        #new_position: (float, float) = (robot_position[0] + step_distance,
        #                                robot_position[1] + step_distance)
        new_position = moves.pop(0)
        print(new_position)
        move_robot_to(supervisor, robot_position, robot_orientation, new_position, 0.1, math.pi)

        gps_readings = gps.getValues()
        robot_position = (gps_readings[0], gps_readings[1])
        compass_readings = compass.getValues()
        robot_orientation = math.atan2(compass_readings[0], compass_readings[1])

        pcd = lidar.getPointCloud()

        #data_tmp = np.array([[point.x, point.y, 0] for point in pcd if math.isfinite(point.x) and math.isfinite(point.y)])


        robot_tf: np.ndarray = create_tf_matrix((robot_position[0], robot_position[1], 0.0), robot_orientation)
        data_tmp = np.array([[point.x, point.y, point.z, 1] for point in pcd if math.isfinite(point.x) and math.isfinite(point.y) and math.isfinite(point.z)])
        data_tmp = data_tmp.T

        #robot_tf[:2,:2] = 0
        #robot_tf[0, 0] = 1
        #robot_tf[1, 1] = 1

        tf_data = robot_tf @ data_tmp
        tf_data = tf_data.T
        tf_data = tf_data[:, :3]

        #pdb.set_trace()


        data += list(tf_data)

        print(len(data))


        #point_cloud = o3d.geometry.PointCloud()
        #point_cloud.points = o3d.utility.Vector3dVector(point_cloud_data)

        # Visualizing the point cloud
        #o3d.visualization.draw_plotly([point_cloud])



    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(data)
    o3d.visualization.draw_plotly([point_cloud])


    pdb.set_trace()

if __name__ == '__main__':
    main()














