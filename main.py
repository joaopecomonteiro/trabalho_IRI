import os
os.environ["WEBOTS_HOME"] = "/usr/local/webots"

from controller import Robot, LidarPoint, Lidar, Compass, GPS
from trabalho.localization_utils import draw_real_vs_estimated_localization
from trabalho.transformations import create_tf_matrix, get_translation, get_rotation

import numpy as np
import math
import pdb




def main() -> None:
    robot: Robot = Robot()

    min_x: float = -0.5
    min_y: float = -0.5
    max_x: float = 0.5
    max_y: float = 0.5

    timestep: int = int(robot.getBasicTimeStep())  # in ms

    lidar: Lidar = robot.getDevice('lidar')
    lidar.enable(int(robot.getBasicTimeStep()))
    lidar.enablePointCloud()

    compass: Compass = robot.getDevice('compass')
    compass.enable(timestep)

    gps: GPS = robot.getDevice('gps')
    gps.enable(timestep)
    robot.step()

    # Read the ground-truth (correct robot pose)
    gps_readings: [float] = gps.getValues()
    actual_position: (float, float) = (gps_readings[0], gps_readings[1])
    compass_readings: [float] = compass.getValues()
    actual_orientation: float = math.atan2(compass_readings[0], compass_readings[1])

    robot_tf: np.ndarray = create_tf_matrix((actual_position[0], actual_position[1], 0.0), actual_orientation)

    # Draw a point cloud for a square map
    num_divisions: int = 100
    fixed_points: [(float, float, float)] = []
    for i in range(num_divisions):
        x: float = min_x + (max_x - min_x) * (i / float(num_divisions - 1))
        fixed_points.append([x, min_y, 0.0])
        fixed_points.append([x, max_y, 0.0])

        y: float = min_y + (max_y - min_y) * (i / float(num_divisions - 1))
        fixed_points.append([min_x, y, 0.0])
        fixed_points.append([max_x, y, 0.0])
    fixed_cloud: np.ndarray = np.asarray(fixed_points)

    estimated_translations, estimated_rotations = find_possible_poses(robot_tf, lidar.getPointCloud(), min_x, max_x, min_y, max_y)
    draw_real_vs_estimated_localization(fixed_cloud,
                                        actual_position, actual_orientation,
                                        estimated_translations, estimated_rotations)

    while robot.step() != -1:
        pass


if __name__ == '__main__':
    main()














