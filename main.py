import os
import platform

if platform.system() == "Linux":
    os.environ["WEBOTS_HOME"] = "/usr/local/webots"


from controller import Robot, Supervisor, GPS, Compass
from utils import warp_robot, move_robot_to

import numpy as np
import math
import pdb




def main() -> None:

    supervisor: Supervisor = Supervisor()

    timestep: int = int(supervisor.getBasicTimeStep())  # in ms

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


    moves = [(0, 0), (0, 1), (0, 2),
             (1, 2), (1, 1), (1, 0),
             (2, 0), (2, 1), (2, 2)]


    while supervisor.step() != -1:
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


if __name__ == '__main__':
    main()














