import time
import numpy as np
import pybullet as p
import pybullet_data

from simulator import Simulator

from security.loadUrdf import loadPlane, loadRobot

from .forwardKinematics import ForwardKinematic

import plotly.graph_objects as go

from urbasic import ISCoin, Joint6D




class RobotCollisionCheck :
    def __init__(self, gui=False, logs=False):
        self.simu = Simulator(gui=gui, deltaT=1 / 100, log=logs)

        self.logs =logs
        self.gui = gui

    def check_working_area(self):
        x_min = -0.31
        x_max = 0.62

        y_min = 0
        y_max = 0.62

        z_min = -0.0
        z_max = 0.62

        return self.simu.isPenInArea(
            x_bounds=(x_min, x_max), y_bounds=(y_min, y_max), z_bounds=(z_min, z_max)
        )

    def isValidConfiguration(self, angles):
        self.simu.resetAtPosition(angles=angles)
        self.simu.stepSimu()

        gotCollision = self.simu.check_collision()
        isInArea = self.check_working_area()

        return not gotCollision and isInArea

    def isValidMovement(self, startAngles, destAngles):
        return self.simu.isMoveValid(startAngles=startAngles, destAngles=destAngles)

    def runSimulation(self, angles):

        collison = False 
        while p.isConnected():
            collison = self.isValidConfiguration(angles)

            if collison is not None:
                result = collison
                collison = False  # Reset collison before returning
                return result
 
collison = [ 0.0, -3.14, 2.2, 0.6, 3.0, 0.0]
nocoll = [0.9509, -1.6623, 0.6353, -0.5976, -1.5722, 0.0]
collGround = [ 1.5,-3.9,-3, 0.0, 0.0, 0.0]
testAngles =   [
                -0.1122,
                -3.0658,
                0.4916,
                2.5814,
                1.4586,
                -1.5638
            ]
if __name__ == "__main__":
   
    # iscoin = ISCoin(host="10.30.5.159", opened_gripper_size_mm=40)
    # angles= list(iscoin.robot_control.get_actual_joint_positions())
    # print(angles)
    test = RobotCollisionCheck(gui=False)

    print(test.isValidConfiguration(testAngles))

    while True:
        pass