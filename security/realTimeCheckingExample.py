from math import radians
import json

from urbasic import ISCoin, Joint6D

from .globalRobotChecking import GlobalRobotChecking


# function to read a simulation json and read it to control the robot
def readJson(path):
	points = []
	with open(path, 'r') as file:
		data = json.load(file)["modTraj"]
		for i in data:
			points.append(i['positions'])
	return points
# Create a new ISCoin object
# UR3e1 IP (closest to window): 10.30.5.158
# UR3e2 IP: 10.30.5.159
iscoin = ISCoin(host="10.30.5.159", opened_gripper_size_mm=40)

interval = 0.5 # in case of need you cam change the interval


# Reset any potential error
iscoin.robot_control.reset_error()

jsonPath = "./traj_test.json"
waypoints = readJson(jsonPath)

checking_task = GlobalRobotChecking(waypoints[0],interval, iscoin=iscoin)  #change with the first angle detected by the robot
checking_task.start()  # Start the task
for i in waypoints:
	
	jo = Joint6D.createFromRadList(i)

	if not checking_task.isValid:
		print("The robot is not in a stable position")
		break
	#print(f'Joints are at {"iscoin.robot_control.get_actual_joint_positions()"} - going to {jo}')
	iscoin.robot_control.movej(jo, a = radians(20), v = radians(20))
	
checking_task.stop()  # Stop the task """
