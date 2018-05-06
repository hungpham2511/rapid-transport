"""
This script run trajectories
"""
import numpy as np
import rospy
import matplotlib.pyplot as plt
import openravepy as orpy
import ftsensorless as ft
from denso_control.controllers import JointPositionController

def execute_trajectory(q_array, joint_controller, robot, dt=1.0/150, slow_factor=1.0):
    """

    Parameters
    ----------
    q_array
    joint_controller
    robot
    dt

    Returns
    -------

    """
    success = True
    for q in q_array:
        t_start = rospy.get_time()
        joint_controller.set_joint_positions(q)
        robot.SetDOFValues(q, range(6))
        t_elapsed = rospy.get_time() - t_start
        if t_elapsed > dt:
            rospy.logwarn("Cycle time is greater than dt")
            success = False
        rospy.sleep(slow_factor * dt - t_elapsed)
    return success

if __name__ == '__main__':
    n = rospy.init_node("open_loop")
    robot = ft.rave_utils.load_openrave_robot('worlds/caged_denso_gripper.env.xml',
                                              'denso_gripper')
    orpy.RaveSetDebugLevel(orpy.DebugLevel.Fatal)

    # Load robot model and setup controller
    manip = robot.SetActiveManipulator('denso_ft_sensor_gripper')
    robot.SetDOFValues([0.7], dofindices=manip.GetGripperIndices())
    robot.SetActiveDOFs(manip.GetArmIndices())
    joint_controller = JointPositionController('denso')
    rospy.sleep(0.5)

    # Load trajectory
    trajectory_name = 'test_trajectory_1'
    saved_data = np.load("trajectories/{:}.npz".format(trajectory_name))
    q_uniform = saved_data['q_uniform']
    ft.rave_utils.move_to_joint_position(q_uniform[0], joint_controller, robot)
    raw_input("[Enter] to run trajectory!")
    execute_trajectory(q_uniform, joint_controller, robot, dt=1.0 / 150, slow_factor=1.0)

    
    



