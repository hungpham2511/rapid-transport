import numpy as np
import openravepy as orpy
import argparse, toppra, toppra_app, os
from denso_control.controllers import JointPositionController
import ftsensorless as ft
import rospy


if __name__ == '__main__':
    n = rospy.init_node("open_loop")
    parse = argparse.ArgumentParser(description="Run parametrized trajectory.")
    parse.add_argument('-t', '--trajectory', help='Input trajectory specification.', required=True)
    parse.add_argument('-e', '--environment', help='Name of the environment. For collision checking during transiting segments only.', required=True)
    parse.add_argument('-r', '--robot', help='Name of the robot. For collision checking during transiting segments only.', required=True)
    parse.add_argument('-s', '--sampling', help="Sampling period.", default=0.006667)
    parse.add_argument('-v', '--verbose', help='More verbose output', action="store_true")
    args = vars(parse.parse_args())

    env = orpy.Environment()
    env.Load(args['environment'])
    robot = env.GetRobot(args['robot'])
    if args['verbose']:
        env.SetViewer('qtosg')
    assert robot is not None, "Unable to load any robot"

    joint_controller = JointPositionController('denso')

    db = toppra_app.database.Database()
    traj_profile = db.retrieve_profile(args['trajectory'], 'trajectory')

    if "waypoints_npz" in traj_profile:
        file_ = np.load(os.path.join(db.get_trajectory_data_dir(), traj_profile['waypoints_npz']))
        ts_waypoints = file_['t_waypoints']
        waypoints = file_['waypoints']
    elif "t_waypoints" in traj_profile:
        ts_waypoints = traj_profile['t_waypoints']
        waypoints = traj_profile['waypoints']
    else:
        raise IOError, "Waypoints not found in trajectory {:}".format(traj_profile['id'])
    path = toppra.SplineInterpolator(ts_waypoints, waypoints)
    q_init = path.eval(0)

    # Move to initial data point
    ft.rave_utils.move_to_joint_position(
        q_init, joint_controller, robot, dt=args['sampling'], require_confirm=True,
        velocity_factor=0.2, acceleration_factor=0.2)

    # Execute trajectory
    cmd = raw_input("Execute the trajectory y/[N]?  ")
    dt = args['sampling']
    if cmd != "y":
        print("Does not execute anything. Exit.")
    else:
        try:
            safety_factor = float(raw_input("Execute at a slower speed? Scale velocity down by [default=1.0]: "))
        except:
            safety_factor = 1.0
        ts_samples = np.arange(0, path.get_duration(), dt * safety_factor)
        qs_samples = path.eval(ts_samples)
        print("Executing trajectory slowed down by {:f}.".format(safety_factor))
        for q in qs_samples:
            t0 = rospy.get_time()
            joint_controller.set_joint_positions(q)
            robot.SetDOFValues(q, range(6))
            t_elapsed = rospy.get_time() - t0
            rospy.sleep(dt - t_elapsed)
        print("Trajectory execution finished! Exit now.")

    exit()

