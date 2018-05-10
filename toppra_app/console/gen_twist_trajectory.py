import argparse, yaml, os
import toppra, toppra_app
import openravepy as orpy
import numpy as np
import hashlib


def main(env=None):
    """Main logic for generate twist program.

    Parameters
    ----------
    env: optional
        An openravepy environment. If passed, reset this environment
        and continue working on it.
    """
    parser = argparse.ArgumentParser(description="A program for generating twist motions.")
    parser.add_argument('-q', "--joint_values", help="Joint values at which to compute the motion.", required=True)
    parser.add_argument('-e', "--environment", help="Path to openrave environment.", default="caged_denso_ft_sensor_suction.env.xml")
    parser.add_argument('-r', "--robot_name", help="Robot name.", default="denso")
    parser.add_argument('-m', "--manipulator", help="Manipulator.", default="denso_suction_cup")
    parser.add_argument('-a', "--angle", help="Max twisting angle.", default="0.2")
    parser.add_argument('-v', "--verbose", action="store_true", default=False)
    args = parser.parse_args()

    db = toppra_app.database.Database()
    if env is None:
        env = orpy.Environment()
    else:
        env.Reset()
    env.Load(toppra_app.utils.expand_and_join(db.get_model_dir(), args.environment))
    if args.verbose:
        env.SetViewer("qtosg")
    robot = env.GetRobot(args.robot_name)
    manip = robot.SetActiveManipulator(args.manipulator)
    robot.SetActiveDOFValues(np.array(yaml.load(args.joint_values)))
    traj_array = toppra_app.generate_twist_at_active_conf(robot, max_angle=float(args.angle))

    identify_string = args.joint_values + args.environment + args.robot_name + args.manipulator + args.angle
    traj_param_id = "twist" + "_" + hashlib.md5(identify_string).hexdigest()[:10]
    cmd = raw_input("Save this traj as {:}? y/[N]".format(traj_param_id))

    if cmd != 'y':
        print("Exit without saving!")
    else:
        np.savez(os.path.join(db.get_trajectory_data_dir(), traj_param_id+".npz"),
                 t_waypoints=np.linspace(0, 1, traj_array.shape[0]),
                 waypoints=traj_array)
        traj_profile = {
            "id": traj_param_id,
            'reparametrized': False,
            'waypoints_npz': traj_param_id+".npz",
            "max_angle": args.angle
        }
        db.insert_profile(traj_profile, "trajectory")
    return True


