import argparse, yaml, os
import toppra_app, toppra, time
import openravepy as orpy
import numpy as np

if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="A program for processing and simplifying contacts from raw data points. ")
    parse.add_argument('-t', '--trajectory', help='Input trajectory specification.', required=True)
    parse.add_argument('-o', '--object', help='Object specification.', required=True)
    args = vars(parse.parse_args())

    db = toppra_app.database.Database()
    traj_profile = db.retrieve_profile(args['trajectory'], "trajectory")
    obj_profile = db.retrieve_profile(args['object'], "object")

    env = orpy.Environment()
    env.Load('models/' + traj_profile['attached_to_robot'])
    robot = env.GetRobots()[0]
    env.SetViewer('qtosg')

    env.Load("models/" + obj_profile['rave_model'])
    obj = env.GetBodies()[1]
    manip = robot.GetManipulator(obj_profile['attached_to_manipulator'])
    T_manip_obj = np.eye(4)
    T_manip_obj[:3, 3] = obj_profile['position']
    T_manip_obj[:3, :3] = obj_profile['orientation']

    while True:
        db = toppra_app.database.Database()
        traj_profile = db.retrieve_profile(args['trajectory'], "trajectory")
        if "waypoints_npz" in traj_profile:
            file_ = np.load(os.path.join(db.get_trajectory_data_dir(), traj_profile['waypoints_npz']))
            path = toppra.SplineInterpolator(file_['t_waypoints'], file_['waypoints'])
            N_waypoints = len(file_['t_waypoints'])
        elif 't_waypoints' in traj_profile:
            path = toppra.SplineInterpolator(traj_profile['t_waypoints'], traj_profile['waypoints'])
            N_waypoints = len(traj_profile['t_waypoints'])
        else:
            raise IOError, "Waypoints not found!"
        q_samples = path.eval(np.arange(0, path.get_duration(), 0.01))
        cmd = raw_input("[Enter]: play trajectory, [r] to reload, [number] to view waypoint, [q] to quit: ")
        if cmd == "q":
            exit()
        elif cmd == "r":
            continue
        elif cmd == "":
            for i, q in enumerate(q_samples):
                robot.SetDOFValues(q, range(6))
                T_obj = manip.GetTransform().dot(T_manip_obj)
                obj.SetTransform(T_obj)
                if env.CheckCollision(robot):
                    print "Robot is in collision at waypoint {:d}!".format(i)
                time.sleep(0.01)
        else:
            try:
                i = int(cmd) - 1
                robot.SetDOFValues(path.waypoints[i], range(6))
                T_obj = manip.GetTransform().dot(T_manip_obj)
                obj.SetTransform(T_obj)
            except Exception as e:
                print(e)



