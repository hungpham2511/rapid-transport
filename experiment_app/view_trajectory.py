import argparse, yaml, os
import toppra_app, toppra, time
import openravepy as orpy
import numpy as np


class ViewTrajectory(object):

    def __init__(self, traj_id, env_dir, robot_name, object_id, link_name, T_link_object):
        self.db = toppra_app.database.Database()
        self.traj_profile = self.db.retrieve_profile(traj_id, "trajectory")
        obj_profile = self.db.retrieve_profile(object_id, "object")
        self.env = orpy.Environment()
        self.env.Load(toppra_app.utils.expand_and_join(self.db.get_model_dir(), env_dir))
        self.robot = self.env.GetRobot(robot_name)
        assert self.robot is not None

        self.env.SetViewer('qtosg')
        self.env.Load(toppra_app.utils.expand_and_join(self.db.get_model_dir(), obj_profile['rave_model']))
        obj = self.env.GetBodies()[-1]
        self.obj = obj

        attach = self.robot.GetManipulator(link_name)
        if attach is None:
            attach = self.robot.GetLink(link_name)
        assert attach is not None
        self.attach = attach
        T_object_model = obj_profile['T_object_model']
        self.T_link_model = np.dot(T_link_object, T_object_model)

    def run_cmd(self, cmd):
        if "waypoints_npz" in self.traj_profile:
            file_ = np.load(
                os.path.join(self.db.get_trajectory_data_dir(), self.traj_profile['waypoints_npz']))
            path = toppra.SplineInterpolator(file_['t_waypoints'], file_['waypoints'])
            N_waypoints = len(file_['t_waypoints'])
        elif 't_waypoints' in self.traj_profile:
            path = toppra.SplineInterpolator(self.traj_profile['t_waypoints'], self.traj_profile['waypoints'])
            N_waypoints = len(self.traj_profile['t_waypoints'])
        else:
            raise IOError, "Waypoints not found!"

        q_samples = path.eval(np.arange(0, path.get_duration(), 0.01))
        if cmd == "q":
            exit(42)
        elif cmd == "r":
            return True
        elif cmd == "":
            for i, q in enumerate(q_samples):
                self.robot.SetDOFValues(q, range(6))
                T_obj = np.dot(self.attach.GetTransform(), self.T_link_model)
                self.obj.SetTransform(T_obj)
                if self.env.CheckCollision(self.robot):
                    print "Robot is in collision at waypoint {:d}!".format(i)
                time.sleep(0.01)
        else:
            try:
                i = int(cmd) - 1
                self.robot.SetDOFValues(path.waypoints[i], range(6))
                T_obj = np.dot(self.attach.GetTransform(), self.T_link_model)
                self.obj.SetTransform(T_obj)
            except Exception as e:
                print(e)


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="A program for viewing trajectory.")
    parse.add_argument('-t', '--trajectory', help='Input trajectory specification.', required=False, default="suctioncup_traj1")
    parse.add_argument('-e', '--environment', help='Path to the environment.', required=False, default="caged_denso_ft_sensor_suction.env.xml")
    parse.add_argument('-r', '--robot_name', help='', required=False, default="denso")
    parse.add_argument('-m', '--manip_name', help='Active manipulator', required=False, default="denso_suction_cup")
    parse.add_argument('-o', '--object', help='Object specification.', required=False, default="kindlebox_light")
    parse.add_argument('-a', '--attach', help='Name of the link or mnaipulator that the object is attached to.', required=False, default="denso_suction_cup")
    parse.add_argument('-T', '--transform', help='T_link_object', required=False, default="[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 9.080e-3], [0, 0, 0, 1]]")
    args = vars(parse.parse_args())

    view_traj = ViewTrajectory(
        args['trajectory'], args['environment'], args['robot_name'], args['object'], args['attach'], np.array(yaml.load(args['transform']))
    )

    while True:
        cmd = raw_input("[Enter]: play trajectory, [r] to reload, [number] to view waypoint, [q] to quit: ")
        view_traj.run_cmd(cmd)
