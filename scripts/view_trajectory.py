import argparse, yaml, os
import toppra_app, toppra, time
import openravepy as orpy
import numpy as np


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

    env = orpy.Environment()

    view_traj = toppra_app.ViewTrajectory(env,
        args['trajectory'], args['environment'], args['robot_name'], args['object'], args['attach'], np.array(yaml.load(args['transform'])), dt=6.67e-3
    )

    while True:
        cmd = raw_input("[Enter]: play trajectory, [r] to reload, [number] to view waypoint, [q] to quit: ")
        view_traj.run_cmd(cmd)
