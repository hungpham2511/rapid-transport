from ..profile_loading import Database
from ..solidobject import SolidObject
from ..contact import Contact
from ..toppra_constraints import create_object_transporation_constraint

import ftsensorless as ft
import rospy
from denso_control.controllers import JointPositionController
import openravepy as orpy
import numpy as np
import toppra
import os
import time
import hashlib
import matplotlib.pyplot as plt

import cvxpy as cvx

DEFAULT_SCENE_PATH = "/home/hung/git/toppra-object-transport/models/robust-exp.env.xml"
DEFAULT_ROBOT_NAME = "denso"


def plot_trajectory_fig(ts, qs, qds, qdds):
    fig = plt.figure(figsize=[5, 6])
    ax = fig.add_subplot(311)
    ax.plot(ts, qs)
    ax.set_title("Joint position")
    ax = fig.add_subplot(312)
    ax.plot(ts, qds)
    ax.set_title("Joint velocity")
    ax = fig.add_subplot(313)
    ax.plot(ts, qdds)
    ax.set_title("Joint acceleration")
    plt.tight_layout()
    return fig


def get_path_from_trajectory_id(trajectory_id):
    """ Read and return the path/trajectory associated with a trajectory id.
    """
    db = Database()
    traj_profile = db.retrieve_profile(trajectory_id, "trajectory")
    # retrieve and define geometric path
    if "waypoints_npz" in traj_profile:
        file_ = np.load(os.path.join(db.get_trajectory_data_dir(), traj_profile['waypoints_npz']))
        ts_waypoints = file_['t_waypoints']
        waypoints = file_['waypoints']
    elif "t_waypoints" in traj_profile:
        ts_waypoints = traj_profile['t_waypoints']
        waypoints = traj_profile['waypoints']
    else:
        raise IOError, "Waypoints not found in trajectory {:}".format(traj_profile['id'])

    # setup toppra instance
    path = toppra.SplineInterpolator(ts_waypoints, waypoints)
    return path


def main(env, scene_path, robot_name, contact_id, object_id, attach, transform, trajectory_id,
         strategy, slowdown, execute, verbose, safety):
    """ A entry point for performing the ROBUST experiment.

    Parameters
    ----------
    env: 
    """
    if scene_path is None:
        scene_path = DEFAULT_SCENE_PATH
    if robot_name is None:
        robot_name = DEFAULT_ROBOT_NAME
    if env is None:
        env = orpy.Environment()
        env.SetViewer('qtosg')
    else:
        env.Reset()
    if verbose:
        import coloredlogs
        coloredlogs.install(level='DEBUG')
        np.set_printoptions(3)
    else:
        import coloredlogs
        coloredlogs.install(level='INFO')
    env.SetDebugLevel(2)
    env.Load(scene_path)
    robot = env.GetRobot(robot_name)
    manip = robot.SetActiveManipulator(attach)
    arm_indices = manip.GetArmIndices()
    db = Database()

    # load object
    object_profile = db.retrieve_profile(object_id, "object")
    solid_object = SolidObject(robot, attach, transform, object_profile['mass'],
                               np.array(object_profile['local_inertia'], dtype=float),
                               dofindices=arm_indices)
    env.Load("/home/hung/git/toppra-object-transport/models/" + object_profile['rave_model'])
    object_model = env.GetBodies()[-1]
    T_object = manip.GetTransform().dot(transform).dot(object_profile['T_object_model'])
    object_model.SetTransform(T_object)
    robot.Grab(object_model)

    contact = Contact.init_from_profile_id(robot, contact_id)
    contact.g_local = contact.g_local * safety

    # load constraints
    contact_constraint = create_object_transporation_constraint(contact, solid_object)
    contact_constraint.set_discretization_type(0)

    print contact_constraint
    vlim_ = np.r_[robot.GetDOFVelocityLimits()]
    alim_ = np.r_[robot.GetDOFAccelerationLimits()]
    vlim = np.vstack((-vlim_, vlim_)).T
    alim = np.vstack((-alim_, alim_)).T
    pc_velocity = toppra.constraint.JointVelocityConstraint(vlim)
    pc_accel = toppra.constraint.JointAccelerationConstraint(alim)
    pc_accel.set_discretization_type(1)

    # setup toppra instance
    path = get_path_from_trajectory_id(trajectory_id)
    gridpoints = np.linspace(0, 1, 201)

    # solve
    dt = 6.67e-3
    # no strategy, only spline interpolation
    if strategy == "nil":
        print("Run with Strategy==[nil]")
        ts = np.arange(0, path.get_duration(), dt * slowdown)
        qs = path.eval(ts)
        qds = path.evald(ts) * slowdown
        qdds = path.evaldd(ts) * slowdown ** 2
        ts = ts / slowdown
        q_init = path.eval(0)
    elif strategy == "kin_only":
        instance = toppra.algorithm.TOPPRA([pc_velocity, pc_accel], path,
                                           gridpoints=gridpoints, solver_wrapper='hotqpOASES')
        traj_ra, aux_traj = instance.compute_trajectory(0, 0)
        ts = np.arange(0, traj_ra.get_duration(), dt * slowdown)
        qs = traj_ra.eval(ts)
        qds = traj_ra.evald(ts) * slowdown
        qdds = traj_ra.evaldd(ts) * slowdown ** 2
        q_init = traj_ra.eval(0)
        ts = ts / slowdown
    elif strategy == "w_contact":
        instance = toppra.algorithm.TOPPRA([pc_velocity, pc_accel, contact_constraint], path,
                                           gridpoints=gridpoints, solver_wrapper='hotqpOASES')
        traj_ra, aux_traj = instance.compute_trajectory(0, 0)
        ts = np.arange(0, traj_ra.get_duration(), dt * slowdown)
        qs = traj_ra.eval(ts)
        qds = traj_ra.evald(ts) * slowdown
        qdds = traj_ra.evaldd(ts) * slowdown ** 2
        q_init = traj_ra.eval(0)
        ts = ts / slowdown
    elif strategy == "w_contact_jerk":
        instance = toppra.algorithm.TOPPRA([pc_velocity, pc_accel, contact_constraint], path,
                                           gridpoints=gridpoints, solver_wrapper='hotqpOASES')
        _, sd_vec, _ = instance.compute_parameterization(0, 0)
        xs = sd_vec ** 2
        ds = gridpoints[1] - gridpoints[0]
        t0 = time.time()
        N = gridpoints.shape[0] - 1
        sddd_bnd = 20
        # matrix for computing jerk: 
        # Jmat = [-2  1 .....  ]
        #        [ 1 -2  1 ... ]
        #        [ 0  1 -2 1 ..]
        Jmat = np.zeros((N+1, N+1))   
        for i in range(N+1):
            Jmat[i, i] = -2
        for i in range(N):
            Jmat[i + 1, i] = 1
            Jmat[i, i + 1] = 1
        Jmat = Jmat / ds ** 2

        # matrix for differentiation
        Amat = np.zeros((N, N + 1))
        for i in range(N):
            Amat[i, i] = -1
            Amat[i, i + 1] = 1
        Amat = Amat / 2 / ds

        xs_var = cvx.Variable(N + 1)
        x_pprime_var = Jmat * xs_var
        s_dddot_var = cvx.mul_elemwise(np.sqrt(xs) + 1e-5, x_pprime_var)
        residue = cvx.Variable()

        constraints = [xs_var >= 0,
                       xs_var[0] == 0,
                       xs_var[-1] == 0,
                       xs_var <= xs,
                       0 <= residue,
                       s_dddot_var - sddd_bnd <= residue,
                       -sddd_bnd - s_dddot_var <= residue]

        obj = cvx.Minimize(1.0 / N * cvx.norm(xs_var - xs, 1)
                           + 1.0 / N * cvx.norm(Amat.dot(xs) - Amat * xs_var, 1)
                           + 1.0 * residue)
        prob = cvx.Problem(obj, constraints)
        status = prob.solve()

        print "post-processing takes {:f} seconds".format(time.time() - t0)
        print "Problem status {:}".format(status)
        print "residual value: {:f}".format(residue.value)
        xs_new = np.array(xs_var.value).flatten()
        xs_new[0] = 0
        xs_new[-1] = 0

        # Compute trajectory
        ss = np.linspace(0, 1, N + 1)
        ts = [0]
        for i in range(1, N + 1):
            sd_ = (np.sqrt(xs_new[i]) + np.sqrt(xs_new[i - 1])) / 2
            ts.append(ts[-1] + (ss[i] - ss[i - 1]) / sd_)
        q_grid = path.eval(ss)
        # joint_traj = toppra.SplineInterpolator(ts, q_grid, bc_type='natural')
        traj_ra = toppra.SplineInterpolator(ts, q_grid, bc_type='clamped')
        ts = np.arange(0, traj_ra.get_duration(), dt * slowdown)
        qs = traj_ra.eval(ts)
        qds = traj_ra.evald(ts) * slowdown
        qdds = traj_ra.evaldd(ts) * slowdown ** 2
        q_init = traj_ra.eval(0)
        ts = ts / slowdown       

    # log_trajectory_plot
    fig = plot_trajectory_fig(ts, qs, qds, qdds)
    transform_hash = hashlib.md5(str(np.array(transform * 10000, dtype=int))).hexdigest()[:5]
    fig_name = "obj_{1}_{2}-traj_{3}-slow_{4}-strat_{0}".format(
        strategy, object_id, transform_hash, trajectory_id, slowdown)
    fig.savefig("{:}_trajectory.pdf".format(fig_name))

    # execute
    
    if execute:
        n = rospy.init_node("abc")
        joint_controller = JointPositionController("denso")
        ft.rave_utils.move_to_joint_position(
            q_init, joint_controller, robot, dt=dt, require_confirm=True,
            velocity_factor=0.2, acceleration_factor=0.2)
        cmd = raw_input("Execute the trajectory y/[N]?  ")
        if cmd != "y":
            print("Does not execute anything. Exit.")
            exit()
        print("Executing trajectory slowed down by {:f}.".format(slowdown))
        for q in qs:
            t0 = rospy.get_time()
            joint_controller.set_joint_positions(q)
            robot.SetDOFValues(q, range(6))
            t_elapsed = rospy.get_time() - t0
            rospy.sleep(dt - t_elapsed)
        print("Trajectory execution finished! Exit now.")
    else:
        for q in qs:
            robot.SetDOFValues(q, range(6))
            time.sleep(dt)

    return True
