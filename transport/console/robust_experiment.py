from ..profile_loading import Database
from ..solidobject import SolidObject
from ..contact import Contact
from ..toppra_constraints import create_object_transporation_constraint

try:
    import raveutils
    import rospy
    from denso_control.controllers import (JointPositionController,
                                           JointControllerBase,
                                           JointTrajectoryController)
    from object_transport.srv import (ExecTrajectorybCap, ExecTrajectorybCapRequest,
                                      ExecTrajectorybCapResponse)
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
except ImportError as e:
    print e
    pass
import openravepy as orpy
import numpy as np
import toppra
import os
import time
import hashlib
import matplotlib.pyplot as plt
import cvxpy as cvx

_tmp_dir = "/home/hung/.temp.toppra/"

import logging
logger = logging.getLogger(__name__)


def move_to_joint_position(q, joint_controller, robot, planner='birrt',
                           max_planner_iterations=20, max_postprocessing_iterations=40,
                           dt=1.0 / 150, require_confirm=True,
                           velocity_factor=0.3,
                           acceleration_factor=0.3):
    """Move the robot to a desired joint position.

    The routine first plans a collision-free path in the internal
    environment of `robot`. Then if a collision-free path is found, it
    executes and waits for trajectory execution to finish before
    terminating.

    Args:
        q (array): Shape (6,). Goal joint position.
        joint_controller (JointPositionController): A connected joint controller.
        robot (OpenRAVE Robot): The Denso model.
        planner (str, optional): OpenRAVE planner to use.
        max_planner_iterations (int, optional): Number of planning iterations.
        max_postprocessing_iterations (int, optional): Number of post-processing iterations.
        dt (float, optional): Time step.
        velocity_factor (float, optional): Reduction factor.
        acceleration_factor (float, optional): Reduction factor.

    Returns:
        out (bool): True if successfully moves the robot to `q`. Else False.

    """
    q_current = joint_controller.get_joint_positions()
    env = robot.GetEnv()
    vlim_original = robot.GetDOFVelocityLimits()
    alim_original = robot.GetDOFAccelerationLimits()
    robot.SetDOFVelocityLimits(vlim_original * velocity_factor)
    robot.SetDOFAccelerationLimits(alim_original * acceleration_factor)
    with env:
        robot.SetDOFValues(q_current, dofindices=robot.GetActiveManipulator().GetArmIndices())
        rave_planner = orpy.RaveCreatePlanner(env, planner)
        params = orpy.Planner.PlannerParameters()
        robot.SetActiveDOFs(robot.GetActiveManipulator().GetArmIndices())
        params.SetRobotActiveJoints(robot)
        params.SetGoalConfig(q)
        params.SetMaxIterations(max_planner_iterations)
        params.SetPostProcessing(
            'ParabolicSmoother',
            '<_nmaxiterations>{0}</_nmaxiterations>'.format(max_postprocessing_iterations))
        success = rave_planner.InitPlan(robot, params)
        if not success:
            return False
        # Plan a trajectory
        traj = orpy.RaveCreateTrajectory(env, '')
        status = rave_planner.PlanPath(traj)
        if status != orpy.PlannerStatus.HasSolution:
            return False

    if require_confirm:
        raw_input("Trajectory planned with {}! Press [Enter] to move to\n q:= {}".format(planner, q))

    robot.SetDOFVelocityLimits(vlim_original)
    robot.SetDOFAccelerationLimits(alim_original)

    if isinstance(joint_controller, JointPositionController):
        # Execute the trajectory
        spec = traj.GetConfigurationSpecification()
        time = 0.
        rate = rospy.Rate(int(1/dt))
        while time < traj.GetDuration():
            trajdata = traj.Sample(time)
            values = spec.ExtractJointValues(trajdata, robot,
                                             robot.GetActiveManipulator().GetArmIndices(), 0)
            # Send joint signal
            joint_controller.set_joint_positions(values)
            robot.SetDOFValues(values, dofindices=robot.GetActiveManipulator().GetArmIndices())
            time += dt
            rate.sleep()
        return True
    else:
        raise ValueError('controller [{}] not supported!'.format(joint_controller))


def ros_traj_from_rave(robot, traj):
    ros_traj = JointTrajectory()
    ros_traj.joint_names = ["j1", "j2", "j3", "j4", "j5", "j6"]
    spec = traj.GetConfigurationSpecification()
    time = 0
    while time < traj.GetDuration():
        trajdata = traj.Sample(time)
        values = spec.ExtractJointValues(trajdata, robot,
                                         robot.GetActiveManipulator().GetArmIndices(), 0)
        pt = JointTrajectoryPoint()
        pt.positions = values
        pt.time_from_start = rospy.Duration(time)
        ros_traj.points.append(pt)
        time += 8e-3
    return ros_traj


def get_shaper(cmd):
    T = 6.67e-3
    K = 0.861
    Td = 0.054 * 2
    if cmd == "nil":
        input_shaper = np.ones(1)
    elif cmd == "ZV":
        Nd = int(Td / T)
        input_shaper = np.zeros(Nd / 2)
        input_shaper[0] = 1 / (1 + K)
        input_shaper[-1] = K / (1 + K)
    elif cmd == "ZVD":
        Nd = int(Td / T)
        input_shaper = np.zeros(Nd)
        input_shaper[0] = 1.0 / (1 + 2 * K + K ** 2)
        input_shaper[Nd / 2] = 2 * K / (1 + 2 * K + K ** 2)
        input_shaper[Nd - 1] = K ** 2 / (1 + 2 * K + K ** 2)
    else:
        raise NotImplementedError
    return input_shaper


def shape_trajectory(qs_unshaped, shaper):
    """ Shape input trajectory with the initialized shaper.
    """
    if shaper is None:
        return qs_unshaped
    else:
        qs_shaped = []
        for i in range(6):
            qi = convole_signal(qs_unshaped[:, i], shaper, qs_unshaped[0, i])
            qs_shaped.append(qi)
        qs_shaped = np.array(qs_shaped).T
        return qs_shaped


def preview(data_dict):
    q_shaped_arr = data_dict['shaped_trajectory']
    transform, strategy, object_id, trajectory_id, slowdown, contact_id = data_dict["problem_data"]
    data, gridpoints = data_dict["profile_data"]
    t_arr, q_arr, qd_arr, qdd_arr = data_dict["unshaped_trajectory"]
    vlim_, alim_ = data_dict["kin_limits"]

    # Plot profile
    transform_hash = hashlib.md5(str(np.array(transform * 10000, dtype=int))).hexdigest()[:5]
    fig_name = "obj_{1}_{2}-traj_{3}-slow_{4}-strat_{0}-contact{5}".format(
        strategy, object_id, transform_hash, trajectory_id, slowdown, contact_id)
    if data is not None:
        sdd_grid, sd_grid, v_grid, K = data
        fig = plt.figure()
        if sd_grid is not None:
            plt.plot(gridpoints, sd_grid ** 2)
        plt.plot(gridpoints, K[:, 0])
        plt.plot(gridpoints, K[:, 1])
        plt.xlim(gridpoints[0] - 0.01, gridpoints[-1] + 0.01)
        fig.savefig(os.path.join(_tmp_dir, "{:}_profile.pdf".format(fig_name)))
        plt.show()

    # log_trajectory_plot
    fig, axs = plt.subplots(2, 2, figsize=[5, 6], sharex=True)
    axs[0, 0].plot(t_arr, q_arr)
    axs[0, 0].set_title("(unshaped) Joint position")
    axs[1, 0].plot(8e-3 * np.arange(q_shaped_arr.shape[0]), q_shaped_arr)
    axs[1, 0].set_title("(shaped) Joint position")
    axs[0, 1].plot(t_arr, qd_arr)
    axs[0, 1].set_title("Joint velocity")
    axs[1, 1].plot(t_arr, qdd_arr)
    axs[1, 1].set_title("Joint acceleration")

    for i in range(6):
        axs[1, 1].plot((t_arr[0], t_arr[-1]), [alim_[i], alim_[i]], '--', color="C" + str(i))
        axs[1, 1].plot((t_arr[0], t_arr[-1]), [-alim_[i], -alim_[i]], '--', color="C" + str(i))

        axs[0, 1].plot((t_arr[0], t_arr[-1]), [vlim_[i], vlim_[i]], '--', color="C" + str(i))
        axs[0, 1].plot((t_arr[0], t_arr[-1]), [-vlim_[i], -vlim_[i]], '--', color="C" + str(i))

    fig.savefig(os.path.join(_tmp_dir, "{:}_trajectory.pdf".format(fig_name)))
    plt.show()


def convole_signal(x_arr, shaper, x_init):
    """Convole `x_arr` with `shaper`. 

    Note that convolution is not done directly on `x_arr` but on
    another array `x_arr'` that is defined below:

    x_arr'[i < 0] = x_init
    x_arr'[0:N] = x_arr
    x_arr'[N:] = x_arr[N-1]

    where N is the length of x_arr.

    The reason is that this array is input to a simulator which takes
    the last value of its input as a constant, and also has initial
    condition x_init.

    """
    N = x_arr.shape[0]
    M = shaper.shape[0]
    x_arr_conv = np.zeros(N + 2 * M - 2)
    x_arr_conv[:M - 1] = x_init
    x_arr_conv[M - 1: M + N - 1] = x_arr
    x_arr_conv[M + N - 1:] = x_arr[-1]
    x_arr_conv = np.convolve(x_arr_conv, shaper, mode='full')[M - 1: 2 * M + N - 2]
    return x_arr_conv


class TrajectoryRunner(object):
    def __init__(self, robot, execute, dt=8e-3):
        self._execute = execute
        self._robot = robot
        self._dt = dt
        # 1: position-controller
        if execute == 1:
            self._joint_controller = JointPositionController("denso")
        # 2: trajectory-controller (via bCapServier)
        elif execute == 2:
            assert False, "This option is depreciated"
            # self._trajectory_controller = bCap_JointTrajectoryController("bcap", 2)
        # 3: trajectory-controller (via denso_control/trajectory controller)
        elif execute == 3:
            self._trajectory_controller = JointTrajectoryController("denso")
        else:
            logger.info("No hardware controller loaded.")

    def execute_traj(self, q_arr):
        """ The entry point of this controller.

        Parameters
        ----------
        q_arr: array
            Waypoints to send to the robots.
        """
        if self._execute == 1:
            return self.execute_traj_position(q_arr)
        elif self._execute == 2:
            return self.execute_traj_trajectory(q_arr)
        elif self._execute == 3:
            return self.execute_traj_trajectory(q_arr)
        else:
            return self.execute_traj_rave(q_arr)

    def execute_traj_trajectory(self, q_arr):
        """ Execute trajectory using the stored trajectory controller.
        """
        q_current = self._trajectory_controller.get_joint_positions()
        self._robot.SetActiveDOFValues(q_current)
        traj0 = raveutils.planning.plan_to_joint_configuration(self._robot, q_arr[0])
        traj0_ros = ros_traj_from_rave(self._robot, traj0)
        traj1_ros = JointTrajectory()
        for i, q in enumerate(q_arr):
            pt = JointTrajectoryPoint()
            pt.positions = q
            pt.time_from_start = rospy.Duration(8e-3 * i)
            traj1_ros.points.append(pt)

        # execute trajectory
        cmd = raw_input("Move to starting configuration y/[N]?  ")
        if cmd != "y":
            print("Does not execute anything. Exit.")
            exit()
        self._trajectory_controller.set_trajectory(traj0_ros)
        self._trajectory_controller.start()
        self._trajectory_controller.wait()
        cmd = raw_input("Execute transport trajectory y/[N]?  ")
        if cmd != "y":
            print("Does not execute anything. Exit.")
            exit()
        self._trajectory_controller.set_trajectory(traj1_ros)
        self._trajectory_controller.start()
        self._trajectory_controller.wait()

    def execute_traj_rave(self, q_arr):
        self._robot.SetDOFValues(q_arr[0], range(6))
        cmd = raw_input("[Enter] to execute path")
        rate = rospy.Rate(125)
        for q in q_arr:
            self._robot.SetDOFValues(q, range(6))
            rate.sleep()

    def execute_traj_position(self, q_arr):
        """Execute a trajectory using the JointPositioncontroller.

        """
        # Move to the starting conf
        cmd = raw_input("Move to starting configuration y/[N]?  ")
        if cmd != "y":
            print("Does not execute anything. Exit.")
            exit()
        if self._execute:
            q_init = q_arr[0]
            move_to_joint_position(
                q_init, self._joint_controller, self._robot, dt=self._dt, require_confirm=True,
                velocity_factor=0.2, acceleration_factor=0.2)
        else:
            q_init = q_arr[0]
            self._robot.SetActiveDOFValues(q_init)

        # Execute starting trajectory
        cmd = raw_input("Run traj? y/[N]?  ")
        if cmd != "y":
            print("Does not execute anything. Exit.")
            exit()
        rate = rospy.Rate(125)
        if self._execute:
            for q in q_arr:
                self._joint_controller.set_joint_positions(q)
                self._robot.SetDOFValues(q, range(6))
                rate.sleep()
        else:
            for q in q_arr:
                self._robot.SetDOFValues(q, range(6))
                rate.sleep()
        print("Trajectory execution finished! Exit now.")


def get_path_from_trajectory_id(trajectory_id):
    """ Read and return the path/trajectory by id.
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


def get_list_of_paths_from_string(robot, input_string):
    """ Return a lists of paths from a command string.
    """
    # input string is a single integer: e.g. "1" or "2"
    try:
        seed = int(input_string)
        return [gen_path(robot, seed)]
    except ValueError:
        # input string is not a single integer. e.g.  "1"
        pass

    # input string is multiple integers separated by semi-colons: e.g. "1,2,3"
    try:
        seed_list = map(int, input_string.split(","))
        return [gen_path(robot, seed) for seed in seed_list]
    except ValueError:
        # input string is not sequence of integer separated by colons. e.g. "1,2,3"
        pass

    try:
        seed0, seed1 = map(int, input_string.split("-"))
        return [gen_path(robot, seed) for seed in range(seed0, seed1)]
    except ValueError:
        # input string is two integers separated by dash. e.g. "1-3"
        pass

    # input string is an trajectory id
    return [get_path_from_trajectory_id(input_string)]


pos1 = np.r_[0.5, -0.3, 0.3]
pos2 = np.r_[0.5, 0.3, 0.4]
size = np.r_[0.2, 0.2, 0.6]
q_nominal1 = np.r_[-0.3, 0.9, 0.9, 0, 0, 0]
q_nominal2 = np.r_[ 0.3, 0.9, 0.9, 0, 0, 0]


def gen_path(robot, seed, max_ppiters=60, max_iters=100):
    """ Generate a collision-free path from a random integer seed.
    """
    np.random.seed(seed)
    q_cur = robot.GetActiveDOFValues()
    robot.SetActiveDOFValues(np.zeros(6))
    all_grabbed = robot.GetGrabbed()
    while True:
        with robot:
            robot.ReleaseAllGrabbed()
            manip = robot.GetActiveManipulator()
            
            # randomize pose1 and pose2
            xy1 = (np.random.rand(3) - 0.5) * size + pos1
            rot1 = (np.random.rand() - 0.5) * np.pi * 0.9
            xy2 = (np.random.rand(3) - 0.5) * size + pos2
            rot2 = (np.random.rand() - 0.5) * np.pi * 0.9
            
            # nominal transform of the end-effector
            T_nominal = np.array([[1.0, 0, 0, 0],
                                  [0, -1.0, 0, 0],
                                  [0, 0, -1.0, 0],
                                  [0, 0, 0, 1.0]])

            T_manip_1 = np.dot(T_nominal, orpy.matrixFromAxisAngle([0, 0, rot1]))
            T_manip_1[:3, 3] = xy1
            T_manip_2 = np.dot(T_nominal, orpy.matrixFromAxisAngle([0, 0, rot2]))
            T_manip_2[:3, 3] = xy2

            qgoals1 = manip.FindIKSolutions(T_manip_1, orpy.IkFilterOptions.CheckEnvCollisions)
            qgoals2 = manip.FindIKSolutions(T_manip_2, orpy.IkFilterOptions.CheckEnvCollisions)

            # Kinematic infeasible
            if len(qgoals1) == 0 or len(qgoals2) == 0:
                continue

            # select configurations that are closest to the nominal configurations
            qgoal1 = qgoals1[np.argmin(np.linalg.norm(qgoals1 - q_nominal1, 1))]
            qgoal2 = qgoals2[np.argmin(np.linalg.norm(qgoals2 - q_nominal2, 1))]

            # weird configurations, skip
            if qgoal1[1] < 0 or qgoal1[2] < 0 or qgoal2[1] < 0 or qgoal2[2] < 0:
                continue

        for body in all_grabbed:
            robot.Grab(body)

        robot.SetActiveDOFValues(qgoal2)

        # Plan trajectory
        traj0 = raveutils.planning.plan_to_joint_configuration(
            robot, qgoal1, max_ppiters=max_ppiters, max_iters=max_iters)

        if traj0 is None:
            continue

        # filter trajectory by angle
        with robot:
            spec = traj0.GetConfigurationSpecification()
            waypoints = []
            ss_waypoints = []
            skip = False
            for i in range(traj0.GetNumWaypoints()):
                data = traj0.GetWaypoint(i)
                dt = spec.ExtractDeltaTime(data)
                if dt > 1e-5 or len(waypoints) == 0:  # If delta is too small, skip it.
                    if len(ss_waypoints) == 0:
                        ss_waypoints.append(0)
                    else:
                        ss_waypoints.append(ss_waypoints[-1] + dt)
                    q = spec.ExtractJointValues(data, robot, robot.GetActiveDOFIndices())
                    waypoints.append(q)
                    robot.SetActiveDOFValues(q)
                    z_dir = manip.GetTransform()[:3, 2]
                    angle = np.arccos(np.dot(z_dir, [0, 0, -1]))
                    if angle > 1.0:
                        skip = True
            if skip:
                continue
            ss_waypoints = np.array(ss_waypoints) / ss_waypoints[-1]
            path = toppra.SplineInterpolator(ss_waypoints, waypoints)

        return path


def main(env, scene_path, robot_name, contact_id, object_id, attach, transform, trajectory_id, strategy, slowdown, execute, verbose, solver, alim_scale=1.0, vlim_scale=1.0, safety=1.0):
    """Entry point to robust experiment.

    Parameters
    ----------
    env :  OpenRAVE Environment or None
    scene_path : str
        Path to an OpenRAVE scene.
    robot_name : str
        Name of the robot.
    contact_id : str
    object_id : str
    attach : str
    transform : (4,4)array
    trajectory_id : str
    strategy : str
    slowdown : float
    execute : bool
    verbose : bool
    solver : str
    vlim_scale: float, optional
        Scale the robot velocity limits accordingly.
        vlim_ = vlim_ * kin_scale
    alim_scale: float, optional
        Scale the robot velocity limits accordingly.
        alim_ = alim_ * kin_scale
    safety : float
        Effect not well understood, do not used.

    """
    # setup
    try:
        n = rospy.init_node("robust_experiment")
    except rospy.ROSException as e:
        print e
    if env is None:
        env = orpy.Environment()
        env.SetViewer('qtosg')
    else:
        env.Reset()
        env.SetViewer('qtosg')
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
    iktype = orpy.IkParameterization.Type.Transform6D
    ikmodel = orpy.databases.inversekinematics.InverseKinematicsModel(robot, iktype=iktype)
    if not ikmodel.load():
        print 'Generating IKFast {0}. It will take few minutes...'.format(iktype.name)
        ikmodel.autogenerate()
        print 'IKFast {0} has been successfully generated'.format(iktype.name)
    arm_indices = manip.GetArmIndices()
    db = Database()
    print("... OpenRAVE and Database loaded")

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

    # load contact
    contact = Contact.init_from_profile_id(robot, contact_id)
    contact.g_local = contact.g_local * safety

    # load constraints
    contact_constraint = create_object_transporation_constraint(contact, solid_object)
    contact_constraint.set_discretization_type(1)
    print("... contact stability constraint formed")
    print contact_constraint

    # Set velocity/accel scaling
    robot.SetDOFVelocityLimits(robot.GetDOFVelocityLimits() * vlim_scale)
    robot.SetDOFAccelerationLimits(robot.GetDOFAccelerationLimits() * alim_scale)

    vlim_ = np.r_[robot.GetDOFVelocityLimits()]
    alim_ = np.r_[robot.GetDOFAccelerationLimits()]
    vlim = np.vstack((-vlim_, vlim_)).T
    alim = np.vstack((-alim_, alim_)).T
    pc_velocity = toppra.constraint.JointVelocityConstraint(vlim)
    pc_accel = toppra.constraint.JointAccelerationConstraint(alim)
    pc_accel.set_discretization_type(1)

    traj_controller = TrajectoryRunner(robot, execute, dt=8e-3)

    # setup toppra instance
    print("... {:d} path(s) loaded. start running experiment.")
    list_of_paths = get_list_of_paths_from_string(robot, trajectory_id)
    for path in list_of_paths:
        gridpoints = np.linspace(0, 1, 101)

        # solve
        dt = 8e-3
        data = None
        fail = False
        # no strategy, only spline interpolation
        if strategy == "nil":
            print("Run with Strategy==[nil]")
            path = toppra.SplineInterpolator(path.ss_waypoints, path.waypoints, bc_type='clamped')
            ts = np.arange(0, path.get_duration(), dt * slowdown)
            qs = path.eval(ts)
            qds = path.evald(ts) * slowdown
            qdds = path.evaldd(ts) * slowdown ** 2
            ts = ts / slowdown
            q_init = path.eval(0)
        # parametrize considering kinematics constraint only
        elif strategy == "kin_only":
            t0 = rospy.get_time()
            instance = toppra.algorithm.TOPPRA([pc_velocity, pc_accel], path,
                                               gridpoints=gridpoints, solver_wrapper=solver)
            traj_ra, aux_traj, data = instance.compute_trajectory(0, 0, return_profile=True)
            ts = np.arange(0, traj_ra.get_duration(), dt * slowdown)
            qs = traj_ra.eval(ts)
            qds = traj_ra.evald(ts) * slowdown
            qdds = traj_ra.evaldd(ts) * slowdown ** 2
            q_init = traj_ra.eval(0)
            ts = ts / slowdown
            t0a = rospy.get_time()
        # parameterization considering contact stability and kinematic constraints
        elif strategy == "w_contact":
            t0 = rospy.get_time()
            instance = toppra.algorithm.TOPPRA([pc_velocity, pc_accel, contact_constraint], path,
                                               gridpoints=gridpoints, solver_wrapper=solver)
            traj_ra, aux_traj, data = instance.compute_trajectory(0, 0, return_profile=True)
            if traj_ra is not None:
                ts = np.arange(0, traj_ra.get_duration(), dt * slowdown)
                qs = traj_ra.eval(ts)
                qds = traj_ra.evald(ts) * slowdown
                qdds = traj_ra.evaldd(ts) * slowdown ** 2
                q_init = traj_ra.eval(0)
                ts = ts / slowdown
            else:
                print("Parameterization fails!")
                fail = True
            t0a = rospy.get_time()
        # parameterization considering contact stability, kinematic constraints and jerk contraints
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

        if fail:
            print("... Parameterization unsuccessful!")
            return False
        else:
            print("... Parameterization successful! \n"
                  ".... Trajectory duration         {:.3f} sec\n"
                  ".... Total Parametrization time  {:.3f} sec" .format(ts[-1], t0a - t0))

        # Trajectory shaping
        shaper = get_shaper("nil")
        # note: trajectory shaping can be used to remove the object
        # natural vibration. Try uncomment the below line to see if it
        # works for your application.
        # shaper = get_shaper("ZVD")
        q_arr = shape_trajectory(qs, shaper)
        # preview stuffs
        preview({"unshaped_trajectory": (ts, qs, qds, qdds),
                 "profile_data": (data, gridpoints),
                 "shaped_trajectory": q_arr,
                 "problem_data": (transform, strategy, object_id, trajectory_id, slowdown, contact_id),
                 "kin_limits": (vlim_, alim_)
        })

        # execute the trajectory
        # set robot velocity and acceleration rate down, so that paths
        # planned online are safe.
        robot.SetDOFVelocityLimits(vlim_ * 0.3)
        robot.SetDOFAccelerationLimits(alim_ * 0.3)
        traj_controller.execute_traj(q_arr)
    return True
