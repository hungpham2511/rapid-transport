import openravepy as orpy
import time
import toppra
import numpy as np
import yaml
import logging

try:
    import raveutils
    import rospy
    from denso_control.controllers import JointPositionController, JointTrajectoryController
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
except ImportError:
    pass

from ..utils import expand_and_join, setup_logging
from ..profile_loading import Database
from ..solidobject import SolidObject
from ..toppra_constraints import create_object_transporation_constraint

logger = logging.getLogger(__name__)
SOLVER = 'seidel'


def main(*args, **kwargs):
    demo = PickAndPlaceDemo(*args, **kwargs)
    demo.view()
    logger.info("Starting demo..")
    success = demo.run()
    if success:
        logger.info("Pick and place demo terminates successfully!")
    else:
        logger.fatal("Pick and place demo fails!")


def plan_to_manip_transform(robot, T_ee_start, q_nominal, max_ppiters=60, max_iters=100):
    """Plan a trajectory from the robot's current configuration to a new
    configuration at which the active manipulator has transform
    `T_ee_start`.

    Parameters
    ----------
    robot: openravepy.Robot
    T_ee_start: (4,4)array
        Desired active manipulator transform.
    q_nominal: (dof,)array
        A desired/preferred configuration to achieve as the goal configuration.
    max_ppiters: int
        Shortcutting iterations.
    max_iters: int
        Planning iterations.

    Returns
    -------
    traj: openravepy.Trajectory

    """
    # Release object to find final goal configuration without collision.
    all_grabbed = robot.GetGrabbed()
    robot.ReleaseAllGrabbed()
    with robot:
        manip = robot.GetActiveManipulator()

        # Find a good final configuration
        qgoals = manip.FindIKSolutions(T_ee_start, orpy.IkFilterOptions.CheckEnvCollisions)
        qgoal = None
        dist = 10000
        for q_ in qgoals:
            dist_ = np.linalg.norm(q_ - q_nominal)
            if dist_ < dist:
                qgoal = q_
                dist = dist_
        if qgoal is None:
            logger.fatal("Unable to find a collision free solution.")
            cmd = raw_input("[Enter] to cont, [i] for ipdb")
            if cmd == "i":
                import ipdb; ipdb.set_trace()
            return None
    for body in all_grabbed:
        robot.Grab(body)
    # Plan trajectory to that point
    traj0 = raveutils.planning.plan_to_joint_configuration(
        robot, qgoal, max_ppiters=max_ppiters, max_iters=max_iters)
    return traj0


class PickAndPlaceDemo(object):
    """A pick-and-place demo.

    Parameters
    ----------
    load_path: str
        Load path to scenario.
    env: optional
        OpenRAVE Environment. If env is None, create a new environment.
    verbose: bool, optional
    execute: int, optional
        If equals 0, only run the trajectories in openrave simulation.
        If equals 3, send to the real hardware using JointTrajectoryController .

    dt: float, optional
        Control sampling time in sending commands to the robots.
    slowdown: float, optional

        A factor to slowdown execution. A value of 0.5 means slowing
        down the final trajectory uniformly so that the old duration
        equals 0.5 times the new duration..

        A value of 1.0 means execute at computed speed.

    """
    def __init__(self, load_path=None, env=None, verbose=False, execute=0, dt=8e-3, slowdown=1.0):
        assert load_path is not None, "A scenario must be supplied"
        self.verbose = verbose
        self.execute = execute
        self.slowdown = slowdown
        if self.verbose:
            setup_logging(level="DEBUG")
            toppra.utils.setup_logging(level="DEBUG")
        else:
            setup_logging(level="INFO")
            toppra.utils.setup_logging(level="INFO")
        db = Database()
        _scenario_dir = expand_and_join(db.get_data_dir(), load_path)
        with open(_scenario_dir) as f:
            self._scenario = yaml.load(f.read())
        _world_dir = expand_and_join(db.get_model_dir(), self._scenario['world_xml_dir'])
        if env is None:
            self._env = orpy.Environment()
        else:
            self._env = env
            self._env.Reset()
        self._env.SetDebugLevel(3)  # Less verbose debug
        self._env.Load(_world_dir)
        self._robot = self._env.GetRobot(self._scenario['robot'])
        self._robot.SetDOFVelocityLimits(self.slowdown * self._robot.GetDOFVelocityLimits())
        self._robot.SetDOFAccelerationLimits(self.slowdown * self._robot.GetDOFAccelerationLimits())
        self._objects = []
        n = rospy.init_node("pick_and_place_planner")
        if self.execute == 0:
            logger.info("Only run in OpenRAVE.")
        elif self.execute == 3:
            logger.info("Be CAREFUL! Will send command to the real Denso!")
            self._trajectory_controller = JointTrajectoryController("denso")
            self._robot.SetActiveDOFValues(self._trajectory_controller.get_joint_positions())
        else:
            logger.error("Other EXECUTE mode not supported.")

        self._dt = dt
        # Load all objects to openRave
        for obj_d in self._scenario['objects']:
            obj = SolidObject.init_from_dict(self._robot, obj_d)
            self._objects.append(obj)
            obj.load_to_env(obj_d['T_start'])
            self._robot.SetActiveManipulator(obj_d['object_attach_to'])
            # Generate IKFast for each active manipulator
            iktype = orpy.IkParameterization.Type.Transform6D
            ikmodel = orpy.databases.inversekinematics.InverseKinematicsModel(self._robot, iktype=iktype)
            if not ikmodel.load():
                print 'Generating IKFast {0}. It will take few minutes...'.format(iktype.name)
                ikmodel.autogenerate()
                print 'IKFast {0} has been successfully generated'.format(iktype.name)
            rave_obj = self._env.GetKinBody(obj_d['name'])
            if self._env.CheckCollision(rave_obj):
                logger.fatal("Object {:} is in collision.".format(rave_obj.GetName()))
                self.view()
                self.check_continue()

    def view(self):
        res = self._env.SetViewer('qtosg')
        time.sleep(0.5)
        return True

    def get_env(self):
        return self._env

    def get_robot(self):
        return self._robot

    def get_object(self, name):
        obj = None
        for obj_ in self._objects:
            if obj_.get_name() == name:
                return obj_
        return obj

    def get_object_dict(self, name):
        for entry_ in self._scenario['objects']:
            if entry_['name'] == name:
                return entry_
        return None

    def get_qstart(self):
        return self._robot.GetActiveDOFValues()

    def verify_transform(self, manip, T_ee_start):
        "Check if transform `T_ee_start` can be reached with manipulator `manip`."
        fail = False
        qstart_col = manip.FindIKSolution(T_ee_start, orpy.IkFilterOptions.CheckEnvCollisions)
        qstart_nocol = manip.FindIKSolution(T_ee_start, orpy.IkFilterOptions.IgnoreEndEffectorCollisions)
        if qstart_col is None:
            fail = True
            logger.fatal("Unable to find a collision-free configuration to reach target transform.")
            if qstart_nocol is None:
                logger.fatal("Reason: kinematic infeasibility.")
                cmd = raw_input("[Enter] to continue/exit. [i] to drop to Ipython.")
                if cmd == "i":
                    import ipdb; ipdb.set_trace()
            else:
                logger.fatal("Reason: collision (kinematically feasible).")
                logger.fatal("Collision might be due to grabbed object only, which is normal. Check viewer.")
                with self._robot:
                    self._robot.SetActiveDOFValues(qstart_nocol)
                    self._env.UpdatePublishedBodies()
                    logger.fatal("Jnt: {:}".format(qstart_nocol / np.pi * 180))
                    cmd = raw_input("[Enter] to continue/exit. [i] to drop to Ipython.")
                    if cmd == "i":
                        import ipdb; ipdb.set_trace()
        return fail

    def check_trajectory_collision(self, traj):
        """ Check collision for a whole trajectory.
        """
        spec = traj.GetConfigurationSpecification()
        in_collision = False
        for i in range(traj.GetNumWaypoints()):
            data = traj.GetWaypoint(i)
            q = spec.ExtractJointValues(data, self._robot, range(6), 0)
            with self._robot:
                self._robot.SetActiveDOFValues(q)
                logger.debug("In collision = {:}".format(self._env.CheckCollision(self._robot)))
                if self._env.CheckCollision(self._robot):
                    in_collision = True
        if in_collision:
            logger.fatal("Robot is in collision!")
        return in_collision

    def extract_waypoints(self, trajectory):
        """ Extract waypoints from an OpenRAVE trajectory.

        Parameters
        ----------
        trajectory: OpenRAVE.Trajectory
        """
        spec = trajectory.GetConfigurationSpecification()
        waypoints = []
        ss_waypoints = []
        for i in range(trajectory.GetNumWaypoints()):
            data = trajectory.GetWaypoint(i)
            dt = spec.ExtractDeltaTime(data)
            if dt > 1e-5 or len(waypoints) == 0:
                if len(ss_waypoints) == 0:
                    ss_waypoints.append(0)
                else:
                    ss_waypoints.append(ss_waypoints[-1] + dt)
                q = spec.ExtractJointValues(data, self._robot, range(6), 0)
                waypoints.append(q)
        return np.array(waypoints), np.array(ss_waypoints)

    def extract_ros_traj(self, trajectory):
        """ Return a ROS trajectory from an OpenRAVE trajectory.
        """
        pass

    def execute_trajectory(self, trajectory):
        """ Execute trajectory on the robot hardware.

        If execute is 3 publish data and send to the robot.

        Parameters
        ----------
        trajectory: OpenRAVE.trajectory

        """
        if trajectory is None:
            return False

        if self.execute == 0:
            self._robot.GetController().SetPath(trajectory)
            self._robot.WaitForController(0)
            return True

        elif self.execute == 3:
            spec = trajectory.GetConfigurationSpecification()
            trajectory_ros = JointTrajectory()
            duration = trajectory.GetDuration()
            for t in np.arange(0, duration, self._dt):
                data = trajectory.Sample(t)
                q = spec.ExtractJointValues(data, self._robot, range(6), 0)
                pt = JointTrajectoryPoint()
                pt.positions = q
                pt.time_from_start = rospy.Duration(t)
                trajectory_ros.points.append(pt)
            self._trajectory_controller.set_trajectory(trajectory_ros)
            self._trajectory_controller.start()
            self._robot.GetController().SetPath(trajectory)

            self._robot.WaitForController(0)
            self._trajectory_controller.wait()

            return True

    def run(self, offset=0.01):
        """Run the demo.

        For each object, the robot first move to a pose that is
        directly on top of it. Afterward, it moves down.  Next, it
        grabs the object to be transported. Finally, it moves to
        another configuration.

        Note: planning is done using rrt, not constrained rrt. The
        later is far too inefficient to be of any use. An even more
        critical issue with constrained rrt is that its path is too
        jerky, failing most smoothing attempts.

        Args:
            offset: (float, optional) Approach distance.
        """
        fail = False
        # Start planning/control loop:
        # Every pick cycle follows the same procedure:
        # 1. APPROACH: visit a configuration that is directly on top of the object to pick
        # 2. REACH:    move down to make contact w the object
        # 3. APPROACH: visit the same configuration as 1.
        # 4. TRANSPORT: visit the goal configuration.
        for obj_dict in self._scenario['objects']:
            t0 = rospy.get_time()
            # Basic setup
            manip_name = obj_dict["object_attach_to"]
            manip = self.get_robot().SetActiveManipulator(manip_name)
            Tstart = np.array(obj_dict['T_start'])  # object's transform
            Tgoal = np.array(obj_dict['T_goal'])
            T_ee_start = np.dot(Tstart, self.get_object(obj_dict['name']).get_T_object_link())
            self.verify_transform(manip, T_ee_start)
            T_ee_top = np.array(T_ee_start)
            try:
                T_ee_top[:3, 3] -= obj_dict['offset'] * T_ee_top[:3, 2]
            except:
                T_ee_top[:3, 3] -= offset * T_ee_top[:3, 2]
            q_nominal = np.r_[-0.3, 0.9, 0.9, 0, 0, 0]

            t1 = rospy.get_time()
            # 1. APPROACH
            logger.info("Plan path to APPROACH")
            traj0 = plan_to_manip_transform(self._robot, T_ee_top, q_nominal, max_ppiters=200, max_iters=100)
            t1a = rospy.get_time()
            self.check_continue()
            fail = not self.execute_trajectory(traj0)
            self.get_robot().WaitForController(0)

            # 2. REACH: Move a "short" trajectory to reach the object
            logger.info("Plan path to REACH")
            t2 = rospy.get_time()
            traj0b = plan_to_manip_transform(self._robot, T_ee_start, q_nominal, max_ppiters=200, max_iters=100)
            t2a = rospy.get_time()
            self.check_continue()
            fail = not self.execute_trajectory(traj0b)
            self.get_robot().WaitForController(0)
            with self._env:
                self._robot.Grab(self.get_env().GetKinBody(obj_dict['name']))
            logger.info("Grabbing the object. Continue moving in 0.3 sec.")

            # 3+4: APPROACH+TRANSPORT: Plan two trajectories, one
            # trajectory to reach the REACH position, another
            # trajectory to reach the GOAL position. Merge them, then
            # execute.
            t3 = rospy.get_time()
            logger.info("Plan path to GOAL")
            ## 1st trajectory
            traj0c = plan_to_manip_transform(self._robot, T_ee_top, q_nominal, max_ppiters=1, max_iters=100)
            traj0c_waypoints, traj0c_ss = self.extract_waypoints(traj0c)
            T_ee_goal = np.dot(Tgoal, self.get_object(obj_dict['name']).get_T_object_link())
            self.verify_transform(manip, T_ee_goal)
            q_nominal = np.r_[0.3, 0.9, 0.9, 0, 0, 0]
            ## 2nd trajectory
            self._robot.SetActiveDOFValues(traj0c_waypoints[-1])
            traj1_transport = plan_to_manip_transform(self._robot, T_ee_goal, q_nominal, max_ppiters=1, max_iters=100)
            self._robot.SetActiveDOFValues(traj0c_waypoints[0])
            traj1_transport_waypoints, traj1_transport_ss = self.extract_waypoints(traj1_transport)
            ## concatenate the two trajectories
            traj2_waypoints = np.vstack((traj0c_waypoints, traj1_transport_waypoints[1:]))

            ## retime
            traj2_ss = np.hstack((traj0c_ss, traj0c_ss[-1] + traj1_transport_ss[1:]))
            traj2_ss[:] = traj2_ss / traj2_ss[-1]
            
            traj2_rave = orpy.RaveCreateTrajectory(self._env, "")
            spec = self._robot.GetActiveConfigurationSpecification()
            traj2_rave.Init(spec)
            for p in traj2_waypoints:
                traj2_rave.Insert(traj2_rave.GetNumWaypoints(), p)
            t3a = rospy.get_time()
            planner = orpy.RaveCreatePlanner(self._env, "ParabolicSmoother")
            params = orpy.Planner.PlannerParameters()
            params.SetRobotActiveJoints(self._robot)
            params.SetMaxIterations(100)
            params.SetPostProcessing('', '')
            success = planner.InitPlan(self._robot, params)
            status = planner.PlanPath(traj2_rave)
            if not success or not status:
                logger.fatal("[Plan Transport Path] Init status: {1:}, Plan status: {0:}. "
                             "Use traj2_rave directly.".format(status, success))
            t3b = rospy.get_time()

            # Retime the transport trajectory and execute it
            logger.debug("Original traj nb waypoints: {:d}".format(traj1_transport.GetNumWaypoints()))
            logger.debug("Retime using toppra.")
            t4 = rospy.get_time()
            contact = self.get_object(obj_dict['name']).get_contact()
            contact_constraint = create_object_transporation_constraint(contact, self.get_object(obj_dict['name']))
            contact_constraint.set_discretization_type(1)
            traj2_retimed = toppra.retime_active_joints_kinematics(traj2_rave, self._robot, additional_constraints=[contact_constraint], solver_wrapper=SOLVER, vmult=0.999, amult=0.999)

            if traj2_retimed is None:
                logger.error("Transport trajectory retime fails! Try again without contact constraints.")
                traj2_retimed = toppra.retime_active_joints_kinematics(traj2_rave, self.get_robot(), additional_constraints=[])
            t4a = rospy.get_time()

            self.check_continue()
            fail = not self.execute_trajectory(traj2_retimed)
            self.get_robot().WaitForController(0)

            # release the object
            logger.info("RELEASE object")
            self._robot.Release(self.get_env().GetKinBody(obj_dict['name']))
            t4b = rospy.get_time()
            logger.info("Time report"
                        "\n - setup              :{0:f} secs"
                        "\n - APPROACH plan      :{1:f} secs"
                        "\n - APPROACH duration  :{6:f} secs"
                        "\n - REACH plan         :{2:f} secs"
                        "\n - REACH duration     :{7:f} secs"
                        "\n - MOVE plan          :{3:f} secs"
                        "\n - MOVE shortcut      :{4:f} secs"
                        "\n - MOVE retime        :{5:f} secs"
                        "\n - MOVE duration      :{8:f} secs"
                        "\n - TOTAL duration     :{9:f} secs".format(
                            t1 - t0, t1a - t1, t2a - t2, t3a - t3, t3b - t3a, t4a - t4,
                            traj0.GetDuration(), traj0b.GetDuration(),
                            traj2_retimed.GetDuration(), t4b - t0))

            # remove objects from environment
            T_cur = self.get_env().GetKinBody(obj_dict['name']).GetTransform()
            T_cur[2, 3] = 0.0
            self.get_env().GetKinBody(obj_dict['name']).SetTransform(T_cur)

        time.sleep(2)
        return not fail

    def check_continue(self):
        cmd = raw_input("[Enter] to execute, [q] to stop.")
        if cmd == "q":
            exit(42)
        else:
            return True
