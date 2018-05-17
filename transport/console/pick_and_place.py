import openravepy as orpy
import time
import toppra
import numpy as np
import yaml
import logging

try:
    import raveutils
    import rospy
    from denso_control.controllers import JointPositionController
except ImportError:
    pass

from ..utils import expand_and_join, setup_logging
from ..profile_loading import Database
from ..solidobject import SolidObject
from ..toppra_constraints import create_object_transporation_constraint

logger = logging.getLogger(__name__)


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
    """ Planning function

    Parameters
    ----------
    robot: openravepy.Robot
    T_ee_start: (4,4)array
        Desired active manipulator transform.
    q_nominal: (dof,)array
        A desired/preferred configuration to achieve the goal transform.
    max_ppiters: int
        Shortcutting iterations.
    max_iters: int
        Planning iterations.

    Returns
    -------
    traj: openravepy.Trajectory

    """
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

        # Plan trajectory to that point
        traj0 = raveutils.planning.plan_to_joint_configuration(
            robot, qgoal, max_ppiters=max_ppiters, max_iters=max_iters)
    for body in all_grabbed:
        robot.Grab(body)
    return traj0


class PickAndPlaceDemo(object):
    """ A pick-and-place demo.

    Parameters
    ----------
    load_path: str
        Load path to scenario.
    env: optional
        OpenRAVE Environment.
    verbose: bool, optional
    execute_hw: bool, optional
        Send trajectory to real robot.
    """
    def __init__(self, load_path=None, env=None, verbose=False, execute_hw=False, dt=1.0 / 150, slowdown=1.0):
        assert load_path is not None
        self.verbose = verbose
        self.execute_hw = execute_hw
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
        _env_dir = expand_and_join(db.get_model_dir(), self._scenario['env'])
        if env is None:
            self._env = orpy.Environment()
        else:
            self._env = env
            self._env.Reset()
        self._env.SetDebugLevel(2)
        self._env.Load(_env_dir)
        self._robot = self._env.GetRobot(self._scenario['robot'])
        self._robot.SetDOFVelocityLimits(self.slowdown * self._robot.GetDOFVelocityLimits())
        self._robot.SetDOFAccelerationLimits(self.slowdown * self._robot.GetDOFAccelerationLimits())
        self._objects = []
        n = rospy.init_node("pick_and_place_planner")
        if self.execute_hw:
            self.joint_controller = JointPositionController('denso')
            self._robot.SetActiveDOFValues(self.joint_controller.get_joint_positions())
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
        "Check if T can be reach using manip."
        # Check that the starting position can be reached
        fail = False
        qstart_col = manip.FindIKSolution(T_ee_start, orpy.IkFilterOptions.CheckEnvCollisions)
        qstart_nocol = manip.FindIKSolution(T_ee_start, orpy.IkFilterOptions.IgnoreEndEffectorCollisions)
        if qstart_col is None:
            fail = True
            logger.fatal("Unable to find a collision free solution.")
            if qstart_nocol is None:
                logger.fatal("Reason: unable to reach this pose.")
                cmd = raw_input("[Enter] to continue/exit. [i] to drop to Ipython.")
                if cmd == "i":
                    import ipdb; ipdb.set_trace()
            else:
                logger.fatal("Reason: collision (robot is able to reach this transform).")
                logger.fatal("Collision might be due to grabbed object only. Check viewer.")
                with self._robot:
                    self._robot.SetActiveDOFValues(qstart_nocol)
                    self._env.UpdatePublishedBodies()
                    cmd = raw_input("[Enter] to continue/exit. [i] to drop to Ipython.")
                    if cmd == "i":
                        import ipdb; ipdb.set_trace()
        return fail

    def check_trajectory_collision(self, traj):
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

    def execute_trajectory(self, trajectory):
        """ Execute the trajectory. 
        
        If execute_hw is true, also publish command message to ROS.
        """
        if trajectory is None:
            return False
        spec = trajectory.GetConfigurationSpecification()
        duration = trajectory.GetDuration()
        for t in np.arange(0, duration, self._dt):
            t_start = rospy.get_time()
            data = trajectory.Sample(t)
            q = spec.ExtractJointValues(data, self._robot, range(6), 0)
            if self.execute_hw:
                self.joint_controller.set_joint_positions(q)
            self.get_robot().SetDOFValues(q)
            self.get_env().UpdatePublishedBodies()
            t_elasped = rospy.get_time() - t_start
            logger.debug("Extraction cost per loop {:f} / {:f}".format(t_elasped, self._dt))
            rospy.sleep(self._dt - t_elasped)
        return True

    def run(self, offset=0.01):
        """ Run the demo.

        For each object, the robot first move to a pose that is directly on top of it. Afterward, it moves down.
        Next, it grabs the object to be transported. Finally, it moves to another configuration.

        """
        q_current = self.get_qstart()
        self.get_robot().SetActiveDOFValues(q_current)
        fail = False
        for obj_d in self._scenario['objects']:
            manip_name = obj_d["object_attach_to"]
            manip = self.get_robot().SetActiveManipulator(manip_name)
            # basemanip = orpy.interfaces.BaseManipulation(self.get_robot())
            Tstart = np.array(obj_d['T_start'])
            Tgoal = np.array(obj_d['T_goal'])
            T_ee_start = np.dot(Tstart, self.get_object(obj_d['name']).get_T_object_link())
            self.verify_transform(manip, T_ee_start)
            T_ee_top = np.array(T_ee_start)
            T_ee_top[:3, 3] -= offset * T_ee_top[:3, 2]

            q_nominal = np.r_[-0.3, 0.9, 0.9, 0, 0, 0]

            # Trajectory which first visit a pose that is on top of the given transform, then go down.
            traj0 = plan_to_manip_transform(self._robot, T_ee_top, q_nominal, max_ppiters=200, max_iters=100)
            self.check_continue()
            fail = not self.execute_trajectory(traj0)
            self.get_robot().WaitForController(0)

            traj0b = plan_to_manip_transform(self._robot, T_ee_start, q_nominal, max_ppiters=200, max_iters=100)
            self.check_continue()
            fail = not self.execute_trajectory(traj0b)
            self.get_robot().WaitForController(0)
            self._robot.Grab(self.get_env().GetKinBody(obj_d['name']))
            logger.info("Grabbing the object. Continue moving in 1 sec.")
            time.sleep(1.0)

            # Trajectory which first visit a pose that is on top of the given transform, then go down.
            traj0c = plan_to_manip_transform(self._robot, T_ee_top, q_nominal, max_ppiters=200, max_iters=100)
            self.check_continue()
            fail = not self.execute_trajectory(traj0c)
            self.get_robot().WaitForController(0)


            # Compute goal transform
            T_ee_goal = np.dot(Tgoal, self.get_object(obj_d['name']).get_T_object_link())
            self.verify_transform(manip, T_ee_goal)
            q_nominal = np.r_[0.3, 0.9, 0.9, 0, 0, 0]
            traj1 = plan_to_manip_transform(self._robot, T_ee_goal, q_nominal, max_ppiters=200, max_iters=100)

            # # Select a goal that is closest to current postion
            # qgoals = manip.FindIKSolutions(T_ee_goal, orpy.IkFilterOptions.CheckEnvCollisions)
            # qgoal = None
            # dist = 10000
            # for q_ in qgoals:
            #     dist_ = np.linalg.norm(q_ - q_nominal)
            #     if dist_ < dist:
            #         qgoal = q_
            #         dist = dist_

            # # traj1 = basemanip.MoveActiveJoints(goal=qgoal, outputtrajobj=True, execute=False)
            # traj1 = raveutils.planning.plan_to_joint_configuration(self._robot, qgoal,
            #                                                        max_ppiters=60, max_iters=100)

            if self.check_trajectory_collision(traj1):
                logger.fatal("There are collisions.")
            trajnew = traj1

            ###  # constraint motion planning, {task frame}:={obj}, {obj frame} = {obj}
            # T_taskframe = np.eye(4)
            # T_taskframe[:3, 3] = manip.GetTransform()[:3, 3] + np.r_[0, 0, 5e-2]
            # T_taskframe_world = np.linalg.inv(T_taskframe)

            # try:
            #     # Constraint planning currently does not work.

            #     traj1 = basemanip.MoveToHandPosition(matrices=[T_ee_goal], outputtrajobj=True,
            #                                          constraintfreedoms=[1, 1, 0, 0, 0, 0],
            #                                          constraintmatrix=T_taskframe_world,
            #                                          constrainttaskmatrix=T_ee_obj,
            #                                          constrainterrorthresh=0.1,
            #                                          execute=False,
            #                                          steplength=0.001)

            #     # traj1 = basemanip.MoveToHandPosition(matrices=[T_ee_goal], outputtrajobj=True,
            #     #                                      execute=False)

            # except orpy.planning_error, e:
            #     fail = True
            #     logger.warn("Constraint planning fails.")
            #     break

            # self.get_robot().GetController().SetPath(traj1)
            # self.get_robot().WaitForController(0)

            # # Shortcutting
            # with self.get_robot():
            #     trajnew = orpy.RaveCreateTrajectory(self.get_env(), "")
            #     trajnew.deserialize(traj1.serialize())
            #     params = orpy.Planner.PlannerParameters()
            #     params.SetRobotActiveJoints(self.get_robot())
            #     params.SetMaxIterations(500)
            #     params.SetPostProcessing('', '')
            #     planner = orpy.RaveCreatePlanner(self.get_env(), method)
            #     success = planner.InitPlan(self.get_robot(), params)

            #     if success:
            #         status = planner.PlanPath(trajnew)

            #     if status == orpy.PlannerStatus.HasSolution:
            #         logger.info("Shortcutting succeeds.")
            #     else:
            #         # fail = True
            #         logger.fatal("Shortcutting fails. Keep running")

            logger.info("Original traj nb waypoints: {:d}".format(traj1.GetNumWaypoints()))

            # Retime now
            logger.info("Retime using toppra.")
            contact = self.get_object(obj_d['name']).get_contact()
            contact_constraint = create_object_transporation_constraint(contact, self.get_object(obj_d['name']))
            traj1new = toppra.retime_active_joints_kinematics(
                trajnew, self.get_robot(), additional_constraints=[contact_constraint])

            self.check_continue()
            fail = not self.execute_trajectory(traj1new)
            self.get_robot().WaitForController(0)
            self._robot.Release(self.get_env().GetKinBody(obj_d['name']))
            logger.info("Releasing the object. Robot stops for 1 secs")
            time.sleep(1)
        time.sleep(2)
        return not fail

    def check_continue(self):
        cmd = raw_input("[Enter] to execute, [q] to stop.")
        if cmd == "q":
            exit(42)
        else:
            return True
