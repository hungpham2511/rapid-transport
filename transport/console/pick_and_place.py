import openravepy as orpy
import time, toppra
import numpy as np
import yaml
import logging

from ..utils import expand_and_join
from ..profile_loading import Database
from ..solidobject import SolidObject
from ..toppra_constraints import create_object_transporation_constraint

logger = logging.getLogger(__name__)


def main(*args, **kwargs):
    demo = PickAndPlaceDemo(*args, **kwargs)
    demo.view()
    print "Starting demo.."
    demo.run()


def plan_to_joint_configuration(
        robot, qgoal, planner='birrt', max_planner_iterations=40,max_postprocessing_iterations=60):
    env = robot.GetEnv()
    rave_planner = orpy.RaveCreatePlanner(env, planner)
    params = orpy.Planner.PlannerParameters()
    params.SetRobotActiveJoints(robot)
    params.SetGoalConfig(qgoal)
    params.SetMaxIterations(max_planner_iterations)
    params.SetPostProcessing('ParabolicSmoother', '<_nmaxiterations>{0}</_nmaxiterations>'.format(max_postprocessing_iterations))
    success = rave_planner.InitPlan(robot, params)
    if not success:
        return None
    # Plan a trajectory
    traj = orpy.RaveCreateTrajectory(env, '')
    status = rave_planner.PlanPath(traj)
    if status != orpy.PlannerStatus.HasSolution:
        return None
    return traj


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
    def __init__(self, load_path=None, env=None, verbose=False, execute_hw=False):
        assert load_path is not None
        self.verbose = verbose
        self.execute_hw = execute_hw
        if self.verbose:
            logging.basicConfig(level="DEBUG")
        else:
            logging.basicConfig(level="INFO")
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
        self._objects = []
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

    def run(self, method="ParabolicSmoother"):
        """ Run the demo.
        """
        q_current = self.get_qstart()
        self.get_robot().SetActiveDOFValues(q_current)
        fail = False
        for obj_d in self._scenario['objects']:
            manip_name = obj_d["object_attach_to"]
            manip = self.get_robot().SetActiveManipulator(manip_name)
            basemanip = orpy.interfaces.BaseManipulation(self.get_robot())
            Tstart = np.array(obj_d['T_start'])
            logger.debug(Tstart)
            Tgoal = np.array(obj_d['T_goal'])

            # Check that the starting position can be reached
            T_ee_start = np.dot(Tstart, self.get_object(obj_d['name']).get_T_object_link())
            qstart_col = manip.FindIKSolution(T_ee_start, orpy.IkFilterOptions.CheckEnvCollisions)
            qstart_nocol = manip.FindIKSolution(T_ee_start, orpy.IkFilterOptions.IgnoreEndEffectorCollisions)
            if qstart_col is None:
                fail = True
                logger.warn("Unable to find a collision free solution.")
                if qstart_nocol is None:
                    logger.warn("Reason: unable to reach this pose.")
                    cmd = raw_input("[Enter] to continue/exit. [i] to drop to Ipython.")
                    if cmd == "i":
                        import IPython
                        if IPython.get_ipython() is None:
                            IPython.embed()
                else:
                    logger.warn("Reason: collision (able to reach).")
                    self._robot.SetActiveDOFValues(qstart_nocol)
                    cmd = raw_input("[Enter] to continue/exit. [i] to drop to Ipython.")
                    if cmd == "i":
                        import IPython
                        if IPython.get_ipython() is None:
                            IPython.embed()
            if fail:
                logger.warn("Breaking from planning loop.")
                break

            # Select a goal that is closest to current postion
            qstarts = manip.FindIKSolutions(T_ee_start, orpy.IkFilterOptions.CheckEnvCollisions)
            qstart = None
            dist = 10000
            for q_ in qstarts:
                dist_ = np.linalg.norm(q_ - self.get_robot().GetActiveDOFValues())
                if dist_ < dist:
                    qstart = q_
                    dist = dist_

            traj0 = plan_to_joint_configuration(self.get_robot(), qstart)
            # traj0 = basemanip.MoveToHandPosition(matrices=[T_ee_start], outputtrajobj=True, maxtries=3)
            self.get_robot().WaitForController(0)
            # import ipdb; ipdb.set_trace()
            self._robot.Grab(self.get_env().GetKinBody(obj_d['name']))

            # constraint motion planning, {task frame}:={obj}, {obj frame} = {obj}
            T_taskframe = np.eye(4)
            T_taskframe[:3, 3] = Tstart[:3, 3]
            T_taskframe_world = np.linalg.inv(T_taskframe)
            T_ee_obj = np.linalg.inv(self.get_object(obj_d['name']).get_T_object_link())
            T_ee_goal = np.dot(Tgoal, self.get_object(obj_d['name']).get_T_object_link())
            try:
                traj1 = basemanip.MoveToHandPosition(matrices=[T_ee_goal], outputtrajobj=True,
                                                     constraintfreedoms=[1, 1, 0, 0, 0, 0],
                                                     constraintmatrix=T_taskframe_world,
                                                     constrainttaskmatrix=T_ee_obj,
                                                     constrainterrorthresh=0.8,
                                                     execute=False,
                                                     steplength=0.002)
            except orpy.planning_error, e:
                fail = True
                logger.warn("Constraint planning fails.")
                break
            self.get_robot().GetController().SetPath(traj1)
            self.get_robot().WaitForController(0)

            # Shortcutting
            with self.get_robot():
                trajnew = orpy.RaveCreateTrajectory(self.get_env(), "")
                trajnew.deserialize(traj1.serialize())
                params = orpy.Planner.PlannerParameters()
                params.SetRobotActiveJoints(self.get_robot())
                params.SetMaxIterations(100)
                params.SetPostProcessing('', '')
                planner = orpy.RaveCreatePlanner(self.get_env(), method)
                success = planner.InitPlan(self.get_robot(), params)

                if success:
                    status = planner.PlanPath(trajnew)

                if status == orpy.PlannerStatus.HasSolution:
                    logger.info("Shortcutting succeeds.")
                else:
                    logger.fatal("Shortcutting fails. Playing traj now to debug.")
                    fail = True
                    break
                    # self.get_robot().GetController().SetPath(traj1)
                    # self.get_robot().WaitForController(0)

            logger.info("Method: {:}, nb waypoints: {:d}, duration {:f}\n"
                        " Playing in 2 sec".format(method, trajnew.GetNumWaypoints(), trajnew.GetDuration()))
            logger.info("Original traj nb waypoints: {:d}".format(traj1.GetNumWaypoints()))

            # Retime now
            logger.info("Retime using toppra.")
            contact = self.get_object(obj_d['name']).get_contact()
            contact_constraint = create_object_transporation_constraint(contact, self.get_object(obj_d['name']))
            traj1new = toppra.retime_active_joints_kinematics(
                trajnew, self.get_robot(), additional_constraints=[contact_constraint])

            self.get_robot().GetController().SetPath(traj1new)
            self.get_robot().WaitForController(0)
            self._robot.Release(self.get_env().GetKinBody(obj_d['name']))

        time.sleep(2)
        return not fail

