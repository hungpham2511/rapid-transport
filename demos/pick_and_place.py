import openravepy as orpy
import toppra_app, time, toppra
import numpy as np
from toppra_app.utils import expand_and_join
import yaml, logging
import os
logger = logging.getLogger(__name__)
logging.basicConfig(level='DEBUG')


class PickAndPlaceDemo(object):
    def __init__(self, load_path=None):
        assert load_path is not None
        db = toppra_app.database.Database()
        _scenario_dir = expand_and_join(db.get_data_dir(), load_path)
        with open(_scenario_dir) as f:
            self._scenario = yaml.load(f.read())
        _env_dir = expand_and_join(db.get_model_dir(), self._scenario['env'])
        self._env = orpy.Environment()
        self._env.SetDebugLevel(0)
        self._env.Load(_env_dir)
        self._robot = self._env.GetRobot(self._scenario['robot'])
        self._objects = []
        # Load all objects to openRave
        for obj_d in self._scenario['objects']:
            obj = toppra_app.SolidObject.init_from_dict(self._robot, obj_d)
            self._objects.append(obj)
            obj.load_to_env(obj_d['T_start'])
            self._robot.SetActiveManipulator(obj_d['object_attach_to'])
        # Generate IKFast if needed
        iktype = orpy.IkParameterization.Type.Transform6D
        ikmodel = orpy.databases.inversekinematics.InverseKinematicsModel(self._robot, iktype=iktype)
        if not ikmodel.load():
            print 'Generating IKFast {0}. It will take few minutes...'.format(iktype.name)
            ikmodel.autogenerate()
            print 'IKFast {0} has been successfully generated'.format(iktype.name)

    def view(self):
        res = self._env.SetViewer('qtosg')
        time.sleep(0.5)
        return res

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

    def run(self):
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
                else:
                    logger.warn("Reason: collision (able to reach).")
                    self._robot.SetActiveDOFValues(qstart_nocol)
            if fail:
                logger.warn("Breaking from planning loop.")
                break
            traj0 = basemanip.MoveToHandPosition(matrices=[T_ee_start], outputtrajobj=True)
            self.get_robot().WaitForController(0)
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

            # Shortcutting

            method = "ParabolicSmoother"

            with self.get_robot():
                trajnew = orpy.RaveCreateTrajectory(self.get_env(), "")
                trajnew.deserialize(traj1.serialize())
                # Shortcut
                params = orpy.Planner.PlannerParameters()
                params.SetRobotActiveJoints(self.get_robot())
                params.SetMaxIterations(100)
                params.SetPostProcessing('', '')
                # params.SetExtraParameters("""<_postprocessing planner="ParabolicTrajectoryRetimer">
                # <_nmaxiterations>40</_nmaxiterations>
                # </_postprocessing>""")
                # Generate the trajectory

                planner = orpy.RaveCreatePlanner(self.get_env(), method)
                success = planner.InitPlan(self.get_robot(), params)
                if success:
                    status = planner.PlanPath(trajnew)

            logger.info("Method: {:}, nb waypoints: {:d}, duration {:f}\n"
                        " Playing in 2 sec".format(method, trajnew.GetNumWaypoints(), trajnew.GetDuration()))
            logger.info("Original traj nb waypoints: {:d}".format(traj1.GetNumWaypoints()))

            # Retime now
            logger.info("Retime using toppra.")
            traj1new = toppra.retime_active_joints_kinematics(trajnew, self.get_robot())
            self.get_robot().GetController().SetPath(traj1new)
            self.get_robot().WaitForController(0)
            self._robot.Release(self.get_env().GetKinBody(obj_d['name']))

            time.sleep(2)

        return not fail


if __name__ == "__main__":
    demo = PickAndPlaceDemo("scenarios/test0.scenario.yaml")
    demo.view()
    demo.run()
    import IPython
    if IPython.get_ipython() is None:
        IPython.embed()
    
