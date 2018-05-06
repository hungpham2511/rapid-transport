import numpy as np
import openravepy as orpy
import numpy, time

# Adjust the building model directory here
model_dir = "denso_ft_gripper_with_base.robot.xml"
env = orpy.Environment()
env.SetViewer('qtosg')


def reset_and_reload(model_dir=model_dir, load_robot=False):
    """This function reset the environment and reload a new robot at `model_dir`

    """
    env.Reset()
    env.Load(model_dir)
    if load_robot:
        robot = env.GetRobots()[0]
        return robot


reset_and_reload()


print """
Let `xml_dir` be the directory of the xml model to be built.

1) edit the xml file
2) run reset_and_reload(xml_dir) to reload and view the model
"""
import IPython; IPython.embed();
