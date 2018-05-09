import pytest, time
import numpy as np
from demos.pick_and_place import PickAndPlaceDemo
import openravepy as orpy

import logging
logging.basicConfig(level="DEBUG")


@pytest.fixture(scope="module", params=[
    ("scenarios/test0.scenario.yaml", "denso")
])
def setup_demo(request):
    scene_name, robot_name = request.param
    demo = PickAndPlaceDemo(scene_name)
    assert demo.view()
    yield demo, robot_name
    demo.get_env().Destroy()


def test_init(setup_demo):
    demo, robot_name = setup_demo
    env = demo.get_env()
    assert env.GetRobot(robot_name) is not None
    assert demo.get_robot() is not None
    assert env.GetKinBody("object1") is not None
    assert env.GetKinBody("object2") is not None


def test_run(setup_demo):
    demo, robot_name = setup_demo
    assert demo.run()
