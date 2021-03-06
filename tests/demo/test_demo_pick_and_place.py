import pytest, time
import numpy as np
from transport.console.pick_and_place import PickAndPlaceDemo
import openravepy as orpy
import transport
transport.utils.setup_logging("DEBUG")


@pytest.fixture(scope="module", params=[
    ("scenarios/test0.scenario.yaml", "denso")
])
def setup_demo(setup, request):
    scene_name, robot_name = request.param
    demo = PickAndPlaceDemo(scene_name, env=setup)
    assert demo.view()
    yield demo, robot_name


def test_init(setup_demo):
    demo, robot_name = setup_demo
    env = demo.get_env()
    assert env.GetRobot(robot_name) is not None
    assert demo.get_robot() is not None
    assert env.GetKinBody("object1") is not None
    assert env.GetKinBody("object2") is not None


def test_run(setup_demo, monkeypatch):
    monkeypatch.setattr("__builtin__.raw_input", lambda s: "")
    demo, robot_name = setup_demo
    assert demo.run()

@pytest.mark.skip(reason="skip hardware test")
def test_run_hw(setup, monkeypatch):
    monkeypatch.setattr("__builtin__.raw_input", lambda s: "")
    demo = PickAndPlaceDemo("scenarios/test0.scenario.yaml", env=setup, execute_hw=True)
    demo.view()
    assert demo.run()
