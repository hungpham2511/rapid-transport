import pytest
import numpy as np
import toppra_app
import openravepy as orpy

@pytest.fixture()
def fixture():
    env = orpy.Environment()
    env.Load('data/lab1.env.xml')
    robot = env.GetRobots()[0]
    robot.SetActiveDOFs(range(6))
    d = {"name": "test",
         "object_profile": "kindlebox_light",
         "object_attach_to": "arm",
         "contact_profile": "suctioncup_skirt_kindlebox_467795874a",
         "contact_attach_to": "wam7"}
    yield robot, d
    env.Destroy()


def test_init_from_dict(fixture):
    robot, dictionary = fixture
    contact = toppra_app.Contact.init_from_dict(robot, dictionary)
    assert contact.get_profile() == dictionary['contact_profile']

