import pytest, time
import toppra_app
import openravepy as orpy
import numpy as np


@pytest.fixture(scope="module")
def object_fixture():
    env = orpy.Environment()
    env.Load('data/lab1.env.xml')
    robot = env.GetRobots()[0]
    robot.SetActiveDOFs(range(6))
    d = {"name": "test",
         "object_profile": "kindlebox_light_fortesting",
         "object_attach_to": "arm",
         "contact_profile": "suctioncup_skirt_kindlebox_467795874a",
         "contact_attach_to": "wam7",
         "T_link_object": [[1,0,0,0], [0, 1, 0, 0], [0, 0, 1, 9.08e-3], [0, 0, 0, 1]]}

    obj = toppra_app.SolidObject.init_from_dict(robot, d)
    yield obj, env, d
    env.Destroy()


def test_init_from_dict(object_fixture):
    """ An SolidObject can be initialized from a dictionary.
    """
    obj, env, d = object_fixture
    assert obj is not None
    assert obj.get_name() == "test"
    assert isinstance(obj.get_contact(), toppra_app.Contact)


def test_init_from_dict_object(object_fixture):
    "Object must be isaniated from kindlebox_light"
    obj, env, d = object_fixture
    # Check  "kindlebox_light"
    db = toppra_app.database.Database()
    obj_profile = db.retrieve_profile(d['object_profile'], "object")

    T_ee_obj = np.eye(4)
    T_ee_obj[:3, 3] = obj_profile['position']
    T_ee_obj[:3, :3] = obj_profile['orientation']

    robot = obj.get_robot()
    try:
        link = robot.GetLink(d['object_attach_to'])
        T_ee = link.GetTransform()
    except AttributeError:
        manip = robot.GetManipulator(d['object_attach_to'])
        T_ee = manip.GetTransform()

    T_obj = obj.compute_frame_transform(robot.GetActiveDOFValues())
    np.testing.assert_allclose(T_obj, T_ee.dot(T_ee_obj))


def test_init_from_dict_contact(object_fixture):
    obj, env, d = object_fixture
    assert obj.get_contact().get_profile() == "suctioncup_skirt_kindlebox_467795874a"


def test_load_to_rave(object_fixture):
    obj, env, d = object_fixture
    obj.load_to_env(np.eye(4))
    assert env.GetKinBody(obj.get_name()) is not None
    np.testing.assert_allclose(env.GetKinBody(obj.get_name()).GetTransform(), np.eye(4))
