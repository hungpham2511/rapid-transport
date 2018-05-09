import toppra_app
import pytest


@pytest.fixture
def contact_fixture(envcage):
    robot = envcage.GetRobots()[0]
    contact = toppra_app.Contact.init_from_profile_id(robot, "suctioncup_skirt_kindlebox")
    input_dict = {
        "name": "object1",
        "object_profile": "kindlebox_light",
        "object_attach_to": "denso_suction_cup",
        "T_link_object": [[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 9.08e-3],
                          [0, 0, 0, 1]]
    }
    solid_object = toppra_app.SolidObject.init_from_dict(robot, input_dict)
    yield robot, contact, solid_object


def test_basic(contact_fixture):
    robot, contact, solid_object = contact_fixture
    cs = toppra_app.ContactSimplifier(robot, contact, solid_object)
    cs_new = cs.simplify()
    assert isinstance(cs_new, toppra_app.Contact)


