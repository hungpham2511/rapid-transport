import transport
import pytest
import logging
import transport.console.simplify_contact as simplify_contact

logging.basicConfig(level="DEBUG")


@pytest.fixture(params=["suctioncup_kindlebox2_fortesting"])
def contact_fixture(envcage, request):
    robot = envcage.GetRobots()[0]
    contact = transport.Contact.init_from_profile_id(robot, request.param)
    input_dict = {
        "name": "object1",
        "object_profile": "kindlebox_light",
        "object_attach_to": "denso_suction_cup",
        "T_link_object": [[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 9.08e-3],
                          [0, 0, 0, 1]]
    }
    solid_object = transport.SolidObject.init_from_dict(robot, input_dict)

    # Remove pre-gen data points
    contact.F_local = None
    contact.g_local = None
    yield robot, contact, solid_object


def test_basic(contact_fixture):
    robot, contact, solid_object = contact_fixture
    cs = transport.ContactSimplifier(robot, contact, solid_object,
                                     N_samples=50, N_vertices=10, verbose=True)
    cs_new, _ = cs.simplify()
    assert isinstance(cs_new, transport.Contact)


def test_simplify_contact_main(envcage, monkeypatch):
    "Test the simplify wrench script."
    monkeypatch.setattr("__builtin__.raw_input", lambda s: "")
    res = simplify_contact.main(env=envcage, contact_id="suctioncup_kindlebox2_fortesting",
                                object_id="kindlebox_light", attach="denso_suction_cup",
                                T_link_object="[[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 3.5e-3], [0, 0, 0, 1]]",
                                robot_id="suctioncup1",
                                verbose=True,
                                N_samples=50,
                                N_vertices=10)
    assert res is True
    
