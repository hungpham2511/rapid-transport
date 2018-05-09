import pytest
import numpy as np
import toppra_app
import openravepy as orpy


@pytest.fixture()
def fixture(setup):
    env = setup
    env.Reset()
    env.Load('data/lab1.env.xml')
    robot = env.GetRobots()[0]
    robot.SetActiveDOFs(range(6))
    yield robot


@pytest.mark.parametrize("contact_profile_id", ["test", "test2"])
def test_init_from_profile(fixture, contact_profile_id):
    db = toppra_app.database.Database()
    contact_profile = db.retrieve_profile(contact_profile_id, "contact")
    robot = fixture
    contact = toppra_app.Contact.init_from_profile_id(robot, contact_profile_id)
    assert contact.get_profile() == contact_profile_id
    if 'raw_data' in contact_profile:
        assert contact.get_raw_data() == contact_profile['raw_data']
    else:
        assert contact.get_raw_data() == []


