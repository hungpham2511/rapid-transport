import pytest
import numpy as np
import transport
import openravepy as orpy


@pytest.mark.parametrize("contact_profile_id", ["test", "test2"])
def test_init_from_profile(envlab1, contact_profile_id):
    "Contact can be insanitated from a single contact profile."
    db = transport.database.Database()
    contact_profile = db.retrieve_profile(contact_profile_id, "contact")
    robot = envlab1
    contact = transport.Contact.init_from_profile_id(robot, contact_profile_id)
    assert contact.get_profile_id() == contact_profile_id
    if 'raw_data' in contact_profile:
        assert contact.get_raw_data() == contact_profile['raw_data']
    else:
        assert contact.get_raw_data() == []

@pytest.mark.parametrize("contact_profile_id", ["test", "test2"])
def test_clone(envlab1, contact_profile_id):
    "Contact can be cloned"
    db = transport.database.Database()
    contact_profile = db.retrieve_profile(contact_profile_id, "contact")
    robot = envlab1
    contact = transport.Contact.init_from_profile_id(robot, contact_profile_id)
    contact_cloned = contact.clone()
    assert contact_cloned.get_profile_id() == contact.get_profile_id()
    assert contact_cloned.get_profile_id() == contact.get_profile_id()
