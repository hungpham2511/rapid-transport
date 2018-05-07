import pytest, os
import toppra_app
import numpy as np


def test_load_profile():
    database = toppra_app.database.Database()
    assert database.retrieve_profile("contact1", "contact") is not None
    database.retrieve_profile("kindlebox_light", "object")
    database.retrieve_profile("topp_fast", "algorithm")
    database.retrieve_profile("suctioncup1_slow", "robot")


def test_load_contacts():
    database = toppra_app.database.Database()
    d = database.get_contact_data_dir()
    f = np.load(os.path.join(d, "contact1_simplified.npz"))
    assert f is not None

    
