from ..contact_simplifier import ContactSimplifier
from ..profile_loading import Database
from .. import utils
from ..contact import Contact
from ..solidobject import SolidObject
import numpy as np
import openravepy as orpy
import yaml
import hashlib
import logging
import os
from datetime import datetime
logger = logging.getLogger(__name__)


def main(env=None, contact_id=None, object_id=None, attach_name=None, T_link_object=None, robot_id=None, verbose=False, N_samples=200, N_vertices=50):
    "Program to simplify contact profile."
    if verbose:
        utils.setup_logging("DEBUG")
    else:
        utils.setup_logging("INFO")
    logger.info("Setup new simplification problem with the followng parameters: N_samples={:d}, N_vertices={:}".format(N_samples, N_vertices))
    if env is None:
        logger.debug("Rave environemnt not given, creating new one.")
        env = orpy.Environment()
    else:
        env.Reset()
    db = Database()
    robot_profile = db.retrieve_profile(robot_id, "robot")
    contact_profile = db.retrieve_profile(contact_id, "contact")
    env.Load(utils.expand_and_join(
        db.get_model_dir(), robot_profile['robot_model']))
    logger.info("Finish initializing Rave environemnt.")
    robot = env.GetRobots()[0]
    contact = Contact.init_from_profile_id(robot, contact_id)
    logger.info("Finish loading contact")
    solid_object = SolidObject.init_from_dict(robot, {
        "name": "object1",
        "object_profile": object_id,
        "object_attach_to": attach_name,
        "T_link_object": np.array(yaml.load(T_link_object))
    })
    simp = ContactSimplifier(robot, contact, solid_object, N_samples=N_samples, N_vertices=N_vertices, verbose=verbose)
    cs_new, hull_simplified = simp.simplify()
    ###########################################################################
    #                         Save new contact to database                    #
    ###########################################################################
    new_contact_id = contact_id + "_" + hashlib.md5(
        "strategy10" + object_id + str(N_samples)
        + str(N_vertices) + T_link_object).hexdigest()[:10]
    cmd = raw_input("Save the simplified contact as [{:}] y/[N]?".format(new_contact_id))
    if cmd != 'y':
        print("Do not save. Exit!")
    else:
        A, b = cs_new.get_constraint_coeffs_local()
        np.savez(os.path.join(db.get_contact_data_dir(), new_contact_id + ".npz"), A=A, b=b)
        new_contact_profile = {
            'id': new_contact_id,
            'N_vertices': hull_simplified.get_vertices().shape[0],
            "N_faces": hull_simplified.get_halfspaces()[0].shape[0],
            "volume": hull_simplified.compute_volume(),
            'description': contact_profile['description'],
            'attached_to_manipulator': contact_profile["attached_to_manipulator"],
            'strategy': "strategy10",
            "position": list(contact_profile['position']),
            "orientation": list(contact_profile['orientation']),
            'constraint_coeffs_file': new_contact_id + ".npz",
            'generated_from': contact_id,
            'generated_on': str(datetime.now())
        }
        db.insert_profile(new_contact_profile, "contact")
        logger.info("Insert new profile as {:}".format(new_contact_id))
    return True
    
