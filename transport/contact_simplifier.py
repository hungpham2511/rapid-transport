import logging
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import argparse, yaml, os, random
import openravepy as orpy
from datetime import datetime

from .profile_loading import Database
from .poly_contact import ConvexHull

from . import utils
from . import poly_contact

logger = logging.getLogger(__name__)


class ContactSimplifier(object):
    """An implementation of Guided Polyhedron Expansion algorithm for
    simplifying contact stabilitty constraint.

    Parameters
    ----------
    robot: openravepy Robot
    contact: Contact
        An input contact object that has a non-empty `raw_data` field or a `extreme` fields.
    solid_object: SolidObject
    N_samples: int, optional
        Number of samples to generate from the object dynamics.
    N_vertices: int, optional
        Number of maximum vertices.
    verbose: bool, optional

    """
    def __init__(self, robot, contact, solid_object, N_samples=500, N_vertices=50, verbose=False, scale=1.0, cover_vertices=False):
        self._N_samples = N_samples
        self._N_vertices = N_vertices
        self._contact = contact
        self._robot = robot
        self._solid_object = solid_object
        self._verbose = verbose
        self._cover_vertices = cover_vertices
        # Generate or load the vertices `self._ws_all` and
        # the faces `self.F_local, self.g_local`.
        assert len(contact.get_raw_data()) > 0, "There is no raw data to work with."
        db = Database()
        ws_list = []
        logger.info("Loading raw data points.")
        for file_name in self._contact.get_raw_data():
            file_dir = os.path.join(db.get_contact_data_dir(), file_name)
            ws_ = utils.load_data_ati_log(file_dir)
            ws_list.append(ws_)
            logger.info("file: {:} [Done]".format(file_name))
        ws_all = np.vstack(ws_list)
        logger.info("Scaling raw wrench data with factor = {:f}".format(scale))
        w_mean = np.mean(ws_all, axis=0)
        ws_all = w_mean + (ws_all - w_mean) * scale
        self._ws_all = ws_all
        if self._verbose:
            utils.preview_plot([[ws_all, 'x', {}]], dur=20)
        hull_full = ConvexHull(ws_all)
        F, g = hull_full.get_halfspaces()
        self._contact.F_local = F
        self._contact.g_local = g
        logger.info("Loading finishes.")

    def simplify(self):
        """ Simplify contact.

        Returns
        -------
        new_contact: Contact
        """
        logger.info("Start sampling {:d} wrenches for guidance.".format(self._N_samples))
        ws_sample = []  # guided samples
        trial = 0  # nb of random trial 
        perc = -1e-9  # current generation progress
        while len(ws_sample) < self._N_samples:
            trial += 1
            qdd_sam, qd_sam = utils.sample_uniform(2, 0.5, 6)
            q_sam = utils.sample_uniform(1, 3, 6)[0]
            T_world_contact = self._contact.compute_frame_transform(q_sam)
            w_sam = self._solid_object.compute_inverse_dyn(q_sam, qd_sam, qdd_sam, T_world_contact)
            if np.all(self._contact.F_local.dot(w_sam) - self._contact.g_local <= 0):
                ws_sample.append(w_sam)
            if float(len(ws_sample)) / self._N_samples >= perc:
                logger.info("Generated {:d} / {:d} samples in {:d} trials".format(len(ws_sample), self._N_samples, trial))
                perc += 0.1
        ws_sample = np.array(ws_sample)
        logger.info("Finish sampling ({:d} trials / {:d} samples)".format(trial, self._N_samples))
        if self._verbose:
            utils.preview_plot([[self._ws_all, 'x', {}], [ws_sample, 'o', {}]])

        # %% Polyhedral expansion
        # vca_indices = poly_contact.vertex_component_analysis(self._ws_all)
        # logger.debug("Generate initial vertices with vca!")
        # simp_vertices = self._ws_all[vca_indices].tolist()

        logger.info("Choose 8 points in the guided wrenches as the initial guess.")
        simp_vertices = random.sample(ws_sample, 8)
        hull = poly_contact.ConvexHull(simp_vertices)
        N_vertices = hull.vertices.shape[0]
        A_cover = []
        b_cover = []
        while N_vertices < self._N_vertices:
            logger.debug("[Projection] N_vertices={:d}".format(N_vertices))
            logger.debug("[Projection] total number of point {:d}".format(len(simp_vertices)))
            logger.debug("[Projection] Generate new convex hull")
            logger.debug("[Projection] Select a face to expand")
            A, b = hull.get_halfspaces()
            N_faces = len(b)
            # Select face to expand
            face_to_expand = None
            max_residue = 1e-9
            for i in range(N_faces):
                residues = ws_sample.dot(A[i]) - b[i]
                # An infinitesimal, 1e-8, ensures that the same
                # samples do not get selected twice.
                face_residue = np.sum(residues[np.where(residues > 1e-8)])
                if face_residue > max_residue:
                    face_to_expand = int(i)
                    max_residue = face_residue

            if face_to_expand is None:
                # This mean all red vertices have been found
                logger.info("Cover inner set!")
                break

            opt_vertex_index = np.argmax(self._ws_all.dot(A[face_to_expand]))
            logger.debug("[Projection] Max residue {:f}".format(max_residue))
            # Add constraints to exclude the furthermost edge
            if self._cover_vertices:
                A_cover.append(np.array(A[face_to_expand]))
                b_cover.append(b[face_to_expand] + max_residue * 0.9)

            simp_vertices.append(self._ws_all[opt_vertex_index])
            hull = poly_contact.ConvexHull(simp_vertices)
            N_vertices = hull.vertices.shape[0]
        simp_vertices = np.array(simp_vertices)
        if len(A_cover) > 0:
            logger.info("COVER_VERTICES is true, stacking {:d} more constraints".format(len(b_cover)))
            A = np.vstack((A, A_cover))
            b = np.hstack((b, b_cover))

        if self._verbose:
            utils.preview_plot([[self._ws_all, "x", {"markersize": 2}],
                                [simp_vertices, "--o", {"linewidth": 0.5}]])

        new_contact = self._contact.clone()
        new_contact.F_local = A
        new_contact.g_local = b
        return new_contact, hull
