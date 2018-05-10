import logging
import numpy as np
import matplotlib.pyplot as plt
import toppra_app, hashlib
import argparse, yaml, os
import openravepy as orpy
from datetime import datetime

logger = logging.getLogger(__name__)


class ContactSimplifier(object):
    """A class implementing Guided Polyhedral Expansion algorithm for
    simplifying contacts.

    Parameters
    ----------
    robot: openravepy.Robot
    contact: Contact
    solid_object: SolidObject

    """
    def __init__(self, robot, contact, solid_object, N_samples=500, N_vertices=50, verbose=False):
        self._N_samples = N_samples
        self._N_vertices = N_vertices
        self._contact = contact
        self._robot = robot
        self._solid_object = solid_object
        self._verbose = verbose
        # Generate a the faces
        assert len(contact.get_raw_data()) > 0, "There is no raw data to work with."
        db = toppra_app.database.Database()
        ws_list = []
        logger.info("Start loading data points.")
        for file_name in self._contact.get_raw_data():
            file_dir = os.path.join(db.get_contact_data_dir(), file_name)
            ws_ = toppra_app.utils.load_data_ati_log(file_dir)
            ws_list.append(ws_)
        ws_all = np.vstack(ws_list)
        self._ws_all = ws_all
        logger.info("Finish loading data points.")
        if self._verbose:
            toppra_app.utils.preview_plot([[ws_all, 'x', 0.2]])
        hull_full = toppra_app.poly_contact.ConvexHull(ws_all)
        F, g = hull_full.get_halfspaces()
        self._contact.F_local = F
        self._contact.g_local = g

    def simplify(self, verbose=False):

        # Sample points
        logger.info("Start sampling")
        ws_thin = []
        trial = 0
        while len(ws_thin) < self._N_samples:
            trial += 1
            qdd_sam, qd_sam = toppra_app.utils.sample_uniform(2, 0.5, 6)
            q_sam = toppra_app.utils.sample_uniform(1, 3, 6)[0]
            T_world_contact = self._contact.compute_frame_transform(q_sam)
            w_sam = self._solid_object.compute_inverse_dyn(q_sam, qd_sam, qdd_sam, T_world_contact)
            if np.all(self._contact.F_local.dot(w_sam) - self._contact.g_local <= 0):
                ws_thin.append(w_sam)
        ws_thin = np.array(ws_thin)
        logger.info("Finish sampling ({:d} trials / {:d} samples)".format(trial, self._N_samples))
        if self._verbose:
            toppra_app.utils.preview_plot([[self._ws_all, 'x', 0.2], [ws_thin, 'o', 0.3]])

        # %% Polyhedral expansion
        vca_indices = toppra_app.poly_contact.vertex_component_analysis(self._ws_all)

        vertices = self._ws_all[vca_indices].tolist()
        vertices_index = list(vca_indices)
        N_vertices = -1
        while N_vertices < self._N_vertices:
            hull = toppra_app.poly_contact.ConvexHull(vertices)
            N_vertices = hull.vertices.shape[0]

            A, b = hull.get_halfspaces()
            # Select face to expand
            face_to_expand = None
            val = 1e-9
            for i in range(A.shape[0]):
                residues = ws_thin.dot(A[i]) - b[i]
                face_val = np.sum(residues[np.where(residues > 0)])
                if face_val > val:
                    face_to_expand = i
                    val = face_val

            if face_to_expand is None:
                # This mean all red vertices have been found
                print "Cover inner set!"
                break
            else:
                opt_vertex_index = np.argmax(self._ws_all.dot(A[face_to_expand]))

            vertices.append(self._ws_all[opt_vertex_index])
            vertices_index.append(opt_vertex_index)
        vertices = np.array(vertices)
        hull = toppra_app.poly_contact.ConvexHull(vertices)
        if verbose:
            fig, axs = plt.subplots(2, 2)
            to_plot = (
                (0, 1, axs[0, 0]),
                (0, 2, axs[0, 1]),
                (3, 4, axs[1, 0]),
                (4, 5, axs[1, 1]))

            for i, j, ax in to_plot:
                ax.scatter(self._ws_all[:, i], self._ws_all[:, j], c='C0', alpha=0.5, s=10)
                ax.scatter(ws_thin[:, i], ws_thin[:, j], marker='x', c='C1', zorder=10, s=50)
                ax.plot(vertices[:, i], vertices[:, j], c='C2')
            timer = fig.canvas.new_timer( interval=3000)  # creating a timer object and setting an interval of 3000 milliseconds
            timer.start()
            timer.add_callback(lambda : plt.close())
            plt.show()

        new_contact = self._contact.clone()
        new_contact.F_local = A
        new_contact.g_local = b
        return new_contact
