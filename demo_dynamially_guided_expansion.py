import numpy as np
import toppra_app, os
import matplotlib.pyplot as plt
import openravepy as orpy

VERBOSE = True
if VERBOSE:
    import coloredlogs
    coloredlogs.install(level='DEBUG')
    np.set_printoptions(3)

db = toppra_app.database.Database()
contact_profile = db.retrieve_profile("nonprehensile_gripper", "contact")
object_profile = db.retrieve_profile("small_alum_block_2", "object")
robot_profile = db.retrieve_profile("gripper1", "robot")

N_kp = len(contact_profile['key_points'])
ws_all = []
ws_all.append(np.array(contact_profile['key_points']))
for file_name in contact_profile['raw_data']:
    file_dir = os.path.join(db.get_contact_data_dir(), file_name)
    ws = toppra_app.utils.load_data_ati_log(file_dir)
    ws_all.append(ws)
ws_all = np.vstack(ws_all)
hull_full = toppra_app.poly_contact.ConvexHull(ws_all)
A, b = hull_full.get_halfspaces()
print(hull_full.report())

env = orpy.Environment()
env.Load('models/' + robot_profile['robot_model'])
robot = env.GetRobots()[0]
manip = robot.GetManipulator(robot_profile['manipulator'])
arm_indices = manip.GetArmIndices()

T_object = np.eye(4)
T_object[:3, 3] = object_profile['position']
T_object[:3, :3] = object_profile['orientation']
solid_object = toppra_app.SolidObject(robot, object_profile['attached_to_manipulator'],
                                      T_object,
                                      object_profile['mass'],
                                      object_profile['local_inertia'],
                                      dofindices=arm_indices)

T_contact = np.eye(4)
T_contact[:3, 3] = contact_profile['position']
T_contact[:3, :3] = contact_profile['orientation']
contact = toppra_app.Contact(robot, contact_profile['attached_to_manipulator'],
                             T_contact, A, b, dofindices=arm_indices)

# Sample points
print ("Start sampling")
N_samples = 100
ws_thin = []
trial = 0
while len(ws_thin) < N_samples:
    trial += 1
    q_sam, qd_sam = toppra_app.utils.sample_uniform(2, 2, 6)
    qdd_sam = toppra_app.utils.sample_uniform(1, 10, 6)[0]
    T_world_contact = contact.compute_frame_transform(q_sam)
    w_sam = solid_object.compute_inverse_dyn(q_sam, qd_sam, qdd_sam, T_world_contact)

    if np.all(A.dot(w_sam) - b <= 0):
        ws_thin.append(w_sam)
ws_thin = np.array(ws_thin)
print ("Finish sampling ({:d} trials / {:d} samples)".format(trial, N_samples))

plt.scatter(ws_all[:, 0], ws_all[:, 1], c='blue', alpha=0.5, s=10)
plt.scatter(ws_thin[:, 0], ws_thin[:, 1], marker='x', c='red', zorder=5, s=50)
plt.show()

# %% Polyhedral expansion
vca_indices = toppra_app.poly_contact.vertex_component_analysis(ws_all)

vertices = ws_all[vca_indices].tolist()
vertices_index = list(vca_indices)
N_vertices = -1
while N_vertices < 100:
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
        opt_vertex_index = np.argmax(ws_all.dot(A[face_to_expand]))

    vertices.append(ws_all[opt_vertex_index])
    vertices_index.append(opt_vertex_index)

# vertices.append(np.zeros(hull_full.get_dim()))
hull = toppra_app.poly_contact.ConvexHull(vertices)

fig, axs = plt.subplots(2, 2)

to_plot = (
    (0, 1, axs[0, 0]),
    (0, 2, axs[0, 1]),
    (3, 4, axs[1, 0]),
    (4, 5, axs[1, 1]),
           )
for i, j, ax in to_plot:
    ax.scatter(ws_all[:, i], ws_all[:, j], c='blue', alpha=0.5, s=10)
    ax.scatter(ws_thin[:, i], ws_thin[:, j], marker='x', c='red', zorder=10, s=50)
    ax.plot(hull.vertices[:, i], hull.vertices[:, j])

plt.show()

print hull.report()
print (np.min(hull.get_halfspaces()[1]))
A, b = hull.get_halfspaces()
np.savez(os.path.join(db.get_contact_data_dir(), "nonprehensile_gripper_simplified_strategy9d.npz"), A=A, b=b)




