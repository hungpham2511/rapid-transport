import numpy as np
import toppra_app, os
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull as sciConvexHull

VERBOSE = True
if VERBOSE:
    import coloredlogs
    coloredlogs.install(level='DEBUG')
    np.set_printoptions(3)

db = toppra_app.database.Database()
contact_profile = db.retrieve_profile("nonprehensile_gripper", "contact")
N_kp = len(contact_profile['key_points'])
ws_all = []
ws_all.append(np.array(contact_profile['key_points']))
for file_name in contact_profile['raw_data']:
    file_dir = os.path.join(db.get_contact_data_dir(), file_name)
    ws = toppra_app.utils.load_data_ati_log(file_dir)
    ws_all.append(ws)
ws_all = np.vstack(ws_all)

c_vec = np.fromstring("""
-1.495e-01  -3.025e-02  -5.841e-03  -1.947e-01   9.604e-01   1.058e-02
-1.470e-01  -3.716e-02  -7.176e-03  -2.392e-01   9.501e-01  -2.133e-02
-1.443e-01  -4.300e-02  -8.303e-03  -2.768e-01   9.388e-01  -5.009e-02
-1.418e-01  -4.782e-02  -9.234e-03  -3.078e-01   9.273e-01  -7.580e-02
-1.394e-01  -5.169e-02  -9.982e-03  -3.327e-01   9.165e-01  -9.857e-02
-1.373e-01  -5.470e-02  -1.056e-02  -3.521e-01   9.069e-01  -1.185e-01
-1.356e-01  -5.690e-02  -1.099e-02  -3.663e-01   8.988e-01  -1.358e-01
-1.341e-01  -5.837e-02  -1.127e-02  -3.757e-01   8.925e-01  -1.504e-01
-1.331e-01  -5.916e-02  -1.142e-02  -3.808e-01   8.882e-01  -1.627e-01
-1.324e-01  -5.934e-02  -1.146e-02  -3.819e-01   8.859e-01  -1.726e-01
-1.322e-01  -5.893e-02  -1.138e-02  -3.793e-01   8.854e-01  -1.803e-01
-1.322e-01  -5.800e-02  -1.120e-02  -3.733e-01   8.868e-01  -1.859e-01
-1.326e-01  -5.656e-02  -1.092e-02  -3.641e-01   8.899e-01  -1.896e-01""",
                      dtype=float, sep=" ").reshape((-1, 6))

# ws_all = np.random.rand(100, 2)
# ws_all = np.array([[1, 1], [3, -40]]).dot(ws_all.T).T

hull_full = toppra_app.poly_contact.ConvexHull(ws_all)
print(hull_full.report())

# %% thinning, choose 1000 samples
print ("Start sampling")
N_samples = 100
ws_thin = []
for i in range(N_samples):
    ws_thin.append(toppra_app.poly_contact.uniform_interior_sampling(hull_full))
ws_thin = np.array(ws_thin)
print ("Finish sampling")

plt.scatter(ws_all[:, 0], ws_all[:, 1], c='blue', alpha=0.5, s=10)
plt.scatter(ws_thin[:, 0], ws_thin[:, 1], marker='x', c='red', zorder=5, s=50)
plt.scatter(c_vec[:, 0],c_vec[:, 1], marker='^', c='green', zorder=10, s=50)
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
    ax.scatter(c_vec[:, i],c_vec[:, j], marker='^', c='green', zorder=10, s=50)
    ax.plot(hull.vertices[:, i], hull.vertices[:, j])

plt.show()

print hull.report()
print (np.min(hull.get_halfspaces()[1]))




