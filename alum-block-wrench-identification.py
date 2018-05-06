"""

The main goal of this script is to generate a set of inequalities
Ax <= b
where x is the set of feasible wrenches transformed to the Alum block's
body-fix frame.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import toppra_app, os
import cvxpy as cvx

import coloredlogs
coloredlogs.install(level='INFO')

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


# %% Face Random sampling
hull = toppra_app.poly_contact.ConvexHull(ws_all)
print(hull.report())

N_sample = 1000
samples = []
for i in range(N_sample):
    samples.append(toppra_app.poly_contact.uniform_face_sampling(hull))

hulls = []
for i in range(8, N_sample):
    hull_ = toppra_app.poly_contact.ConvexHull(samples[:i])
    hulls.append(hull_)

vols = [hull_.compute_volume() for hull_ in hulls]
N_faces = [len(hull_.get_faces()) for hull_ in hulls]
N_vertices = [len(hull_.get_vertices()) for hull_ in hulls]

plt.plot(range(8, N_sample), vols)
plt.show()

exit()


# %%
vca_indices = toppra_app.poly_contact.vertex_component_analysis(ws_all)
hull_vca = toppra_app.poly_contact.ConvexHull(ws_all[vca_indices])
print(hull_vca.report())

poly_vca_indices = toppra_app.poly_contact.poly_expansion(ws_all, vca_indices, 50, max_area=False)
hull_poly_vca = toppra_app.poly_contact.ConvexHull(ws_all[poly_vca_indices])
print(hull_poly_vca.report())

hull_kp = toppra_app.poly_contact.ConvexHull(contact_profile['key_points'])
print(hull_kp.report())

poly_kp_indices = toppra_app.poly_contact.poly_expansion(ws_all, range(N_kp), 50, max_area=False)
hull_poly_kp = toppra_app.poly_contact.ConvexHull(ws_all[poly_kp_indices])
print(hull_poly_kp.report())

exit()

# %% Idea number 1: binning
ws_e_ = ws_e[10000:]
centroid = np.zeros(6)
N_bin = ws_e_.shape[0] / 50
ws_inbin = [centroid]
for i in range(N_bin - 1):
    i_max = np.argmax(np.linalg.norm(ws_e_[i * 50: (i+1) * 50], axis=1))
    ws_inbin.append(ws_e_[i_max + i * 50])
ws_inbin = np.array(ws_inbin)
hull2 = toppra_app.poly_contact.ConvexHull(ws_inbin)
print(hull2.report())
ws_inbin_ = ws_inbin[np.where(np.linalg.norm(ws_inbin, axis=1) > 20)]
hull3 = toppra_app.poly_contact.ConvexHull(ws_inbin_.tolist() + [centroid])
print(hull3.report())

plt.plot(hull2.vertices)
plt.show()
# %%
# Preview
index_to_view = range(0, 18200)
plt.plot(index_to_view, ws_e[index_to_view])
plt.show()


# %%
plt.plot(range(10000, 50000), np.linalg.norm(ws_e, axis=1)[10000:50000])
plt.show()


# %%
v_indices = hull.hull.vertices
v = hull.hull.points[v_indices]

plt.scatter(v_indices, np.linalg.norm(v, axis=1))
plt.show()
# %%


plt.plot(ws_e, c='blue')
plt.plot(hull.hull.vertices, hull.hull.points[hull.hull.vertices], c='red')
plt.show()

# %%
hull2 = toppra_app.poly_contact.simplify_vertex_analysis(
    hull, {'max_iter': 50, 'max_area': True})
print(hull2.report())

# %% Redudant constraints identification: no redundancy found
A, b = hull2.get_halfspaces()
N_constraint = A.shape[0]

for i in range(N_constraint):
    b_aug = np.array(b)
    b_aug[i] += 1
    x = cvx.Variable(6)
    obj = cvx.Maximize(A[i] * x)
    prob = cvx.Problem(obj, [A * x <= b_aug])
    prob.solve()
    if prob.value <= b[i]:
        print "Constraint {:d} is redudant!".format(i)
        print(prob.value, b[i])


# %%

# Projection
dof1, dof2 = 3, 5
plt.scatter(hull.vertices[:, dof1], hull.vertices[:, dof2], marker='x', c='blue', alpha=0.9)
plt.scatter(hull2.vertices[:, dof1], hull2.vertices[:, dof2], s=60, marker='o', c='red')
plt.show()

# %%

hull3 = toppra_app.poly_contact.simplify_curvature(
    hull, {'n_unit_vectors': 100000, 'n_extremes': 100})
print(hull3.report())

# %%
n_clusters_kmeans = [100, 200, 500, 1000, 2000]
hull_kmeans = map(
    lambda  n: toppra_app.poly_contact.simplify_kmean(hull, {'n_clusters': n}),
    n_clusters_kmeans)

reports = [h.report() for h in [hull] + hull_kmeans]
for r in reports:
    print r

# %%

# Frame transform
# Here I use the Ad operator to perform the transformation
# [Ad_T_ab]^T w_a = w_b  (Equation 3.101 in Park and Lynch, 2017)
# Equation for the Ad operator is
# [Ad_T] = [R    0 ]
#          [[p]R R ]

# Position of the two centers of the tip inner edges of the open gripper
# are: [0, -0.0425, 0.16796] and [0, 0.0425, 0.16796] respectively.

print "Transform wrenches to the object frame"
T_eo = np.array([[0, 1, 0, 0.0e-3],
                 [0, 0, 1, -0.0425 + 25.2e-3 / 2],
                 [1, 0, 0, 0.16796 - 25.2e-3 / 2],
                 [0, 0, 0, 1]])

p = T_eo[:3, 3]
p_skew = np.array([[0, -p[2], p[1]],
                   [p[2], 0, -p[0]],
                   [-p[1], p[0], 0]])

Ad_T_eo = np.block([[T_eo[:3, :3], np.zeros((3, 3))],
                    [p_skew.dot(T_eo[:3, :3]), T_eo[:3, :3]]])

ws_o = np.array([Ad_T_eo.T.dot(w) for w in ws_e])

# Convex hull
print "Generating convex hull..."
hull = ConvexHull(ws_o)  # The convex hull
ws_hull = hull.points[hull.vertices]  # Coordinates of the vertices of the hull
print "Convex hull has: {:d} points and {:d} equation".format(hull.vertices.shape[0], hull.equations.shape[0])

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(211, projection='3d')
ax.scatter(ws_hull[:, 0], ws_hull[:, 1], ws_hull[:, 2])
plt.title("Scatter plot of forces")
ax = fig.add_subplot(212, projection='3d')
ax.scatter(ws_hull[:, 3], ws_hull[:, 4], ws_hull[:, 5])
plt.title("Scatter plot of forces")
plt.show()


# H-rep from ConvexHull A_hull p <= b_hull
A_hull = hull.equations[:, :-1]
b_hull = - hull.equations[:, -1]

centroid = np.mean(ws_hull, axis=0)
print "Centroid: {:}".format(centroid)
assert np.all(A_hull.dot(centroid) <= b_hull)
print "Check that the centroid satisfies all plane equation successful!"
print "Saving (A, b) to {:}...".format("data/alum_block_rubber_contact_contact")
np.savez("data/alum_block_rubber_contact_contact", A=A_hull, b=b_hull)

