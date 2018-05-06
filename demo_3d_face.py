import numpy as np
import toppra_app

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
pts = np.random.rand(10, 3)
hull = toppra_app.poly_contact.ConvexHull(pts)
samples = np.array([toppra_app.poly_contact.uniform_face_sampling(hull) for i in range(200)])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(hull.vertices[:, 0], hull.vertices[:, 1], hull.vertices[:, 2], c='blue')
for face in hull.faces:
    xs = face[:, 0].tolist() + [face[0, 0]]
    ys = face[:, 1].tolist() + [face[0, 1]]
    zs = face[:, 2].tolist() + [face[0, 2]]
    ax.plot(xs, ys, zs, '--', c='blue')
ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c='red', marker='x')
plt.show()


