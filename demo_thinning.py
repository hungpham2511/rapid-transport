import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
import toppra_app

pts = np.random.rand(10, 2)
hull = toppra_app.poly_contact.ConvexHull(pts)
pts_thin = np.array([toppra_app.poly_contact.uniform_interior_sampling(hull) for i in range(20)])
hull_thin = toppra_app.poly_contact.ConvexHull(pts_thin)

for i, hull_ in enumerate([hull, hull_thin]):
    for p1, p2 in hull_.get_faces():
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], c="C" + str(i))
plt.scatter(pts_thin[:, 0], pts_thin[:, 1], c='C1')
plt.show()

exit()
hull = ConvexHull(pts)
delaunay = Delaunay(hull.points[hull.vertices])
triangulations = delaunay.points[delaunay.simplices]
tri_areas = [toppra_app.utils.compute_volume_simplex(tri) for tri in triangulations]
tri_prob = tri_areas / np.sum(tri_areas)
N_tri = triangulations.shape[0]

pts_thin = []
for i in range(20):
    tri_sel = triangulations[np.random.choice(N_tri, p=tri_prob)]
    abg = np.random.dirichlet([1, 1, 1])
    pts_thin.append(tri_sel.T.dot(abg))
pts_thin = np.array(pts_thin)
hull_thin = ConvexHull(pts_thin)

for simplex in delaunay.simplices:
    p1, p2, p3 = delaunay.points[simplex]
    plt.plot([p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]], '--', c='gray')

for i, hull_ in enumerate([hull, hull_thin]):
    for simplex in hull_.simplices:
        p1, p2 = hull_.points[simplex]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], c="C" + str(i))
plt.scatter(pts_thin[:, 0], pts_thin[:, 1], c='C1')
plt.show()