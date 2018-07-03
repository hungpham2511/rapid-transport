import argparse, yaml
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt
import openravepy as orpy

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def show3d(input_dict):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    poly3d_collection_vertices = []
    for cube_ in input_dict:
        if cube_ == "unit":
            continue

        p_com = np.array(input_dict[cube_]['com_position'], dtype=float).reshape(3,)
        dx, dy, dz = np.array(input_dict[cube_]['size'], dtype=float)

        vertices = np.array([
            [dx / 2, -dy/2, -dz/2],
            [-dx / 2, -dy/2, -dz/2],
            [-dx / 2, -dy/2, dz/2],
            [dx / 2, -dy/2, dz/2],
            [dx / 2, dy/2, -dz/2],
            [-dx / 2, dy/2, -dz/2],
            [-dx / 2, dy/2, dz/2],
            [dx / 2, dy/2, dz/2],
        ]) + p_com

        poly3d_collection_vertices.extend([
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[3], vertices[0], vertices[4], vertices[7]],
        ])

        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2])
        ax.text(p_com[0], p_com[1], p_com[2], cube_, color="black")
    poly = Poly3DCollection(poly3d_collection_vertices, alpha=0.1, facecolors='cyan', linewidths=1, edgecolors='black')
    # poly.set_alpha(0.25)
    ax.add_collection(poly)
    set_axes_equal(ax)

    plt.show()


class Geometry(object):
    def compute_inertia_at_point(self, p):
        """ Compute the moment of inertia at point `p`.

        Parameters
        ----------
        p: (3,)array
            The point of interest.

        Returns
        -------
        (3,3)array
            Moment of inertia matrix.
        """
        r = p - self.com_position  # Vector from the com to point p
        I_com = self.I_local + self.mass * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
        return I_com

    def preview_in_rave(self, env):
        pass

    def get_mass(self):
        return self.mass

    def get_com_position(self):
        return self.com_position

    def get_volume(self):
        return self.volume

    def get_density(self):
        return self.density


class Cube(Geometry):
    """ A Cube is a simple 3D rectangle.

    """
    def __init__(self, mass, size, com_position, name=""):
        self.name = name
        self.mass = mass
        self.size = np.array(size).reshape(3,)
        self.com_position = np.array(com_position).reshape(3,)
        self.volume = self.size[0] * self.size[1] * self.size[2]
        self.density = self.mass / self.volume

        dx, dy, dz = self.size
        self.I_local = self.mass / 12 * np.array([
            [dy ** 2 + dz ** 2, 0, 0],
            [0, dx ** 2 + dz ** 2, 0],
            [0, 0, dx ** 2 + dy ** 2]
        ])

    def preview_in_rave(self, env):
        body = orpy.RaveCreateKinBody(env, "")
        body.SetName(self.name)
        body.InitFromBoxes(np.array([[self.com_position[0], self.com_position[1], self.com_position[2],
                                      self.size[0] / 2, self.size[1] / 2, self.size[2] / 2]]), True)
        env.Add(body, True)

class HollowCube(Geometry):
    def __init__(self, mass, outer_size, thickness, com_position, name=""):
        self.name = name
        self.mass = mass
        self.outer_size = np.array(outer_size).reshape(3,)
        self.thickness = thickness
        self.com_position = np.array(com_position).reshape(3,)

        dx, dy, dz = self.outer_size
        dx_inner, dy_inner, dz_inner = self.outer_size - 2 * self.thickness
        assert (self.outer_size > 2 * self.thickness).all(), "Thickness too high"
        self.outer_volume = dx * dy * dz
        self.inner_volume = dx_inner * dy_inner * dz_inner
        self.volume = self.outer_volume - self.inner_volume

        self.density = self.mass / self.volume

        self.I_local = self.density / 12 * (self.outer_volume * np.array([
            [dy ** 2 + dz ** 2, 0, 0],
            [0, dx ** 2 + dz ** 2, 0],
            [0, 0, dx ** 2 + dy ** 2]
        ]) - self.inner_volume * np.array([
            [dy_inner ** 2 + dz_inner ** 2, 0, 0],
            [0, dx_inner ** 2 + dz_inner ** 2, 0],
            [0, 0, dx_inner ** 2 + dy_inner ** 2]
        ]))


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="This program computes inertial properties of an "
                                                "assembly of cubes.")
    parse.add_argument('-i', '--input', help='A file contains the description of the cubes.', required=True)
    parse.add_argument('-v', '--verbose', help='More verbose output', action="store_true")
    args = vars(parse.parse_args())

    input_dict = yaml.load(open(args['input']))

    # Unit
    if input_dict['unit'] == "mm":
        mult = 1e-3
    elif input_dict['unit'] == 'cm':
        mult = 1e-2
    elif input_dict['unit'] == 'm':
        mult = 1.0

    # Load objects
    all_objects = []
    for cube_ in input_dict:
        if cube_ == "unit":
            continue

        if input_dict[cube_]['kind'] == "cube":
            size = [mult * float(eval(str(s))) for s in input_dict[cube_]['size']]
            com_position = [mult * float(eval(str(s))) for s in input_dict[cube_]['com_position']]
            mass = float(eval(str(input_dict[cube_]['mass'])))
            all_objects.append(Cube(mass, size, com_position, name=cube_))
        elif input_dict[cube_]['kind'] == "hollow_cube":
            outsize = [mult * float(eval(str(s))) for s in input_dict[cube_]['outer_size']]
            com_position = [mult * float(eval(str(s))) for s in input_dict[cube_]['com_position']]
            mass = float(eval(str(input_dict[cube_]['mass'])))
            thickness = mult * float(eval(str(input_dict[cube_]['thickness'])))
            all_objects.append(HollowCube(mass, outsize, thickness, com_position, name=cube_))
        else:
            raise NotImplementedError

    # Visualize
    if args['verbose']:
        env = orpy.Environment()
        for obj in all_objects:
            obj.preview_in_rave(env)
        env.SetViewer('qtosg')

    # Start computing
    m_total = np.sum([obj.get_mass() for obj in all_objects])
    p_com = np.sum([obj.get_mass() * obj.get_com_position() for obj in all_objects], axis=0) / m_total
    I_total = np.sum([obj.compute_inertia_at_point(p_com) for obj in all_objects], axis=0)

    # Report
    print m_total, p_com
    print I_total * 1e6

    unit = raw_input("Preview in: [mm] ")
    if unit == "mm" or unit == "":
        unit = "mm"
        mult = 1e3
    elif unit == "m":
        mult = 1.
    else:
        raise NotImplementedError
    np.set_printoptions(formatter = {'float': lambda x: format(x, '6.3E')})
    print("\nMoment of Inertia about COM in kg {:}^2".format(unit))
    print(I_total * mult ** 2)
    print("\nMass in kg: \n{:f}".format(m_total))
    print("\nCOM position in {:}\n{:}".format(unit, p_com * mult))

