import sympy
from sympy import pprint, sqrt, cos, sin, latex

sympy.init_printing(use_unicode=False)
print("Sympy version {:}".format(sympy.__version__))


def main():
    x, y, z, theta_x, theta_y, theta_z = sympy.symbols('x, y, z, theta_x, theta_y, theta_z ')
    r, alpha = sympy.symbols('r, alpha')
    
    T_homo = sympy.Matrix([
        [1 - 0.5 * (theta_z ** 2 + theta_y ** 2), theta_x * theta_y / 2 - theta_z, theta_x * theta_z / 2 + theta_y, x],
        [theta_x * theta_y / 2 + theta_z, 1 - 0.5 * (theta_x ** 2 + theta_z ** 2), theta_y * theta_z / 2 - theta_x, y],
        [theta_x * theta_z / 2 - theta_y, theta_y * theta_z / 2 + theta_x, 1 - 0.5 * (theta_x ** 2 + theta_y ** 2), z],
        [0, 0, 0, 1]
    ])

    pos_original = sympy.Matrix([r * cos(alpha), r * sin(alpha), 0, 1])
    pos_new = T_homo * pos_original

    print ("Transformation matrix:")
    pprint(T_homo)

    print ("New position of a point on the circle:")
    pprint(pos_new)
    print(latex(pos_new))

if __name__ == '__main__':
    main()
