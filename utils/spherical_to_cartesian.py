import numpy as np


def spherical_to_cartesian(theta, phi, z):
    """
    Convert spherical coordinates to cartesian coordinates

    :param theta: Theta
    :param phi: Phi
    :param z: Z

    :return: Cartesian coordinates, (x, y, z)
    """
    x = z * np.cos(theta) * np.sin(phi)
    y = z * np.sin(theta) * np.sin(phi)
    z = z * np.cos(phi)

    return x, y, z
