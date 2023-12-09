import numpy as np
import numpy.typing as npt
from typing import Tuple


class DroneSim:
    def __init__(self,
                 theta: float = 0.0,
                 phi: float = 0.0,
                 z: float = 0.0,
                 dtheta: float = 0.0,
                 dphi: float = 0.0,
                 dz: float = 0.0
                 ) -> None:
        """
        Initialize the drone simulator
        """
        self.theta = theta
        self.phi = phi
        self.z = z
        self.dtheta = dtheta
        self.dphi = dphi
        self.dz = dz

    def update(self, action: npt.ArrayLike) -> None:
        """
        Update the drone position and velocity based on the given action

        :param action: Action to take, (dtheta, dphi, dz)
        """
        dtheta, dphi, dz = action
        self.theta += dtheta
        self.phi += dphi
        self.z += dz
        self.dtheta = dtheta
        self.dphi = dphi
        self.dz = dz

    def get_state(self) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        """
        Get the current drone position and velocity

        :return: Current drone position and velocity, (theta, phi, z, dtheta, dphi, dz)
        """
        return np.array([self.theta, self.phi, self.z]), np.array([self.dtheta, self.dphi, self.dz])

    def set_state(self,
                  theta: float = 0.0,
                  phi: float = 0.0,
                  z: float = 0.0,
                  dtheta: float = 0.0,
                  dphi: float = 0.0,
                  dz: float = 0.0
                  ) -> None:
        """
        Set the current drone position and velocity

        :param theta: Theta
        :param phi: Phi
        :param z: Z
        :param dtheta: dTheta
        :param dphi: dPhi
        :param dz: dZ
        """
        self.theta = theta
        self.phi = phi
        self.z = z
        self.dtheta = dtheta
        self.dphi = dphi
        self.dz = dz
