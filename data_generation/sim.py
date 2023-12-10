from typing import Dict, Any, Tuple
import numpy as np
import numpy.typing as npt


class Sim:

    _ECG_COOLDOWN = 10  # time steps needed to lower ECG
    _EDA_COOLDOWN = 100  # time steps needed to lower EDA

    def __init__(self,
                 age: float,
                 gender: str,
                 height: float,
                 weight: float,
                 income: float,
                 education: str,
                 occupation: str,
                 marital_status: str,
                 robot_experience: str) -> None:

        self.age = age
        self.gender = gender
        self.height = height
        self.weight = weight
        self.income = income
        self.education = education
        self.occupation = occupation
        self.marital_status = marital_status
        self.robot_experience = robot_experience

        self.ecg = 0
        self.ecg_cooldown = Sim._ECG_COOLDOWN  # time steps needed to lower ECG

        self.eda = 0
        self.eda_cooldown = Sim._EDA_COOLDOWN  # time steps needed to lower EDA

        self.stress_level = 1  # minimal stress

    def gen_biofeedback(self,
                        robot_pos: Tuple[float, float, float],
                        robot_vel: Tuple[float, float, float]) -> Dict[str, Any]:
        """
        Generate biofeedback data for a given robot position and velocity

        :param robot_pos: Robot position (theta, phi, z)
        :param robot_vel: Robot velocity (theta, phi, z)

        :return: Biofeedback data
        """
        return {
            'ecg': self.update_ecg(robot_pos, robot_vel),
            'eda': self.gen_eda(robot_pos, robot_vel),
            'stress_level': self.gen_stress_level(robot_pos, robot_vel)
        }

    def update_ecg(self,
                   robot_pos: npt.ArrayLike,
                   robot_vel: npt.ArrayLike) -> None:
        """
        Update ECG data for a given robot position and velocity

        :param robot_pos: Robot position (theta, phi, z)
        :param robot_vel: Robot velocity (theta, phi, z)
        """

        # I don't know what the actual scale of ECG data is, so here
        # I use a scale of 0 to 1

        occupation_score = 0 if self.occupation in ['student', 'engineer', 'scientist'] else 1
        robot_score = 0 if self.robot_experience == 'yes' else 1
        age_score = 1 if self.age < 20 or self.age > 40 else 0

        stress_score = occupation_score + robot_score + age_score

        theta, phi, z = robot_pos
        dtheta, dphi, dz = robot_vel

        ecg = ((10 - z) ** stress_score) / 1.5 + np.exp(dtheta * dphi * dz) + 1 ** (theta * phi * z) + 3 * np.random.normal(0, 0.4)
        ecg /= 100

        if ecg < self.ecg - 0.01:
            self.ecg_cooldown -= 1

            if self.ecg_cooldown <= 0:
                self.ecg_cooldown = Sim._ECG_COOLDOWN
                self.ecg = np.max([0, ecg])
        else:
            self.ecg = ecg

    def update_eda(self,
                   robot_pos: Tuple[float, float, float],
                   robot_vel: Tuple[float, float, float]) -> float:
        """
        Update EDA data for a given robot position and velocity

        :param robot_pos: Robot position (theta, phi, z)
        :param robot_vel: Robot velocity (theta, phi, z)

        :return: EDA data
        """

        # I don't know what the actual scale of EDA data is, so here
        # I use a scale of 0 to 10 to have some variation from ECG

        theta, phi, z = robot_pos
        dtheta, dphi, dz = robot_vel

        

    def gen_stress_level(self,
                         robot_pos: Tuple[float, float, float],
                         robot_vel: Tuple[float, float, float],
                         prev_stress: float) -> float:
        """
        Generate stress level data for a given robot position and velocity

        :param robot_pos: Robot position (theta, phi, z)
        :param robot_vel: Robot velocity (theta, phi, z)
        :param prev_stress: Previous stress level data

        :return: Stress level data
        """
        return 0  # TODO: implement
