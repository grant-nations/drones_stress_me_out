from typing import Dict, Any, Tuple
import numpy as np
import numpy.typing as npt


class Sim:
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
            'ecg': self.gen_ecg(robot_pos, robot_vel),
            'eda': self.gen_eda(robot_pos, robot_vel),
            'stress_level': self.gen_stress_level(robot_pos, robot_vel)
        }

    def gen_ecg(self,
                robot_pos: npt.ArrayLike,
                robot_vel: npt.ArrayLike,
                prev_ecg: float) -> float:
        """
        Generate ECG data for a given robot position and velocity

        :param robot_pos: Robot position (theta, phi, z)
        :param robot_vel: Robot velocity (theta, phi, z)
        :param prev_ecg: Previous ECG data

        :return: ECG data
        """

        # I don't know what the actual scale of ECG data is, so here
        # I use a scale of 0 to 1

        occupation_score = 0 if self.occupation in ['student', 'engineer', 'scientist'] else 1
        robot_score = 0 if self.robot_experience == 'no' else 1
        age_score = 1 if self.age < 20 or self.age > 40 else 0

        ecg_delta = np.random.normal(0, 0.1) + occupation_score + robot_score + age_score
        ecg_delta *= np.linalg.norm(robot_vel) / 10 + np.linalg.norm(robot_pos) / 10

        return 1 / (1 + np.exp(-(prev_ecg + ecg_delta)))

    def gen_eda(self,
                robot_pos: Tuple[float, float, float],
                robot_vel: Tuple[float, float, float],
                prev_eda: float) -> float:
        """
        Generate EDA data for a given robot position and velocity

        :param robot_pos: Robot position (theta, phi, z)
        :param robot_vel: Robot velocity (theta, phi, z)
        :param prev_eda: Previous EDA data

        :return: EDA data
        """
        return 0  # TODO: implement

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
