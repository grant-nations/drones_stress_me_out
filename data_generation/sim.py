from typing import Dict, Any, Tuple
import numpy as np
import numpy.typing as npt


class Sim:

    _ECG_COOLDOWN = 10  # time steps needed to lower ECG
    _EDA_COOLDOWN = 50  # time steps needed to lower EDA

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
        # self.prev_ecg = self.ecg
        self.ecg_cooldown = Sim._ECG_COOLDOWN  # time steps needed to lower ECG

        self.eda = 0
        self.eda_cooldown = Sim._EDA_COOLDOWN  # time steps needed to lower EDA

        self.stress_level = 0  # minimal stress

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

        ecg = ((10 - z) ** stress_score) / 1.5 + 5 * np.exp(dtheta * dphi * dz) + \
            1 ** (theta * phi * z) + 3 * np.random.normal(0, 0.4)
        ecg /= 100

        if ecg < self.ecg - 0.01:
            if self.ecg_cooldown <= 0:
                cooldown_ecg = self.ecg - 0.001

                if cooldown_ecg < ecg:
                    self.ecg_cooldown = Sim._ECG_COOLDOWN

                    # self.prev_ecg = self.ecg
                    self.ecg = ecg
                else:
                    # self.prev_ecg = self.ecg
                    self.ecg = cooldown_ecg + np.random.normal(0, 0.005)
            else:
                self.ecg_cooldown -= 1
        else:
            # self.prev_ecg = self.ecg
            self.ecg = ecg

            if self.ecg_cooldown < Sim._ECG_COOLDOWN:
                self.ecg_cooldown = Sim._ECG_COOLDOWN

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

        occupation_score = 0 if self.occupation in ['student', 'engineer', 'scientist'] else 1
        robot_score = 0 if self.robot_experience == 'yes' else 1
        age_score = 1 if self.age < 20 or self.age > 40 else 0
        bmi = self.weight / (self.height ** 2) * 703
        bmi_score = 1 if bmi < 4 or bmi > 7 else 0

        stress_score = occupation_score + robot_score + age_score + bmi_score

        eda = ((10 - z) ** stress_score) / 4 + 1 ** (theta * phi * z)
        eda /= 70

        if eda < self.eda:
            eda += np.random.normal(0, 0.01)
            
            if self.eda_cooldown <= 0:
                cooldown_eda = self.eda - 0.0005

                if cooldown_eda < eda:
                    self.eda_cooldown = Sim._ECG_COOLDOWN
                    self.eda = eda
                else:
                    self.eda = cooldown_eda + np.random.normal(0, 0.01)
            else:
                self.eda_cooldown -= 1
        else:
            self.eda = eda

            if self.eda_cooldown < Sim._ECG_COOLDOWN:
                self.eda_cooldown = Sim._ECG_COOLDOWN

    def update_stress_level(self,
                            robot_pos: Tuple[float, float, float],
                            robot_vel: Tuple[float, float, float]) -> None:
        """
        Generate stress level data for a given robot position and velocity

        :param robot_pos: Robot position (theta, phi, z)
        :param robot_vel: Robot velocity (theta, phi, z)

        :return: Stress level data
        """

        # stress level is on scale of 0 to 9 inclusive

        theta, phi, z = robot_pos
        dtheta, dphi, dz = robot_vel

        marital_status_score = 1 if self.marital_status == 'single' else 0
        income_score = 1 if self.income < 100000 else 0

        stress_score = marital_status_score + income_score

        stress_level = ((9 - z) ** stress_score) + 0.2 * dz + np.abs(np.pi - theta)
        stress_level += self.ecg + 2 * self.eda
        stress_level /= 2.5

        self.stress_level = np.min([stress_level, 9])

    def to_dict(self):
        return {
            "age": self.age,
            "gender": self.gender,
            "height": self.height,
            "weight": self.weight,
            "income": self.income,
            "education": self.education,
            "occupation": self.occupation,
            "marital_status": self.marital_status,
            "robot_experience": self.robot_experience
        }
