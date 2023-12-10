import numpy as np
from data_generation.drone_sim import DroneSim
from typing import Tuple
from data_generation.sim import Sim
import pandas as pd
import os
from utils.io import generate_unique_filename


def spiral_motion(z_offset: float = 1.0, repeat: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This function generates a spiral motion in the theta, phi, and z directions
    that starts at theta = 0, phi = 0, z = `z_offset` and ends at theta = 2pi,
    phi = pi/2, z = 10 + `z_offset`, then returns to the starting position. The
    spiral motion is repeated `repeat` times.

    :param z_offset: Offset in the z direction
    :param repeat: Number of times to repeat the spiral motion

    :return: theta, phi, z
    """
    t = 0.01 * np.arange(100)

    theta = t * 2 * np.pi
    phi = t * np.pi / 2
    z = (1 - t) * 10 + z_offset

    theta = np.concatenate((theta, theta))
    phi = np.concatenate((phi, phi[::-1]))
    z = np.concatenate((z, z[::-1]))

    theta_copy = theta.copy()
    phi_copy = phi.copy()
    z_copy = z.copy()

    for _ in range(repeat - 1):
        theta = np.concatenate((theta, theta_copy))
        phi = np.concatenate((phi, phi_copy))
        z = np.concatenate((z, z_copy))

    return theta, phi, z


if __name__ == "__main__":

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')

    # SIMULATE DRONE MOTION
    theta, phi, z = spiral_motion(repeat=3)
    timepoints = len(theta)

    drone_sim = DroneSim(theta=theta[0], phi=phi[0], z=z[0])
    init_pos, init_vel = drone_sim.get_state()

    dtheta = np.zeros(timepoints)
    dphi = np.zeros(timepoints)
    dz = np.zeros(timepoints)

    for i in range(timepoints):
        if i == 0:
            dtheta[i] = 0
            dphi[i] = 0
            dz[i] = 0
        else:
            dtheta[i] = theta[i] - theta[i - 1]
            dphi[i] = phi[i] - phi[i - 1]
            dz[i] = z[i] - z[i - 1]

    # SIMULATE BIOFEEDBACK DATA
    sim = Sim(age=23,
              gender="male",
              height=176,
              weight=325,
              income=0,
              education="college",
              occupation="unemployed",
              marital_status="relationship",
              robot_experience="no")

    sim_demo_df = pd.DataFrame(sim.to_dict(), index=[0])

    filename = os.path.join(data_dir, 'demo_data.csv')
    filename = generate_unique_filename(filename)

    sim_demo_df.to_csv(filename, index=False)

    ecg_response_offset = 5  # time offset between drone position and ECG response
    eda_response_offset = 15  # time offset between drone position and EDA response

    time_data = np.arange(timepoints)

    ecg_data = np.zeros(timepoints)
    eda_data = np.zeros(timepoints)
    stress_data = np.zeros(timepoints)

    for i in range(timepoints):

        if i < ecg_response_offset:
            ecg_data[i] = np.random.normal(0, 0.001)
        else:
            robot_pos = np.array(
                [theta[i - ecg_response_offset],
                 phi[i - ecg_response_offset],
                 z[i - ecg_response_offset]])

            robot_vel = np.array([0, 0, 0]) if i - ecg_response_offset == 0 else np.array(
                [theta[i - ecg_response_offset] - theta[i - ecg_response_offset - 1],
                 phi[i - ecg_response_offset] - phi[i - ecg_response_offset - 1],
                 z[i - ecg_response_offset] - z[i - ecg_response_offset - 1]])

            sim.update_ecg(robot_pos, robot_vel)
            ecg_data[i] = sim.ecg

        if i < eda_response_offset:
            eda_data[i] = np.random.normal(0, 0.001)

        else:
            robot_pos = np.array(
                [theta[i - eda_response_offset],
                 phi[i - eda_response_offset],
                 z[i - eda_response_offset]])

            robot_vel = np.array([0, 0, 0]) if i - eda_response_offset == 0 else np.array(
                [theta[i - eda_response_offset] - theta[i - eda_response_offset - 1],
                 phi[i - eda_response_offset] - phi[i - eda_response_offset - 1],
                 z[i - eda_response_offset] - z[i - eda_response_offset - 1]])

            sim.update_eda(robot_pos, robot_vel)
            eda_data[i] = sim.eda

        robot_pos = np.array([theta[i], phi[i], z[i]])

        robot_vel = np.array([0, 0, 0]) if i == 0 else np.array(
            [theta[i] - theta[i - 1],
             phi[i] - phi[i - 1],
             z[i] - z[i - 1]])

        sim.update_stress_level(robot_pos, robot_vel)
        stress_data[i] = sim.stress_level

    # SAVE DATA

    input_df = pd.DataFrame({
        'theta': theta,
        'phi': phi,
        'z': z,
        'dtheta': dtheta,
        'dphi': dphi,
        'dz': dz,
        'ecg': ecg_data,
        'eda': eda_data,
    })

    stress_df = pd.DataFrame({
        'stress_level': stress_data
    })

    filename = os.path.join(data_dir, 'stress_labels.csv')
    filename = generate_unique_filename(filename)

    stress_df.to_csv(filename, index=False)

    filename = os.path.join(data_dir, 'drone_and_bio_input.csv')
    filename = generate_unique_filename(filename)

    input_df.to_csv(filename, index=False)
