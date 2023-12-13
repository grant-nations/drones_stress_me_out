import numpy as np
import numpy.typing as npt
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


def random_direction() -> npt.ArrayLike:
    """
    This function generates a random direction in the theta, phi, and z
    directions.

    :return: a numpy array of shape (3,) containing the random direction
    """
    # create random direction
    direction = np.zeros(3)
    direction[0] = np.random.uniform(0, 2 * np.pi)
    direction[0] *= np.random.choice([-1, 1])
    direction[1] = np.random.uniform(0, np.pi / 2)
    direction[1] *= np.random.choice([-1, 1])
    direction[2] = np.random.choice([-1, 1])

    return direction


def random_speed(min_speed: float = 0.01,
                 max_speed: float = 0.5,
                 z_scale: float = 5.0) -> npt.ArrayLike:
    """
    This function generates a random speed (step size) in the theta, phi, and z directions.

    :return: a numpy array of shape (3,) containing the random speed
    """
    # create random speed
    velocity = np.zeros(3)
    velocity[0] = np.random.uniform(min_speed, max_speed)
    velocity[1] = np.random.uniform(min_speed, max_speed)
    velocity[2] = np.random.uniform(min_speed, max_speed) * z_scale

    return velocity


def random_motion(timepoints: int,
                  min_radius: float = 1,
                  max_radius: float = 10,
                  min_speed: float = 0.001,
                  max_speed: float = 0.015,
                  max_segment_duration: int = 100,
                  seed: int = 42,):
    """
    This function generates random motion in the theta, phi, and z directions
    that avoids entering the `human_radius` around the human, centered at
    theta = 0, phi = 0, z = 0. The random motion is repeated `repeat` times."""

    np.random.seed(seed)

    robot_positions = np.zeros((timepoints, 3))
    robot_velocities = np.zeros((timepoints, 3))

    # create random starting position
    starting_pos = np.zeros(3)
    starting_pos[0] = np.random.uniform(0, 2 * np.pi)
    starting_pos[1] = np.random.uniform(-np.pi/2, np.pi / 2)
    starting_pos[2] = np.random.uniform(min_radius, max_radius)

    robot_positions[0] = starting_pos
    robot_velocities[0] = np.zeros(3)

    t = 1
    while t < timepoints - 1:
        # get random direction to follow
        direction = random_direction()

        # get random speed at which to move
        speed = random_speed(min_speed=min_speed, max_speed=max_speed)

        # get number of time steps to move in this direction
        num_steps = np.random.randint(1, max_segment_duration)

        for i in range(num_steps):
            if t >= timepoints - 1:
                break

            next_pos = robot_positions[t - 1] + direction * speed

            # check if next position is within human radius
            if next_pos[2] < min_radius or next_pos[2] > max_radius:
                # skip the rest of this segment
                break
            else:
                robot_positions[t] = next_pos
                robot_velocities[t] = direction * speed
                t += 1

    return robot_positions, robot_velocities


if __name__ == "__main__":

    SEED = 43

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')

    timepoints = 1000
    drone_pos, drone_vel = random_motion(timepoints=timepoints, seed=SEED)

    theta = drone_pos[:, 0]
    phi = drone_pos[:, 1]
    z = drone_pos[:, 2]
    dtheta = drone_vel[:, 0]
    dphi = drone_vel[:, 1]
    dz = drone_vel[:, 2]

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
            robot_pos = drone_pos[i - ecg_response_offset]
            robot_vel = np.array([0, 0, 0]) if i - ecg_response_offset == 0 else drone_vel[i - ecg_response_offset]

            sim.update_ecg(robot_pos, robot_vel)
            ecg_data[i] = sim.ecg

        if i < eda_response_offset:
            eda_data[i] = np.random.normal(0, 0.001)

        else:
            robot_pos = drone_pos[i - eda_response_offset]
            robot_vel = np.array([0, 0, 0]) if i - eda_response_offset == 0 else drone_vel[i - eda_response_offset]

            sim.update_eda(robot_pos, robot_vel)
            eda_data[i] = sim.eda

        robot_pos = drone_pos[i]
        robot_vel = drone_vel[i]

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
