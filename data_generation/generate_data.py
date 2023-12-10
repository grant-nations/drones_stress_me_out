import numpy as np
import numpy.typing as npt
from data_generation.drone_sim import DroneSim
from typing import Tuple
from utils.spherical_to_cartesian import spherical_to_cartesian
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from data_generation.sim import Sim


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
    z = (1- t) * 10 + z_offset

    theta = np.concatenate((theta, theta))
    phi = np.concatenate((phi, phi[::-1]))
    z = np.concatenate((z, z[::-1]))

    for _ in range(repeat - 1):
        theta = np.concatenate((theta, theta))
        phi = np.concatenate((phi, phi))
        z = np.concatenate((z, z))

    return theta, phi, z


def update_drone_frame(frame: int,
                       theta: np.ndarray,
                       phi: np.ndarray,
                       z: np.ndarray,
                       drone_sim: DroneSim,
                       line: Axes3D) -> Axes3D:
    """
    Update the frame of the animation

    :param frame: Frame number
    :param theta: Theta
    :param phi: Phi
    :param z: Z
    :param drone_sim: Drone simulator
    :param line: Line

    :return: Line
    """
    drone_sim.move_to(theta[frame], phi[frame], z[frame])

    pos, vel = drone_sim.get_state()

    x, y, z = spherical_to_cartesian(*pos)

    line.set_data(x, y)
    line.set_3d_properties(z)

    return line,


def update_ecg_frame(frame: int,
                     ecg_data: npt.ArrayLike,
                     time_data: npt.ArrayLike,
                     line: Axes3D) -> Axes3D:
    """
    Update the frame of the animation

    :param frame: Frame number
    :param ecg_data: ECG data
    :param time_data: Timestamps
    :param line: Line

    :return: Line
    """

    line.set_data(time_data[:frame], ecg_data[:frame])
    return line,


if __name__ == "__main__":

    # SIMULATE DRONE MOTION
    theta, phi, z = spiral_motion(repeat=1)

    drone_sim = DroneSim(theta=theta[0], phi=phi[0], z=z[0])
    init_pos, init_vel = drone_sim.get_state()

    # SIMULATE ECG DATA
    sim = Sim(age=23,
              gender="male",
              height=175,
              weight=225,
              income=0,
              education="college",
              occupation="unemployed",
              marital_status="relationship",
              robot_experience="no")

    time_data = np.arange(len(theta))
    ecg_data = np.zeros(len(theta))

    for i in range(len(theta)):
        robot_pos = np.array([theta[i], phi[i], z[i]])
        robot_vel = np.array([0, 0, 0])
        sim.update_ecg(robot_pos, robot_vel)
        ecg_data[i] = sim.ecg

    # SIMULATE EDA DATA

    # TODO

    # SIMULATE STRESS LEVEL DATA

    # TODO

    # CREATE DRONE ANIMATION

    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    line_drone, = ax1.plot(init_pos[0], init_pos[1], init_pos[2], 'X', color='r', alpha=1, lw=3, label='Drone')

    # Plot human position with blue 'o'
    line_human, = ax1.plot(0, 0, 0, 'o', color='b', alpha=1, lw=3, label='Human')

    ax1.set_xlim(-10, 10)
    ax1.set_ylim(-10, 10)
    ax1.set_zlim(-10, 10)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    ax1.legend()  # Add a legend to differentiate between drone and human positions

    # CREATE ECG ANIMATION

    ax2 = fig.add_subplot(1, 2, 2)
    line_ecg, = ax2.plot(time_data[0], ecg_data[0], color='purple', alpha=1, lw=2, label='ECG')

    ax2.set_xlim(0, len(theta))
    ax2.set_ylim(0, 1)

    ax2.set_xlabel('Time')
    ax2.set_ylabel('ECG')

    ax2.legend()

    # CREATE EDA ANIMATION

    # TODO

    # CREATE STRESS LEVEL ANIMATION

    # TODO

    # RUN ANIMATIONS

    drone_ani = FuncAnimation(fig,
                              update_drone_frame,
                              frames=len(theta),
                              fargs=(theta, phi, z, drone_sim, line_drone),
                              interval=50,
                              blit=True,
                              repeat=False)

    ecg_ani = FuncAnimation(fig,
                            update_ecg_frame,
                            frames=len(theta),
                            fargs=(ecg_data, time_data, line_ecg),
                            interval=50,
                            blit=True,
                            repeat=False)

    plt.show()