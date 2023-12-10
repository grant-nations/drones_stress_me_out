import numpy as np
import numpy.typing as npt
from data_generation.drone_sim import DroneSim
from typing import Tuple
from utils.calculations import spherical_to_cartesian
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from data_generation.sim import Sim
import pandas as pd
import os
from utils.io import generate_unique_filename


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


def update_data_frame(frame: int,
                      ecg_data: npt.ArrayLike,
                      time_data: npt.ArrayLike,
                      line: Axes3D) -> Axes3D:
    """
    Update the frame of the animation

    :param frame: Frame number
    :param ecg_data: ECG or EDA data
    :param time_data: Timestamps
    :param line: Line

    :return: Line
    """

    line.set_data(time_data[:frame], ecg_data[:frame])
    return line,


if __name__ == "__main__":
    # CREATE DRONE ANIMATION

    fig = plt.figure(figsize=(10, 10))

    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
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

    ax2 = fig.add_subplot(2, 2, 2)
    line_ecg, = ax2.plot(time_data[0], ecg_data[0], color='purple', alpha=1, lw=2, label='ECG')

    ax2.set_xlim(0, timepoints)
    ax2.set_ylim(0, 1)

    ax2.set_xlabel('Time')
    ax2.set_ylabel('ECG')

    ax2.legend()

    # CREATE EDA ANIMATION

    ax3 = fig.add_subplot(2, 2, 3)
    line_eda, = ax3.plot(time_data[0], eda_data[0], color='green', alpha=1, lw=2, label='EDA')

    ax3.set_xlim(0, timepoints)
    ax3.set_ylim(0, 10)

    ax3.set_xlabel('Time')
    ax3.set_ylabel('EDA')

    ax3.legend()

    # CREATE STRESS LEVEL ANIMATION

    ax4 = fig.add_subplot(2, 2, 4)
    line_stress, = ax4.plot(time_data[0], eda_data[0], color='red', alpha=1, lw=2, label='Stress Level')

    ax4.set_xlim(0, timepoints)
    ax4.set_ylim(0, 10)

    ax4.set_xlabel('Time')
    ax4.set_ylabel('Stress Level')

    ax4.legend()

    # RUN ANIMATIONS
    repeat = False

    drone_ani = FuncAnimation(fig,
                              update_drone_frame,
                              frames=timepoints,
                              fargs=(theta, phi, z, drone_sim, line_drone),
                              interval=50,
                              blit=True,
                              repeat=repeat)

    ecg_ani = FuncAnimation(fig,
                            update_data_frame,
                            frames=timepoints,
                            fargs=(ecg_data, time_data, line_ecg),
                            interval=50,
                            blit=True,
                            repeat=repeat)

    eda_ani = FuncAnimation(fig,
                            update_data_frame,
                            frames=timepoints,
                            fargs=(eda_data, time_data, line_eda),
                            interval=50,
                            blit=True,
                            repeat=repeat)

    stress_ani = FuncAnimation(fig,
                               update_data_frame,
                               frames=timepoints,
                               fargs=(stress_data, time_data, line_stress),
                               interval=50,
                               blit=True,
                               repeat=repeat)
    plt.show()
