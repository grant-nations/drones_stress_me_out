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
from models.lstm.stress_pred_lstm import StressPredictionLSTM
import torch
import json


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
                      data: npt.ArrayLike,
                      time_data: npt.ArrayLike,
                      line: Axes3D) -> Axes3D:
    """
    Update the frame of the animation

    :param frame: Frame number
    :param data: ECG or EDA data
    :param time_data: Timestamps
    :param line: Line

    :return: Line
    """

    line.set_data(time_data[:frame], data[:frame])
    return line,


def update_stress_data_frame(frame: int,
                             stress_data: npt.ArrayLike,
                             prediction_data: npt.ArrayLike,
                             time_data: npt.ArrayLike,
                             line_actual: Axes3D,
                             line_predicted: Axes3D) -> Tuple[Axes3D, Axes3D]:
    """
    Update the frame of the stress level animation

    :param frame: Frame number
    :param stress_data: Stress level data
    :param prediction_data: Predicted stress level data
    :param time_data: Timestamps
    :param line_actual: Actual stress level line
    :param line_predicted: Predicted stress level line

    :return: Actual stress level line, predicted stress level line
    """

    line_actual.set_data(time_data[:frame], stress_data[:frame])
    line_predicted.set_data(time_data[:frame], prediction_data[:frame])

    return line_actual, line_predicted


if __name__ == "__main__":
    TRAINED_MODEL_NAME = 'lstm.pt'

    # GET SIMULATION DATA
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    raw_data_dir = os.path.join(data_dir, 'raw')
    processed_data_dir = os.path.join(data_dir, 'processed')

    drone_bio_input_dataframe = pd.read_csv(os.path.join(raw_data_dir, 'drone_and_bio_input.csv'))
    stress_labels_dataframe = pd.read_csv(os.path.join(raw_data_dir, 'stress_labels.csv'))

    theta = drone_bio_input_dataframe['theta'].values
    phi = drone_bio_input_dataframe['phi'].values
    z = drone_bio_input_dataframe['z'].values

    dtheta = drone_bio_input_dataframe['dtheta'].values
    dphi = drone_bio_input_dataframe['dphi'].values
    dz = drone_bio_input_dataframe['dz'].values

    ecg_data = drone_bio_input_dataframe['ecg'].values
    eda_data = drone_bio_input_dataframe['eda'].values

    stress_data = stress_labels_dataframe['stress_level'].values

    timepoints = len(theta)
    time_data = np.arange(timepoints)

    # CREATE STRESS PREDICTION MODEL
    demo_dataframe = pd.read_csv(os.path.join(processed_data_dir, 'demo_data.csv'))
    initial_stress_level = 1  # NOTE: this is hacky, but we only have one sim right now

    trained_model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'training', 'saved_models')
    trained_model = StressPredictionLSTM(hidden_dim=128,
                                         device=torch.device('cpu'),
                                         num_layers=1,
                                         dropout=0)

    trained_model.load_state_dict(torch.load(os.path.join(trained_model_dir, TRAINED_MODEL_NAME)))

    # PREDICT STRESS LEVELS

    # get mean and std from training data

    mean_sdt = {}

    with open(os.path.join(processed_data_dir, 'mean_std.json'), 'r') as f:
        mean_std = json.load(f)

    h0, c0 = trained_model.encode_demographics(torch.from_numpy(demo_dataframe.values.squeeze().astype(np.float32)))
    predicted_stress_data = np.zeros(timepoints)

    correct = 0
    total = 0

    prev_stress_level = initial_stress_level
    for i in range(timepoints):
        drone_bio_input = np.array(
            [theta[i],
             phi[i],
             z[i],
             dtheta[i],
             dphi[i],
             dz[i],
             ecg_data[i],
             eda_data[i]]) 
        
        drone_bio_input = (drone_bio_input - mean_std['mean']) / mean_std['std']

        drone_bio_input = torch.from_numpy(drone_bio_input.astype(np.float32))

        stress_prediction, h0, c0 = trained_model.predict(h0, c0, drone_bio_input, prev_stress_level)
        predicted_stress_data[i] = stress_prediction

        prev_stress_level = stress_prediction

        if stress_prediction == np.round(stress_data[i]):
            correct += 1

        total += 1

    print(f'Accuracy: {correct / total}')

    # CREATE DRONE ANIMATION
    drone_sim = DroneSim(theta=theta[0], phi=phi[0], z=z[0])
    init_pos, init_vel = drone_sim.get_state()

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
    line_stress_actual, = ax4.plot(time_data[0], stress_data[0], color='red', alpha=1, lw=2, label='Stress Level')
    line_stress_predicted, = ax4.plot(time_data[0], predicted_stress_data[0], color='black', alpha=0.6, lw=2,
                                      label='Predicted Stress Level')

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

    stress_ani = FuncAnimation(
        fig, update_stress_data_frame, frames=timepoints,
        fargs=(stress_data, predicted_stress_data, time_data, line_stress_actual, line_stress_predicted),
        interval=50, blit=True, repeat=repeat)

    plt.tight_layout()

    plt.show()
