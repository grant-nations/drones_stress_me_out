import pandas as pd
import os
from models.lstm.stress_pred_lstm import StressPredictionLSTM
from utils.io import generate_unique_filename
import torch
from data.drone_bio_dataset import DroneBioDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":

    train_suffix = "-1"
    validation_suffix = "-2"

    demo_data_filename = f"demo_data{train_suffix}.csv"

    train_input_data_filename = f"drone_and_bio_input{train_suffix}.csv"
    train_labels_data_filename = f"stress_labels{train_suffix}.csv"
    train_mean_std_filename = f"mean_std{train_suffix}.json"

    validation_input_data_filename = f"drone_and_bio_input{validation_suffix}.csv"
    validation_labels_data_filename = f"stress_labels{validation_suffix}.csv"
    validation_mean_std_filename = f"mean_std{validation_suffix}.json"

    model_save_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
    model_save_path = generate_unique_filename(os.path.join(model_save_dir, 'lstm.pt'))

    # ------------ HYPERPARAMETERS ------------#
    hidden_dim = 128
    num_layers = 1
    dropout = 0
    epochs = 1000
    w_decay = 0.0001
    batch_size = 1  # NOTE: this is because we only have one sim right now
    shuffle_data = False  # NOTE: this is because we only have one sim right now
    # -----------------------------------------#

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Using device: {device}\n')

    # get training and validation data loaders

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')

    train_input_dataframe = pd.read_csv(os.path.join(data_dir, train_input_data_filename))
    train_labels_dataframe = pd.read_csv(os.path.join(data_dir, train_labels_data_filename))

    train_initial_stress_level = 0  # NOTE: this is hacky, but we only have one sim right now
    train_prev_stress_levels_df = train_labels_dataframe[['stress_level']].shift(1).fillna(train_initial_stress_level)
    demo_df = pd.read_csv(os.path.join(data_dir, demo_data_filename))

    training_data = DroneBioDataset([train_input_dataframe],
                                    [train_prev_stress_levels_df],
                                    [demo_df],
                                    [train_labels_dataframe])
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=shuffle_data)

    validation_input_dataframe = pd.read_csv(os.path.join(data_dir, validation_input_data_filename))
    validation_labels_dataframe = pd.read_csv(os.path.join(data_dir, validation_labels_data_filename))

    validation_initial_stress_level = 0  # NOTE: this is hacky, but we only have one sim right now
    validation_prev_stress_levels_df = validation_labels_dataframe[[
        'stress_level']].shift(1).fillna(validation_initial_stress_level)

    validation_data = DroneBioDataset([validation_input_dataframe],
                                      [validation_prev_stress_levels_df],
                                      [demo_df],
                                      [validation_labels_dataframe])
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=shuffle_data)

    # create model
    model = StressPredictionLSTM(hidden_dim=hidden_dim,
                                 device=device,
                                 num_layers=num_layers,
                                 dropout=dropout).to(device)

    print("Model architecture:")
    print(model)
    print()

    # train model
    print(f"Training model for {epochs} epochs...", end='', flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=w_decay)

    model.train_model(train_loader=train_loader,
                      val_loader=validation_loader,
                      optimizer=optimizer,
                      num_epochs=epochs,
                      patience=5,
                      min_delta=0.5,
                      print_every=10)
    print("Done.")

    # save model
    print(f'Saving model to {model_save_path}...', end='', flush=True)
    torch.save(model.state_dict(), model_save_path)

    print('Done.')

    # get training accuracy
    print('Getting training accuracy...', end='', flush=True)

    train_acc, train_loss = model.validate(train_loader)

    print(f'Done. Training accuracy: {train_acc}, Training loss: {train_loss}')
