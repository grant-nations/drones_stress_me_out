import pandas as pd
import os
from models.lstm.stress_pred_lstm import StressPredictionLSTM
from utils.io import generate_unique_filename
import torch
from data.drone_bio_dataset import DroneBioDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":

    model_save_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
    model_save_path = generate_unique_filename(os.path.join(model_save_dir, 'lstm.pt'))

    # ------------ HYPERPARAMETERS ------------#
    hidden_dim = 128
    num_layers = 1
    dropout = 0
    epochs = 100
    w_decay = 0.0001
    batch_size = 1  # NOTE: this is because we only have one sim right now
    shuffle_data = False  # NOTE: this is because we only have one sim right now
    # -----------------------------------------#

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Using device: {device}\n')

    # get training and validation data loaders

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')

    input_dataframe = pd.read_csv(os.path.join(data_dir, 'drone_and_bio_input.csv'))
    labels_dataframe = pd.read_csv(os.path.join(data_dir, 'stress_labels.csv'))

    initial_stress_level = 1  # NOTE: this is hacky, but we only have one sim right now
    prev_stress_levels_df = labels_dataframe[['stress_level']].shift(1).fillna(initial_stress_level)
    demo_df = pd.read_csv(os.path.join(data_dir, 'demo_data.csv'))

    # NOTE: because we only have one sim right now, we only have one input dataframe
    training_data = DroneBioDataset([input_dataframe],
                                    [prev_stress_levels_df],
                                    [demo_df],
                                    [labels_dataframe])
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=shuffle_data)

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
                      val_loader=None,
                      optimizer=optimizer,
                      num_epochs=epochs,
                      print_every=10)
    print("Done.")

    # save model
    print(f'Saving model to {model_save_path}...', end='', flush=True)
    torch.save(model.state_dict(), model_save_path)

    print('Done.')

    # get training accuracy
    print('Getting training accuracy...', end='', flush=True)

    val_data = DroneBioDataset([input_dataframe],
                               [prev_stress_levels_df],
                               [demo_df])
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=shuffle_data)

    train_acc, train_loss = model.validate(val_loader)

    print(f'Done. Training accuracy: {train_acc}, Training loss: {train_loss}')