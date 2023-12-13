import pandas as pd
import os
from models.lstm.stress_pred_lstm import StressPredictionLSTM
from utils.io import generate_unique_filename
import torch
from data.drone_bio_dataset import DroneBioDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":

    train_suffixes = ["", "-1", "-2", "-3", "-4", "-5", "-6", "-7"]
    validation_suffix = "-8"
    # test_suffix = "-9"

    demo_data_filename = f"demo_data.csv"  # NOTE: only one demo file for now (one sim)

    train_input_data_filename_prefix = f"drone_and_bio_input"
    train_labels_data_filename_prefix = f"stress_labels"

    validation_input_data_filename = f"drone_and_bio_input{validation_suffix}.csv"
    validation_labels_data_filename = f"stress_labels{validation_suffix}.csv"

    model_save_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
    model_save_path = generate_unique_filename(os.path.join(model_save_dir, 'lstm.pt'))

    # ------------ HYPERPARAMETERS ------------#
    hidden_dims = [128, 256, 512]
    num_layers = [2, 1]
    dropout = [0, 0.2, 0.5]
    epochs = [300]  # (max epochs)
    lr = [0.0001]
    w_decay = [0.001, 0.0001]
    batch_size = 7
    shuffle_data = True
    # -----------------------------------------#

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Using device: {device}\n')

    # GET TRAINING DATA LOADER

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')

    train_input_dfs = []
    train_prev_stress_levels_dfs = []
    train_demo_dfs = []
    train_labels_dfs = []

    for train_suffix in train_suffixes:
        train_input_dataframe = pd.read_csv(os.path.join(
            data_dir, train_input_data_filename_prefix + train_suffix + '.csv'))
        train_labels_dataframe = pd.read_csv(os.path.join(
            data_dir, train_labels_data_filename_prefix + train_suffix + '.csv'))

        train_initial_stress_level = 0  # NOTE: this is hack
        train_prev_stress_levels_df = train_labels_dataframe[[
            'stress_level']].shift(1).fillna(train_initial_stress_level)
        demo_df = pd.read_csv(os.path.join(data_dir, demo_data_filename))

        train_input_dfs.append(train_input_dataframe)
        train_prev_stress_levels_dfs.append(train_prev_stress_levels_df)
        train_demo_dfs.append(demo_df)
        train_labels_dfs.append(train_labels_dataframe)

    training_data = DroneBioDataset(train_input_dfs,
                                    train_prev_stress_levels_dfs,
                                    train_demo_dfs,
                                    train_labels_dfs)
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=shuffle_data)

    # GET VALIDATION DATA LOADER

    validation_input_dataframe = pd.read_csv(os.path.join(data_dir, validation_input_data_filename))
    validation_labels_dataframe = pd.read_csv(os.path.join(data_dir, validation_labels_data_filename))

    validation_initial_stress_level = 0  # NOTE: this is hack
    validation_prev_stress_levels_df = validation_labels_dataframe[[
        'stress_level']].shift(1).fillna(validation_initial_stress_level)

    validation_data = DroneBioDataset([validation_input_dataframe],
                                      [validation_prev_stress_levels_df],
                                      [demo_df],
                                      [validation_labels_dataframe])
    validation_loader = DataLoader(validation_data, batch_size=1, shuffle=False)

    best_model = None
    best_val_acc = 0

    for hidden_dim in hidden_dims:
        for layers in num_layers:
            for drop in dropout:
                for epoch in epochs:
                    for learning_rate in lr:
                        for decay in w_decay:
                            # create model
                            model = StressPredictionLSTM(hidden_dim=hidden_dim,
                                                         device=device,
                                                         num_layers=layers,
                                                         dropout=drop).to(device)

                            print("Model architecture:")
                            print(model)
                            print()

                            # train model
                            print(f"Training model for {epoch} epochs...", end='', flush=True)
                            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)

                            model.train_model(train_loader=train_loader,
                                              #   val_loader=validation_loader,
                                              val_loader=None,
                                              optimizer=optimizer,
                                              num_epochs=epoch,
                                            #   patience=20,
                                            #   min_delta=0.5,
                                              print_every=30)
                            print("Done.")

                            # get training accuracy
                            train_loader = DataLoader(training_data, batch_size=1, shuffle=False)
                            print('Getting training accuracy...', end='', flush=True)

                            train_acc, train_loss = model.validate(train_loader)

                            print(f'Done. Training accuracy: {train_acc}, Training loss: {train_loss}')

                            # get validation accuracy

                            print('Getting validation accuracy...', end='', flush=True)

                            val_acc, val_loss = model.validate(validation_loader)

                            print(f'Done. Validation accuracy: {val_acc}, Validation loss: {val_loss}')

                            if val_acc > best_val_acc:
                                best_val_acc = val_acc
                                best_model = model

    # save model
    print(f'Saving model to {model_save_path}...', end='', flush=True)
    torch.save(best_model.state_dict(), model_save_path)

    print('Done.')