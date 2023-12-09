import torch
import torch.nn as nn
from utils.early_stopper import EarlyStopper
from models.demo_nn import DemoNN

DEMOGRAPHIC_DATA_INPUT_DIM = 14  # collaborator demographic data input dimension
LSTM_INPUT_DIM = 9  # lstm input dimension
STRESS_LEVELS = 10  # number of stress levels


class StressPredictionLSTM(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 device: torch.device,
                 num_layers: int = 1,
                 dropout: int = 0) -> None:
        super(StressPredictionLSTM, self).__init__()
        """
        Initialize a stress prediction model that uses LSTM to predict stress levels
        on a scale of 1 to 10 at each time step from drone position, drone velocity,
        collaborator ECG, collaborator EDA, collaborator stress level at the previous time step. 
        It also encodes collaborator demographic data using a neural network for the initial hidden
        state of the LSTM.

        :param hidden_dim: The number of features in the hidden state h
        :param device: Device to run the model on
        :param output_dim: The number of features in the output y (default: 10 for 10 stress levels)
        :param num_layers: Number of recurrent layers (default: 1)
        :param dropout: Dropout rate between two consecutive layers (default: 0)
        """

        self.demo_input_dim = DEMOGRAPHIC_DATA_INPUT_DIM  # collaborator demographic data input dimension
        self.demo_model = DemoNN(self.demo_input_dim, hidden_dim)  # collaborator demographic data model

        # lstm input is comprised of drone position (theta, phi, z), drone velocity (dtheta, dphi, dz), collaborator ECG,
        # collaborator EDA, and collaborator stress level at the previous time step
        lstm_input_dim = LSTM_INPUT_DIM

        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)

        self.fc = nn.Linear(hidden_dim, STRESS_LEVELS)
        self.device = device

    def forward(self, X: torch.Tensor, stress_levels: torch.Tensor, demo: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass of the model with the given input

        :param X: sequence input to the model, tensor of shape (batch_size, seq_length, input_dim)
        :param stress_levels: collaborator stress levels at each time step, shifted by -1 to include the initial
        stress level, tensor of shape (batch_size, seq_length)
        :param demo: collaborator demographic data, tensor of shape (batch_size, demo_input_dim)

        :return: output of the model, tensor of shape (batch_size, seq_length, output_dim)
        """

        # encode collaborator demographic data using a neural network for the initial hidden state of the LSTM
        h0 = self.demo_model(demo).unsqueeze(0).to(self.device)  # h0 shape: (1, batch_size, hidden_size)
        c0 = torch.zeros_like(self.h0).to(self.device)  # c0 shape: (1, batch_size, hidden_size)

        # concat_input shape: (batch_size, seq_length, input_dim + 1)
        concat_input = torch.cat((X, stress_levels.unsqueeze(-1)), dim=-1)

        out, _ = self.lstm(concat_input, (h0, c0))  # out shape: (batch_size, seq_length, hidden_size)
        out = self.fc(out)  # out shape: (batch_size, seq_length, STRESS_LEVELS)

        return out

    def train_model(self,
                    train_loader: torch.utils.data.DataLoader,
                    val_loader: torch.utils.data.DataLoader | None,
                    loss_fn: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    num_epochs: int = 100,
                    patience: int = 5,
                    min_delta: float = 0.0,
                    print_every: int = 10) -> None:
        """
        Train the model

        :param train_loader: Training data loader
        :param val_loader: Validation data loader (optional)
        :param loss_fn: Loss function
        :param optimizer: Optimizer
        :param device: Device to run the model on
        :param num_epochs: Number of epochs to train the model for
        :param patience: Number of epochs to wait before early stopping
        :param min_delta: Minimum change in validation loss to be considered as improvement
        :param print_every: Print training and validation loss after every print_every epochs

        :return: None
        """
        early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader, loss_fn, optimizer, device)
            val_loss = None

            if val_loader is not None:
                val_loss = self.validate(val_loader, loss_fn, device)

                if early_stopper.early_stop(val_loss):
                    print(f'Early stopping. Epoch: {epoch + 1}')
                    break

            if (epoch + 1) % print_every == 0:
                if val_loader is not None:
                    print(f'Epoch: {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}')
                else:
                    print(f'Epoch: {epoch + 1}/{num_epochs}, Train Loss: {train_loss}')

    def train_epoch(self,
                    train_loader: torch.utils.data.DataLoader,
                    loss_fn: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device) -> float:
        """
        Train the model for one epoch

        :param train_loader: Training data loader
        :param loss_fn: Loss function
        :param optimizer: Optimizer
        :param device: Device to run the model on

        :return: Average loss for the epoch
        """
        self.train()
        train_loss = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # forward pass
            output = self(data).permute(0, 2, 1)  # output shape: (batch_size, STRESS_LEVELS, seq_length)

            # calculate loss
            loss = loss_fn(output, target)

            # backprop
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader.dataset)
        return train_loss

    def validate(self,
                 val_loader: torch.utils.data.DataLoader,
                 loss_fn: nn.Module,
                 device: torch.device) -> float:
        """
        Validate the model

        :param val_loader: Validation data loader
        :param loss_fn: Loss function
        :param device: Device to run the model on

        :return: Average loss for the epoch
        """
        self.eval()
        val_loss = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)

                # forward pass
                output = self(data).permute(0, 2, 1)

                # calculate loss
                loss = loss_fn(output, target)
                val_loss += loss.item()

        val_loss /= len(val_loader.dataset)

        return val_loss
