import torch
import torch.nn as nn
from utils.early_stopper import EarlyStopper
from models.lstm.demo_nn import DemoNN
from typing import Tuple, Union

DEMOGRAPHIC_DATA_INPUT_DIM = 9  # collaborator demographic data input dimension
LSTM_INPUT_DIM = 9  # lstm input dimension (theta, phi, z, dtheta, dphi, dz, ecg, eda, stress level)
STRESS_LEVELS = 10  # number of stress levels (1 to 10)


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

        self.num_layers = num_layers

    def forward(self, bio_drone_data: torch.Tensor, stress_levels: torch.Tensor, demo: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass of the model with the given input

        :param bio_drone_data: sequence input to the model, tensor of shape (batch_size, seq_length, input_dim)
        :param stress_levels: collaborator stress levels at each time step, shifted by -1 to include the initial
        stress level, tensor of shape (batch_size, seq_length)
        :param demo: collaborator demographic data, tensor of shape (batch_size, demo_input_dim)

        :return: output of the model, tensor of shape (batch_size, seq_length, output_dim)
        """

        # encode collaborator demographic data using a neural network for the initial hidden state of the LSTM
        h0 = self.demo_model(demo).unsqueeze(0).to(self.device)  # h0 shape: (1, batch_size, hidden_size)
        c0 = torch.zeros_like(h0).to(self.device)  # c0 shape: (1, batch_size, hidden_size)

        h0 = h0.repeat(self.num_layers, 1, 1)  # h0 shape: (num_layers, batch_size, hidden_size)
        c0 = c0.repeat(self.num_layers, 1, 1)  # c0 shape: (num_layers, batch_size, hidden_size)

        # concat_input shape: (batch_size, seq_length, input_dim + 1)
        concat_input = torch.cat((bio_drone_data, stress_levels.unsqueeze(-1)), dim=-1)

        out, _ = self.lstm(concat_input, (h0, c0))  # out shape: (batch_size, seq_length, hidden_size)
        out = self.fc(out)  # out shape: (batch_size, seq_length, STRESS_LEVELS)

        return out

    def train_model(self,
                    train_loader: torch.utils.data.DataLoader,
                    val_loader: Union[torch.utils.data.DataLoader, None],
                    optimizer: torch.optim.Optimizer,
                    loss_fn: nn.Module = nn.CrossEntropyLoss(),
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
            train_loss = self.train_epoch(train_loader, loss_fn, optimizer)
            val_loss = None

            if val_loader is not None:
                _, val_loss = self.validate(val_loader, loss_fn)

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
                    optimizer: torch.optim.Optimizer) -> float:
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

        for bio_drone_input, prev_stress_input, demo_input, target in train_loader:

            bio_drone_input = bio_drone_input.to(self.device)
            prev_stress_input = prev_stress_input.to(self.device)
            demo_input = demo_input.to(self.device)
            target = target.type(torch.LongTensor).to(self.device)

            optimizer.zero_grad()

            # forward pass
            output = self(bio_drone_input, prev_stress_input, demo_input).permute(
                0, 2, 1)  # output shape: (batch_size, STRESS_LEVELS, seq_length)

            # calculate loss
            loss = loss_fn(output, target)

            # backprop
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader.dataset)
        return train_loss

    def encode_demographics(self, demo_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the initial hidden state of the LSTM. NOTE: this function is to be used for prediction only,
        as it does not add the batch_size dimension in the hidden and cell state.

        :param demo_input: collaborator demographic data, tensor of shape (batch_size, demo_input_dim)

        :return: initial hidden state and initial cell state of the LSTM (both tensors of shape (1, hidden_size))
        """
        h0 = self.demo_model(demo_input).unsqueeze(0).to(self.device)
        c0 = torch.zeros_like(h0).to(self.device)

        h0 = h0.repeat(self.num_layers, 1)
        c0 = c0.repeat(self.num_layers, 1)

        return h0, c0

    def predict(self,
                h0: torch.Tensor,
                c0: torch.Tensor,
                bio_drone_input: torch.Tensor,
                prev_stress_level: int) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Predict the collaborator stress level given hidden state, cell state, bio_drone_input, and prev_stress_input

        :param h0: hidden state of LSTM cell, tensor of shape (1, hidden_size) (sequence len is 1)
        :param c0: cell state of LSTM cell, tensor of shape (1, hidden_size)
        :param bio_drone_input: bio_drone_input, tensor of shape (1, input_dim)
        :param prev_stress_level: previous stress level, int

        :return: stress prediction, hidden state, cell state
        """

        bio_drone_input = bio_drone_input.to(self.device).unsqueeze(0)
        prev_stress_level = torch.tensor(prev_stress_level).to(self.device).unsqueeze(0).type(torch.float32)

        # concat_input shape: (1, input_dim + 1) (sequence len is 1)
        concat_input = torch.cat((bio_drone_input, prev_stress_level[None, ...]), dim=-1)

        # out shape: (1, hidden_size)
        out, (h0, c0) = self.lstm(concat_input, (h0, c0))

        # out shape: (1, STRESS_LEVELS)
        out = self.fc(out)

        # stress_prediction shape: (1)
        stress_prediction = torch.argmax(out, dim=-1).item()

        return stress_prediction, h0, c0

    def validate(self,
                 val_loader: torch.utils.data.DataLoader,
                 loss_fn: nn.Module = nn.CrossEntropyLoss()) -> Tuple[float, float]:
        """
        Validate the model

        :param val_loader: Validation data loader
        :param loss_fn: Loss function
        :param device: Device to run the model on

        :return: Validation accuracy and validation loss
        """
        self.eval()
        val_loss = 0
        val_acc = 0

        with torch.no_grad():
            for bio_drone_input, prev_stress_input, demo_input, target in val_loader:
                # NOTE: this should be a batch size of 1

                bio_drone_input = bio_drone_input.to(self.device)
                prev_stress_input = prev_stress_input.to(self.device)
                demo_input = demo_input.to(self.device)
                target = target.to(self.device).type(torch.LongTensor)

                stress_predictions = torch.zeros_like(target).to(self.device)

                # forward pass
                initial_stress = prev_stress_input[:, 0].item()

                # encode collaborator demographic data using a neural network for the initial hidden state of the LSTM
                h0 = self.demo_model(demo_input).unsqueeze(0).to(self.device)  # h0 shape: (1, batch_size, hidden_size)
                c0 = torch.zeros_like(h0).to(self.device)  # c0 shape: (1, batch_size, hidden_size)

                h0 = h0.repeat(self.num_layers, 1, 1)  # h0 shape: (num_layers, batch_size, hidden_size)
                c0 = c0.repeat(self.num_layers, 1, 1)  # c0 shape: (num_layers, batch_size, hidden_size)

                prev_stress_level = initial_stress
                for t in range(target.shape[1]):
                    bio_drone_t = bio_drone_input[:, t, :].unsqueeze(1)  # bio_drone_t shape: (batch_size, 1, input_dim)
                    prev_stress_level = torch.tensor(prev_stress_level).unsqueeze(0).to(self.device)

                    # concat_input shape: (batch_size, seq_length, input_dim + 1)
                    concat_input = torch.cat((bio_drone_t, prev_stress_level[None, None, ...]), dim=-1)
                    # NOTE: prev_stress_level[None, None, ...] is to add two dimensions to prev_stress_level
                    # so that it can be concatenated with bio_drone_t

                    # out shape: (batch_size, seq_length, hidden_size)
                    out, (h0, c0) = self.lstm(concat_input, (h0, c0))

                    out = self.fc(out)  # out shape: (batch_size, seq_length, STRESS_LEVELS)

                    prev_stress_level = torch.argmax(out, dim=-1).item()
                    stress_predictions[0, t] = prev_stress_level

                    # calculate loss
                    loss = loss_fn(out[:, 0, :], target[:, t])

                    val_loss += loss.item()

                # calculate accuracy per time step
                correct = (stress_predictions == target).sum().item()

                val_acc += correct / target.shape[1]
                val_loss /= target.shape[1]

        val_loss /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)

        return val_acc, val_loss
