import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers=1):
        super(LSTM, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.rnn = nn.LSTM(self.input_channels, self.hidden_channels, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        batch_size, timesteps, C = X.size()
        output, _ = self.rnn(X)
        output = self.fc(output)
        output = self.sigmoid(output)
        output = output.view(batch_size, timesteps, -1)

        return output
