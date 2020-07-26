import torch
import torch.nn as nn
import torchvision.models as models

class CNNLSTM(nn.Module):
    def __init__(self, cnn_type, hidden_channels, num_layers=1, if_cnn_trainabel=True):
        super(CNNLSTM, self).__init__()
        self.cnn_type = cnn_type
        self.if_cnn_trainabe = if_cnn_trainabel
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        if self.cnn_type == 'resnet18':
            self.cnn = models.resnet18(pretrained=True)
            self.lstm_input_size = 512
        elif self.cnn_type == 'resnet50':
            self.cnn = models.resnet50(pretrained=True)
            self.lstm_input_size = 2048
        elif self.cnn_type == 'resnet101':
            self.cnn = models.resnet101(pretrained=True)
            self.lstm_input_size = 2048
        elif self.cnn_type == 'resnet152':
            self.cnn = models.resnet152(pretrained=True)
            self.lstm_input_size = 2048
        else:
            raise ValueError("Unknown network_type option")

        modules = list(self.cnn.children())[:-1]
        self.cnn = nn.Sequential(*modules)

        if not self.if_cnn_trainabe:
            for p in self.cnn.parameters():
                p.requires_grad = False

        self.rnn = nn.LSTM(self.lstm_input_size, self.hidden_channels, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        batch_size, timesteps, C, H, W = X.size()
        seq_mask = torch.sum(X, 4)
        seq_mask = torch.sum(seq_mask, 3)
        seq_mask = torch.sum(seq_mask, 2)
        seq_mask = seq_mask == 0
        X = X.view(batch_size*timesteps, C, H, W)
        output = self.cnn(X)
        output = output.view(batch_size, timesteps, -1)
        output[seq_mask] = 0
        output, _ = self.rnn(output)
        output = self.fc(output)
        output = self.sigmoid(output)
        output = output.view(batch_size, timesteps)
        # output[seq_mask] = -1

        return output
