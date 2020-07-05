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
        elif self.cnn_type == 'resnet50':
            self.cnn = models.resnet50(pretrained=True)
        elif self.cnn_type == 'resnet101':
            self.cnn = models.resnet101(pretrained=True)
        elif self.cnn_type == 'resnet152':
            self.cnn = models.resnet152(pretrained=True)
        else:
            raise ValueError("Unknown network_type option")

        modules=list(self.cnn.children())[:-1]
        self.cnn = nn.Sequential(*modules)

        if not self.if_cnn_trainabe:
            for p in self.cnn.parameters():
                p.requires_grad = False

        self.rnn = nn.LSTM(512, self.hidden_channels, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        batch_size, timesteps, C, H, W = input.size()
        input = input.view(batch_size*timesteps, C, H, W)
        output = self.cnn(input)
        output = output.view(batch_size, timesteps, -1)
        output, _ = self.rnn(output)
        output = self.fc(output)
        output = self.sigmoid(output)
        output = output.view(batch_size, timesteps, -1)

        return output
