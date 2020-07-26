import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse

from cnn_dataloader import CNNLSTMDataLoader
from cnn_lstm import CNNLSTM

parser = argparse.ArgumentParser(description='text_img_matching')
parser.add_argument('--modelpath', type=str, required=False, default=None,
                    help='Path to saved model')
parser.add_argument('--totest', type=str, required=False, default=None,
                    help='Do testing or training')

global args
args = parser.parse_args()
load_network_path = args.modelpath
to_test = args.totest

network_type = 'resnet101'
hidden_channels = 512
num_layers = 3
if_cnn_trainabel = False
learning_rate = 0.01
use_q_loss = True
model_save_path = '../test_model.pth'
n_epochs = 1
rl_gamma = 0.999

image_shape = (224,224,3)
frame_interval = 3
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 2
data_num = 100
terminal_weight = 64

dpath = 'D:/Beamng_research/recordings/Beamng_dataset_30fps'
train_annotation = "train_annotation.txt"
test_annotation = "test_annotation.txt"
val_annotation = "val_annotation.txt"
train_dataset = CNNLSTMDataLoader(dpath, train_annotation,
                                 frame_interval=frame_interval,
                                 device=device,
                                 batch_size=batch_size,
                                 data_num=data_num,
                                 terminal_weight=terminal_weight)
train_loader = iter(train_dataset)
test_dataset = CNNLSTMDataLoader(dpath, test_annotation,
                                 frame_interval=frame_interval,
                                 device=device,
                                 batch_size=batch_size,
                                 data_num=data_num,
                                 terminal_weight=terminal_weight)
test_loader = iter(test_dataset)
val_dataset = CNNLSTMDataLoader(dpath, val_annotation,
                                 frame_interval=frame_interval,
                                 device=device,
                                 batch_size=batch_size,
                                 data_num=data_num,
                                 terminal_weight=terminal_weight)
val_loader = iter(val_dataset)

class Model:
    def __init__(self):
        self.network_type = network_type
        self.device = device
        self.rl_gamma = rl_gamma

        if load_network_path != None:
            self.network =  torch.load(load_network_path)
        else:
            self.network = CNNLSTM(network_type, hidden_channels,
                                   num_layers, if_cnn_trainabel)


        if torch.cuda.device_count() > 1:
            print('Use', torch.cuda.device_count(), 'gpus')
            self.network = nn.DataParallel(self.network)
        else:
            self.network = self.network.to(self.device)

        if use_q_loss:
            self.update_func = self.q_update
        else:
            self.update_func = self.prob_update
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

    def q_loss(self, q_current, q_future, weights):
        diff = q_future - q_current
        loss = (weights*(diff**2)).mean()
        return loss

    def q_update(self, x, y, weights, actual):
        q_current = self.network(x)
        q_temp = q_current.detach().clone()
        q_future = q_temp.detach().clone()
        q_future[:, :-1] = q_temp[:, 1:]
        q_future = self.rl_gamma * q_future + y
        loss = self.q_loss(q_current, q_future, weights)
        self.network.zero_grad()
        loss.backward()
        self.optimizer.step()
        return q_current, loss.item()

    def prob_loss(self, predictions, actual):
        return nn.MSELoss(predictions, actual)

    def prob_update(self, x, y, weights, actual):
        predictions = self.network(x)
        loss = self.prob_loss(predictions, actual)
        self.network.zero_grad()
        loss.backward()
        self.optimizer.step()
        return predictions, loss.item()

    def update(self, x, y, weights, actual):
        return self.update_func(x, y, weights, actual)

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()

    def save(self):
        torch.save(self.network, model_save_path)

def train(model):
    for epoch in range(n_epochs):
        model.train()
        print("Epoch:", epoch)
        for i, batch in enumerate(train_loader):
            x, y, weights, actual = batch
            q_current, loss = model.update(x, y, weights, actual)
            #print(q_current)
            if i % 10 == 0:
                print("  iteration:", i, ' loss: ', loss)

        # less loss to save
        model.save()
        # if epoch % 5 == 0:
        #     model.eval()
        #     # evaluate

def main():
    if to_test == None:
        print('loading model')
        model = Model()
        print('start training')
        train(model)
    else:
        exit('testing not implemented')
        # testing

if __name__ == '__main__':
    main()
