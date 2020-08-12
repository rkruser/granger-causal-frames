import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

from cnn_dataloader import CNNLSTMDataLoader
from cnn_lstm import CNNLSTM

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True

torch.set_printoptions(sci_mode=False)

parser = argparse.ArgumentParser(description='car_crash_prediction')
parser.add_argument('--modelpath', type=str, required=False, default=None,
                    help='Path to saved model')
parser.add_argument('--totest', type=str, required=False, default=None,
                    help='Do testing or training')
parser.add_argument('--batchsize', type=int, required=False, default=1)
parser.add_argument('--qloss', type=str, required=False, default=None)
parser.add_argument('--nepochs', type=int, required=False, default=10)
parser.add_argument('--datanum', type=int, required=False, default=-1)
parser.add_argument('--modelname', type=str, required=False, default='test')
parser.add_argument('--learningrate', type=float, required=False, default=0.01)
parser.add_argument('--networktype', type=str, required=False, default='resnet101')

global args
args = parser.parse_args()
load_network_path = args.modelpath
to_test = args.totest
use_q_loss = args.qloss
batch_size = args.batchsize
n_epochs = args.nepochs
data_num = args.datanum
model_name = args.modelname
learning_rate = args.learningrate
network_type = args.networktype

hidden_channels = 512
num_layers = 3
if_cnn_trainabel = False
model_save_path = '/vulcanscratch/ywen/car_crash/models/'
rl_gamma = 0.5

image_shape = (224,224,3)
frame_interval = 3
terminal_weight = 1
if_shuffle = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dpath = '/vulcanscratch/ywen/car_crash/BeamNG_dataset'
train_annotation = "train_annotation.txt"
test_annotation = "test_annotation.txt"
val_annotation = "val_annotation.txt"
train_dataset = CNNLSTMDataLoader(dpath, train_annotation,
                                 frame_interval=frame_interval,
                                 device=device,
                                 batch_size=batch_size,
                                 if_shuffle=if_shuffle,
                                 data_num=data_num,
                                 terminal_weight=terminal_weight)
# train_loader = iter(train_dataset)
test_dataset = CNNLSTMDataLoader(dpath, test_annotation,
                                 frame_interval=frame_interval,
                                 device=device,
                                 batch_size=batch_size,
                                 if_shuffle=if_shuffle,
                                 data_num=data_num,
                                 terminal_weight=terminal_weight)
test_loader = iter(test_dataset)
val_dataset = CNNLSTMDataLoader(dpath, val_annotation,
                                 frame_interval=frame_interval,
                                 device=device,
                                 batch_size=batch_size,
                                 if_shuffle=if_shuffle,
                                 data_num=data_num,
                                 terminal_weight=terminal_weight)
val_loader = iter(val_dataset)

class Model:
    def __init__(self):
        self.network_type = network_type
        self.device = device
        self.rl_gamma = rl_gamma
        self.prob_loss_func = nn.MSELoss()
        self.network = CNNLSTM(network_type, hidden_channels,
                               num_layers, if_cnn_trainabel).cuda()

        if load_network_path != None:
            print('use pretrained model')
            self.network.load_state_dict(torch.load(load_network_path))

        if torch.cuda.device_count() > 1:
            print('use', torch.cuda.device_count(), 'gpus')
            self.network = nn.DataParallel(self.network, device_ids=range(torch.cuda.device_count()))

        if use_q_loss:
            print('use q learning!')
            self.update_func = self.q_update
        else:
            print('use prob update!')
            self.update_func = self.prob_update
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

    def q_loss(self, q_current, q_future, weights):
        diff = q_future - q_current
        loss = (weights * (diff ** 2)).mean()
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
        return self.prob_loss_func(predictions, actual)

    def prob_update(self, x, y, weights, actual):
        predictions = self.network(x)
        loss = self.prob_loss(predictions, actual)
        self.network.zero_grad()
        loss.backward()
        self.optimizer.step()
        return predictions, loss.item()

    def no_grad_forward(self, x):
        with torch.no_grad():
            predictions = self.network(x)
        return predictions

    def update(self, x, y, weights, actual):
        return self.update_func(x, y, weights, actual)

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()

    def save(self, save_path):
        if torch.cuda.device_count() > 1:
            torch.save(self.network.module.state_dict(), save_path)
        else:
            torch.save(self.network.state_dict(), save_path)


def train(model):
    min_loss = float('inf')
    for epoch in range(n_epochs):
        # reshuffle the dataset after each iteration
        train_dataset.shuffle_samples()
        train_loader = iter(train_dataset)
        print_ben = int(len(train_loader)/15)

        print('Epoch:', epoch + 1)
        model.train()
        pbar = tqdm.tqdm(total=len(train_loader))

        for i, batch in enumerate(train_loader):
            x, y, weights, actual = batch
            q_current, loss = model.update(x, y, weights, actual)

            if i % print_ben == 0:
                print('  iteration:' + str(i + 1) + ' loss:' + str(loss))
                print((actual + 0.4)[:2])
                print(q_current.detach().clone()[:2])

            del q_current, loss
            # torch.cuda.empty_cache()

            pbar.update(1)

        print('validating')
        curr_loss = test(model, val_loader)

        if curr_loss < min_loss:
            print('lower loss! current loss:', curr_loss)
            min_loss = curr_loss
            model.save(model_save_path + model_name + '_best_model.pth')
        else:
            print('higher loss! current loss:', curr_loss)

        del curr_loss
        torch.cuda.empty_cache()

    model.save(model_save_path + model_name + '_final_model.pth')

def test(model, data_loader):
    model.eval()
    val_loss = []
    pbar = tqdm.tqdm(total=len(data_loader))

    for i, batch in enumerate(data_loader):
        x, _, _, actual = batch
        predictions = model.no_grad_forward(x)
        curr_loss = model.prob_loss(predictions, actual)
        val_loss.append(curr_loss.item())

        if i % 2 == 0:
            print((actual + 0.4)[:2])
            print(predictions.detach().clone()[:2])

        pbar.update(1)

    return np.mean(val_loss)

def main():
    if to_test == None:
        print('loading model')
        model = Model()
        print('start training')
        train(model)
    else:
        print('testing')
        if load_network_path == None:
            print('please provide model path!')
            exit()
        model = Model()
        curr_loss = test(model, val_loader)
        print('loss:', curr_loss)

if __name__ == '__main__':
    main()
