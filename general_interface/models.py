import torch
import torch.nn as nn
import os
import sys
import numpy as np
import time
import argparse
from argparse import Namespace


def default_map_batch_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, tuple) or isinstance(batch, list):
        converted = []
        for b in batch:
            converted.append(default_map_batch_to_device(b, device))
        return type(batch)(converted)
    else:
        return batch


class SequenceNet(nn.Module):
    def __init__(self, input_features=3, intermediate_features=256, embedding_features=3):
        super().__init__()

        self.embedding_net = nn.Sequential(
                nn.Linear(input_features, intermediate_features),
                nn.ReLU(),
                nn.Linear(intermediate_features, embedding_features)
                )
        self.prediction_net = nn.Linear(embedding_features, 1)

    def embed(self, x):
        return self.embedding_net(x)

    def forward(self, x):
        x = x.view(len(x),-1)
        x = self.embed(x)
        return self.prediction_net(x)


class LinearNet(nn.Module):
    def __init__(self, input_features=3):
        super().__init__()

        self.net = nn.Linear(input_features,1)

    def embed(self, x):
        pass #Not needed

    def forward(self, x):
        x = x.view(len(x),-1)
        return self.net(x)

class LstmNet(nn.Module):
    def __init__(self, input_features=3, intermediate_features=256, embedding_features=3):
        super().__init__()

        self.lstm = nn.LSTM(input_features, intermediate_features, 3, batch_first=True)
        self.embedding_net = nn.Linear(intermediate_features, embedding_features)
        self.prediction_net = nn.Linear(embedding_features, 1)

    def embed(self, x):
        batch_s, series_l = x.shape[:2]
        x = x.view(batch_s, series_l, -1)
        x, _ = self.lstm(x)
        x = x[-1,:]
        return self.embedding_net(x).squeeze()

    def forward(self, x):
        batch_s, series_l = x.shape[:2]
        x = x.view(batch_s, series_l, -1)
        x, _ = self.lstm(x)
        x = x[-1,:]
        x = self.embedding_net(x)
        return self.prediction_net(x).squeeze()

def default_network_constructor(network_type='sequence_net', input_features=3, intermediate_features=256, embedding_features=3):
    if network_type == 'sequence_net':
        return SequenceNet(input_features=input_features, intermediate_features=intermediate_features, embedding_features=embedding_features)
    elif network_type == 'linear_net':
        return LinearNet(input_features=input_features)
    elif network_type == 'lstm_net':
        return LstmNet(input_features=input_features, intermediate_features=intermediate_features, embedding_features=embedding_features)
    else:
        print("Unrecognized network type")
        sys.exit(1)


def default_optim_constructor(network, **kwargs):
    return torch.optim.Adam(network.parameters(), **kwargs)




def q_loss(q_current, q_future, weights):
    diff = q_future-q_current
    loss = (weights*(diff**2)).mean()
    return loss

def prob_loss(predictions, actual):
    return nn.functional.binary_cross_entropy_with_logits(predictions, actual)

def q_update(network, optim, batch, cfg):
    x, r, terminal = batch[0], batch[1], batch[-1] #convention: the last thing in the batch is the terminal labels
    x_current = x[0]
    x_future = x[1]

    q_current = network(x_current).squeeze(1)

    weights = torch.ones(len(x_current), device=x_current.device)
    with torch.no_grad():
        weights[terminal] = cfg.terminal_weight
        future_preds = network(x_future).squeeze(1)
        future_preds[terminal] = 0
        q_future = cfg.rl_gamma*future_preds+r

    loss = q_loss(q_current, q_future, weights)

    network.zero_grad()
    loss.backward()
    optim.step()

    return loss.item()

def prob_update(network, optim, batch, cfg):
    x, y = batch[0], batch[1]
    predictions = network(x).squeeze(1)
    loss = prob_loss(predictions, y)
    network.zero_grad()
    loss.backward()
    optim.step()

    return loss.item()

def predict_batch(network, x, cfg):
#    x, y = batch[0], batch[1]
    predictions = network(x).squeeze(1)
    return predictions

def embed_batch(network, x, cfg):
    embeddings = network.embed(x)
    return embeddings

'''
Generic models:
    Take neural nets, optimizer settings, loss settings, and track them
'''

default_model_config = Namespace(
        save_to = 'model.pth',
        load_from = None,
        network_constructor=default_network_constructor,
        network_args={'network_type':'sequence_net'},
        optim_constructor=default_optim_constructor,
        optim_args={'lr':0.0002},
        update_func=q_update,
        update_cfg = Namespace(rl_gamma=0.997, terminal_weight=1),
        predict_func=predict_batch,
        predict_func_cfg = Namespace(),
        embed_func=embed_batch,
        embed_cfg = Namespace(),
        device='cuda:0',
        map_batch_to_device=default_map_batch_to_device,
        )


class GenericModel:
    def __init__(self, cfg):
        self.cfg = cfg # contains savename and everything else
        self._build()
        if self.cfg.load_from is not None:
            self._load()

    def _build(self):
        self.network = self.cfg.network_constructor(**self.cfg.network_args)
        self.device = self.cfg.device
        self.network = self.network.to(self.device)
        self.optim = self.cfg.optim_constructor(self.network, **self.cfg.optim_args) # take network and other args
        self.update_func = self.cfg.update_func # take network, optimizer, and data batch, return losses
        self.predict_func = self.cfg.predict_func # take network and data batch, return predictions
        self.map_batch_to_device = self.cfg.map_batch_to_device
        self.embed_func = self.cfg.embed_func

    def _load(self):
        with open(self.cfg.load_from, 'rb') as f:
            print("Loading model from", self.cfg.load_from)
            params = torch.load(f, map_location=self.cfg.device)
            self.network.load_state_dict(params['network_state_dict'])
            self.optimizer.load_state_dict(params['optimizer_state_dict'])

    def save(self):
        with open(self.cfg.save_to, 'wb') as f:
            torch.save({'network_state_dict':self.network.state_dict(), 'optimizer_state_dict':self.optim.state_dict()}, f)

    def update(self, batch):
        batch = self.map_batch_to_device(batch, self.device)
        return self.update_func(self.network, self.optim, batch, self.cfg.update_cfg)

    def embed(self, batch):
        batch = self.map_batch_to_device(batch, self.device)
        return self.embed_func(self.network, batch, self.cfg.embed_cfg)

    def predict(self, batch):
        batch = self.map_batch_to_device(batch, self.device)
        return self.predict_func(self.network, batch, self.cfg.predict_func_cfg)




# For later extension to LSTMs
class GenericRecurrentModel(GenericModel):
    pass


'''
Training, testing, metric extraction, progress saving
'''



'''
Plotting, results viewing, results saving

'''
def update_model_on_dataset(model, dataset, print_every=100):
    losses = []
    for i, batch in enumerate(dataset):
        loss = model.update(batch)
        losses.append(loss)
        if (i+1)%print_every == 0:
            print("Iteration {0}, average_loss {1}".format(i+1, np.mean(losses)))

    return losses

def train_model_on_dataset(model, dataset, print_every=100, save_every = 5, n_epochs=100):
    for epoch in range(n_epochs):
        print("Epoch", epoch)
        update_model_on_dataset(model,dataset, print_every=print_every)

        if (epoch+1)%save_every == 0:
            print("Saving")
            model.save()

    print("Saving final")
    model.save()


def predict_classifier_model_on_dataset(model, dataset):
    total = 0
    num = 0
    for batch in dataset:
        predictions = model.predict(batch[0]).to('cpu').detach().numpy()
        y = batch[1].to('cpu').detach().numpy()
        accuracy = ((predictions > 0) == (y > 0.5)).mean() #.float().mean()
        total += accuracy*len(y)
        num += len(y)

    total_accuracy = total/num
    return total_accuracy



def predict_sequence_model_on_dataset(model, dataset, sequence_score_func):
    all_predictions = []
    all_scores = []
    for i in range(dataset.num_sequences()):
        seq = dataset.get_sequence(i)
        seq_predictions = []
        for batch in seq:
            predictions = model.predict(batch[0]).to('cpu').detach().numpy()
        seq_predictions = np.concatenate(seq_predictions)
        all_predictions.append(seq_predictions)
        all_scores.append(sequence_score_func(seq_predictions, seq.global_label))

    return all_predictions, all_scores
