from artificial_data import *
from loading_utils import *
from models import *

import torch 
import torch.nn as nn
import os
import sys
import numpy as np
import time
import argparse
from argparse import Namespace
from copy import copy



def train_and_visualize(markov_process, model):

    '''
    SequenceObject options
    '''
    sequence_train_mode = copy(default_sequence_mode)
    sequence_train_mode.window_size = 1
    sequence_train_mode.return_transitions=True
    sequence_train_mode.post_transform = postprocess_2

    sequence_test_mode = copy(default_sequence_mode)
    sequence_test_mode.window_size = 1
    sequence_test_mode.return_transitions=False
    sequence_test_mode.post_transform = postprocess_2
    sequence_test_mode.collate_func = collatefunc_2


    '''
    Dataset options
    '''
    train_dset_opts = copy(default_sequence_dataset_options)
    train_dset_opts.sequence_mode = sequence_train_mode

    test_dset_opts = copy(default_sequence_dataset_options)
    test_dset_opts.sequence_mode = sequence_test_mode
    test_dset_opts.sample_mode = 'sequential'
    test_dset_opts.collate_fn = collatefunc_2


    '''
    Create the datasets
    '''
    train_dataset = MarkovSequenceDataset(1000, markov_process, options=train_dset_opts)

    print("Training RL model on markov dataset")
    train_model_on_dataset(model, train_dataset, print_every=100, save_every = 5, n_epochs=30)

    states, values = markov_process.get_all_states_and_values(model.cfg.update_cfg.rl_gamma)

    states = torch.from_numpy(states).float()

    predicted_values = model.predict(states).detach().numpy()

    visualize_markov_sequence(states, values, predicted_values=predicted_values)























def experiment():
    markov_process = MarkovProcess(renderer=feature_renderer_2)

    model_config = copy(default_model_config)
    model_config.save_to = 'markov_model_2_epochs_100_gamma_95.pth'
    model_config.update_cfg.rl_gamma=0.95
    model_config.device='cpu'

    model = GenericModel(model_config)
    
    train_and_visualize(markov_process, model)


if __name__ == '__main__':
    experiment()
