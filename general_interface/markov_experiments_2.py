# Import data models, network models, loading objects
from artificial_data import *
from loading_utils import *
from models import *
from helper_functions import *
from metrics_and_results import *



# Import Standard libraries
import torch
import torch.nn as nn
import os
import sys
import numpy as np
import time
import argparse
from argparse import Namespace
from copy import copy




########


def experiment1():
    '''
    Construct markov train and test dataset and loaders
    '''
    # Construct the markov process
    
    # feature_renderer_2 makes the features correlate perfectly with the reward
    # The reward is not random, but based precisely on the last markov state
    markov_process = MarkovProcess(renderer=feature_renderer_2)

    dataset_options = Namespace(
                                collate_fn = collatefunc_2,
                                sequence_mode = Namespace(
                                        collate_fn = collatefunc_2,
                                        post_transform = postprocess_2,
                                        window_size = 1
                                    )
                            )

    train_dataset = MarkovSequenceDataset(
                        1000, 
                        markov_process, 
                        options=dataset_options
                        )
    test_dataset = MarkovSequenceDataset(
                        1000, 
                        markov_process, 
                        options=dataset_options
                        )
    test_dataset.set_modes(Namespace(return_transitions=False))


    '''
    Construct models 
    '''

    # Baseline needs a different label function!

    # Basically need to switch the renderer label_func in markov_process
    # Or make use of the global label in the regressor so as not to need to do that?
    baseline = GenericModel(
                cfg = Namespace(
                    save_to = 'baseline_regressor.pth',
                    network_args={'network_type':'sequence_net'},
                    update_func = prob_update_markov_regressor,
                    device='cpu'
                 )
                ) # Need to add config options to each
    model = GenericModel(
             cfg = Namespace(
                save_to = 'q_model_regressor.pth',
                network_args={'network_type':'sequence_net'},
                update_func = q_update,
                device='cpu'
             )
            )


    '''
    Train model on dataset
    '''
    train_dataset.set_modes(Namespace(return_transitions=False))
    train_model_on_dataset(baseline, train_dataset, n_epochs=1)

    train_dataset.set_modes(Namespace(return_transitions=True))
    train_model_on_dataset(model, train_dataset, n_epochs=1)


    '''
    Test model on all possible states and plot
    '''
    # set model on eval mode before this?
    states, values = markov_process.get_all_states_and_values(model.cfg.update_cfg.rl_gamma)
    states = torch.from_numpy(states).float()

    predicted_baseline_values = baseline.predict(states).detach().cpu().numpy()
    predicted_model_values = model.predict(states).detach().cpu().numpy()

    visualize_markov_sequence(states, values, predicted_baseline_values)
    visualize_markov_sequence(states, values, predicted_model_values)


    '''
    Run on test set and extract time series plots and accuracy metrics
    '''
    # Need a way of running a model on a dataset and getting each separate time series    
    # Then use functions from metrics_and_results
    
    all_baseline_predictions, all_baseline_scores = predict_sequence_model_on_dataset(baseline, test_dataset)
    all_model_predictions, all_model_scores = predict_sequence_model_on_dataset(model, test_dataset)


    '''
    Construct embedding train and test datasets
    '''
    
    #embedded_baseline_train_dataset = SequenceSampleDataset(train_dataset, embedding_model=baseline)
    #embedded_baseline_test_dataset = SequenceSampleDataset(train_dataset, embedding_model=baseline)
    #embedded_train_dataset = SequenceSampleDataset(train_dataset, embedding_model=model)
    #embedded_test_dataset = SequenceSampleDataset(train_dataset, embedding_model=model)



    '''
    Train classifiers on embeddings
    '''
    #embedding_linear_model = GenericModel() # give options


    '''
    Test on embedding test dataset and get accuracies
    '''
    # Run embedding model on whole embedding dataset using same method as before

    # Have the functions predict_sequence_model_on_dataset and predict_classifier_model_on_dataset






if __name__ == '__main__':
    experiment1()
