import torch 
import torch.nn as nn
import os
import sys
import numpy as np
import time
import argparse
from argparse import Namespace







'''
Unit testing
'''

def test_sequence_object():
#    seq = 2*np.arange(10)
#    seq2 = np.arange(10)
    seq = np.arange(40).reshape((10,2,2))
    seq2 = np.arange(10)
    mode = default_sequence_mode
    mode.window_size = 2
    mode.post_transform = postprocess_test
    mode.return_transitions = True
    mode.pad_beginning = False
#    mode.
    s = SequenceObject(seq, seq2, mode=mode)

    
    print(len(s))
    for i in range(len(s)):
        print(s[i])


def test_sequence_dataset():
    seqs = np.arange(600).reshape((5,10,3,2,2))
    seq2 = np.arange(50).reshape((5,10))

    mode = default_sequence_mode
    mode.window_size = 4
    mode.post_transform = postprocess_1
    mode.return_transitions = False
    mode.pad_beginning = False

    opts = default_sequence_dataset_options
    opts.sequence_mode = mode
    opts.batch_size = 7
    opts.sample_mode = 'sequential'

    sobjs = [ SequenceObject(s, s2, mode=mode) for s,s2 in zip(seqs,seq2) ]

    sdat = SequenceDataset(sobjs, options = opts)

    for batch in sdat:
        print(batch)
        print(batch[0].shape)




'''
Putting it all together
'''

def experiment_1():
    n_epochs = 30

    markov_process = MarkovProcess(transition_matrix=transition_matrix_1, 
                       state_mapping=state_mapping_1,
                       terminal_states = terminal_states_1,
                       terminal_values = terminal_values_1,
                       sequence_render = render_sequence_2,
                       sequence_labeler = label_sequence_1,
                       random_terminal=True,
                       )

    sequence_dataset_options = default_sequence_dataset_options
    # Adjust values here


    sequence_train_mode = Namespace(
            window_size = 3,
            return_transitions = True,
            pad_beginning = True,
            return_global_label = False,
            post_transform = postprocess_1,
            null_transform = nullfunc_1,
            collate_fn = collatefunc_1,
            batch_size = 64, # typically unused
            )

    sequence_test_mode = Namespace(
            window_size = 3,
            return_transitions = False,
            pad_beginning = True,
            return_global_label = False,
            post_transform = postprocess_1,
            null_transform = nullfunc_1,
            collate_fn = collatefunc_2,
            batch_size = 64, # typically unused
            )

    sequence_train_dataset_options = Namespace(
            sequence_mode = sequence_train_mode,
            collate_fn = collatefunc_1,
            batch_size = 64,
            sample_mode = 'random',
            preload_num = None,
            )

    sequence_test_dataset_options = Namespace(
            sequence_mode = sequence_test_mode,
            collate_fn = collatefunc_2,
            batch_size = 64,
            sample_mode = 'sequential',
            preload_num = None,
            )


    markov_train_dataset =  MarkovSequenceDataset(1000, markov_process, options=sequence_train_dataset_options)
    markov_test_dataset =  MarkovSequenceDataset(1000, markov_process, options=sequence_test_dataset_options)

    model_config = Namespace(
        save_to = 'sequence_model_render_2_random_terminal.pth',
        load_from = None,
        network_constructor=default_network_constructor, 
        network_args={'network_type':'sequence_net', 'input_features':9}, 
        optim_constructor=default_optim_constructor, 
        optim_args={'lr':0.0002}, 
        update_func=q_update, 
        update_cfg = Namespace(rl_gamma=0.977, terminal_weight=1),
        predict_func=predict_batch, 
        predict_func_cfg = Namespace(),
        embed_func=embed_batch,
        embed_cfg = Namespace(),
        device='cpu', 
        map_batch_to_device=default_map_batch_to_device,
        )

    model = GenericModel(model_config)

    print("Training RL model on markov dataset")
    train_model_on_dataset(model, markov_train_dataset, print_every=100, save_every = 5, n_epochs=n_epochs)
    
    embedding_train_input_sequences = MarkovSequenceDataset(50, markov_process, options=sequence_test_dataset_options)
    embedding_train_dataset = SequenceSampleDataset(embedding_train_input_sequences, embedding_model=model, sample_func=default_sample_func, 
                                    collate_fn=collatefunc_2, batch_size=64)
    non_embedding_train_dataset = SequenceSampleDataset(embedding_train_input_sequences, embedding_model=None, sample_func=default_sample_func, 
                                    collate_fn=collatefunc_2, batch_size=64)
    embedding_test_dataset = SequenceSampleDataset(markov_test_dataset, embedding_model=model, sample_func=default_sample_func, 
                                    collate_fn=collatefunc_2, batch_size=64)
    non_embedding_test_dataset = SequenceSampleDataset(markov_test_dataset, embedding_model=None, sample_func=default_sample_func, 
                                    collate_fn=collatefunc_2, batch_size=64)


    linear_model_on_embeddings_config = Namespace(
        save_to = 'linear_model_on_embeddings_render_2_random_terminal.pth',
        load_from = None,
        network_constructor=default_network_constructor, 
        network_args={'network_type':'linear_net', 'input_features':3}, 
        optim_constructor=default_optim_constructor, 
        optim_args={'lr':0.001}, 
        update_func=prob_update, 
        update_cfg = Namespace(rl_gamma=0.977, terminal_weight=1),
        predict_func=predict_batch, 
        predict_func_cfg = Namespace(),
        embed_func=embed_batch,
        embed_cfg = Namespace(),
        device='cpu', 
        map_batch_to_device=default_map_batch_to_device,
        )

    linear_model_off_embeddings_config = Namespace(
        save_to = 'linear_model_off_embeddings_render_2_random_terminal.pth',
        load_from = None,
        network_constructor=default_network_constructor, 
        network_args={'network_type':'linear_net','input_features':9}, 
        optim_constructor=default_optim_constructor, 
        optim_args={'lr':0.001}, 
        update_func=prob_update, 
        update_cfg = Namespace(rl_gamma=0.977, terminal_weight=1),
        predict_func=predict_batch, 
        predict_func_cfg = Namespace(),
        embed_func=embed_batch,
        embed_cfg = Namespace(),
        device='cpu', 
        map_batch_to_device=default_map_batch_to_device,
        )


    linear_model_on_embeddings = GenericModel(linear_model_on_embeddings_config)
    linear_model_off_embeddings = GenericModel(linear_model_off_embeddings_config)

    print("Training linear model on embeddings")
    train_model_on_dataset(linear_model_on_embeddings, embedding_train_dataset, print_every=10, save_every=5, n_epochs=n_epochs)
    print("Training linear model on raw points")
    train_model_on_dataset(linear_model_off_embeddings, non_embedding_train_dataset, print_every=10, save_every=5, n_epochs=n_epochs)

    embedding_accuracy = predict_classifier_model_on_dataset(linear_model_on_embeddings, embedding_test_dataset)
    no_embedding_accuracy = predict_classifier_model_on_dataset(linear_model_off_embeddings, non_embedding_test_dataset)


    print("Non-embedding accuracy", no_embedding_accuracy)
    print("Embedding accuracy", embedding_accuracy)


'''
Run from command line
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_sequence_object', action='store_true')
    parser.add_argument('--test_sequence_dataset', action='store_true')
    parser.add_argument('--experiment_1', action='store_true')
    opt = parser.parse_args()

    if opt.test_sequence_object:
        print("Testing sequence object")
        test_sequence_object()
    if opt.test_sequence_dataset:
        print("Testing sequence dataset")
        test_sequence_dataset()

    if opt.experiment_1:
        experiment_1()








# Next up:
#  construct generic_model class
#  fill in markov class and auxiliaries
#  construct train/test functions and metric savers
#  construct time series eval functions
#  construct embedding loader sequence object that can pick embeddings at random, and sequence dataset for that
#  Construct small basic networks
#  Construct several functions that build and test specific cases
#  Do the experiment with the 1d series, then build to mnist
#  make graphs and save them













# Results:

# Reward unrelated to secondary label: both linear accuracies 0.5















