#import torch 
#import torch.nn as nn
#import os
#import sys
import numpy as np
#import time
import argparse
#from argparse import Namespace

from loading_utils import *

from copy import copy

'''
Markov Process 1
'''
tp = 0.05
#     0       1     2      3      4      5      6      7      8      9
transition_matrix_1 = np.array([
  [ 0.334, 0.333, 0.000, 0.000, 0.000, 0.000, 0.333, 0.000, 0.000, 0.000 ], # 0
  [ 0.500-tp/2, 0.000, 0.500-tp/2, 0.000, 0.000, 0.000, 0.000, 0.000, tp,      0.000 ], # 1
  [ 0.000, 0.500-tp/2, 0.000, 0.500-tp/2, 0.000, 0.000, 0.000, 0.000, tp,      0.000 ], # 2
  [ 0.000, 0.000, 0.300-tp/2, 0.000, 0.700-tp/2, 0.000, 0.000, 0.000, tp,      0.000 ], # 3
  [ 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000,             0.000 ], # 4
  [ 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,             1.000 ], # 5 (to terminal 1)
  [ 0.500-tp/2, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.500-tp/2, tp,      0.000 ], # 6
  [ 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000-tp, 0.000, tp,             0.000 ],  # 7
  [ 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000,             0.000 ],  # Terminal zero
  [ 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,             1.000 ]  # Terminal one
])
state_mapping_1 = np.array([ 0, 1, 2, 3, 4, 5, -1, -2, 8, 9]) 
terminal_states_1 = np.arange(len(transition_matrix_1),dtype=int)[np.abs(transition_matrix_1.diagonal()-1) < 1e-8]
rewards_1 = np.array([0.0, 0, 0, 0, 0, 0, 0, 0, 0, 1])


'''
Rendering functions:
    For use in artificial sequence generators

'''

'''
Takes a mapped state sequence.

Returns a sequence with constant added features that don't correlate to the terminal label.
sequence_values : An array of the rendered values
feature_labels : An array the length of the returned sequence filled with the constant feature label.
feature_label : The constant feature label
'''
def render_sequence_1(states, terminal_label):
    sequence_values = []

    choice_matrix = [[-1,-1],[-1,1],[1,-1],[1,1]] #Can make this correlate with terminal_label instead
    choice = np.random.choice(4)
    feature_label = 1.0 if choice in [0,3] else 0.0 #Not linearly separable
    features = choice_matrix[choice]

    for s in states:
        sequence_values.append([s]+features)

    feature_labels = np.empty(len(states))
    feature_labels.fill(feature_label)

    return np.array(sequence_values), feature_labels, feature_label


'''
Like render_sequence_1, except feature_label is highly correlated with terminal label
'''
def render_sequence_2(states, terminal_label):
    sequence_values = []

    choice_matrix = [[-1,-1],[-1,1],[1,-1],[1,1]]
    if terminal_label == 1:
        choice = np.random.choice(4, p=[0.4, 0.1, 0.1, 0.4])
    else:
        choice = np.random.choice(4, p=[0.1, 0.4, 0.4, 0.1])
    feature_label = 1.0 if choice in [0,3] else 0.0 #Not linearly separable
    features = choice_matrix[choice]

    for s in states:
        sequence_values.append([s]+features)

    feature_labels = np.empty(len(states))
    feature_labels.fill(feature_label)

    return np.array(sequence_values), feature_labels, feature_label



'''
Take a state sequence and a terminal label and return a sequence of rewards at each state.
'''
def label_sequence_1(states, terminal_label):
    rewards = np.zeros(len(states))
    rewards[-1] = terminal_label
    return rewards


'''
MarkovProcess

This defines a markov state-transition process.

transition_matrix : A 2D numpy array whose values are transition probabilities between states represented by rows/cols
state_mapping : A numpy array that maps abstract markov states (0-N) to specific real-number (or possible vector) values.
terminal_states : A numpy array listing the terminal states
rewards : A numpy array giving the rewards for each state
sequence_render : A function which takes a list of state values and a label and returns a time series based on the state sequence (in the simplest case, can trivially return the state sequence itself).
sequence_labeler : A function that takes a list of values and a label and returns the rewards at each state (usually mostly zero).
random_terminal : A boolean indicating whether the given sequence should be randomly assigned a terminal label.
'''
class MarkovProcess:
    def __init__(self, transition_matrix=transition_matrix_1, 
                       state_mapping=state_mapping_1,
                       terminal_states = terminal_states_1,
                       rewards = rewards_1,
                       sequence_render = render_sequence_1,
                       sequence_labeler = label_sequence_1,
                       random_terminal=False,
                       ):

        self.transition_matrix = transition_matrix
        self.state_mapping = state_mapping_1
        self.terminal_states = terminal_states
        self.rewards = rewards
        self.sequence_renderer = sequence_render
        self.sequence_labeler = sequence_labeler
        self.random_terminal = random_terminal

        self.n_states = len(self.transition_matrix)
        self.state_indices = np.arange(self.n_states, dtype=int)


    '''
    Returns:
      sequence : A numpy array of the rendered sequence
      rewards : An array giving the rewards at each state in the sequence
      feature_labels : An array giving the labels of the hidden features, if this is applicable
      global_labels : Tuple of the different labels for the entire sequence; (terminal label, feature label)
      states : The abstract markov states corresponding to the sequence (may have different length than the sequence, as the sequence may be much expanded).
    '''
    def sample(self):
        states, terminal_label = self.sample_states()
        rewards = self.sequence_labeler(states, terminal_label)

        sequence, feature_labels, global_feature_label = self.sequence_renderer(states, terminal_label)
        global_labels = (terminal_label, global_feature_label)
        return sequence, rewards, feature_labels, global_labels, states


    '''
    Sample a sequence of abstract states from the markov process, decide on a label, then return the mapped sequence values.
    '''
    def sample_states(self):
        state = 0
        state_sequence = [state]
        
        while state not in self.terminal_states:
            state = np.random.choice(self.n_states, p=self.transition_matrix[state])
            state_sequence.append(state)

        if self.random_terminal:
            terminal_label = np.random.choice([0.0,1.0])
        else:
            terminal_label = self.rewards[state_sequence[-1]]

        mapped_state_sequence = np.array([self.state_mapping[s] for s in state_sequence]) #Map the states

        return mapped_state_sequence, terminal_label





'''
A derived class of SequenceDataset that constructs a dataset by sampling n_sequences from the given markov_process, creating SequenceObjects out of each sequence (initialized using options.sequence_mode), then initializing the underlying SequenceDataset using these objects and the given options.
'''
class MarkovSequenceDataset(SequenceDataset):
    def __init__(self, n_sequences, markov_process, options=default_sequence_dataset_options):
        sequence_objects = []
        for i in range(n_sequences):
            sequence, rewards, feature_labels, global_labels, states = markov_process.sample()
            seq_obj = SequenceObject(sequence, rewards, feature_labels, states, global_label=global_labels, mode=options.sequence_mode)
            sequence_objects.append(seq_obj)

        super().__init__(sequence_objects, options=options)







def test1():
    mp = MarkovProcess()

    for i in range(5):
        sequence, rewards, feature_labels, global_labels, states = mp.sample()
        print("Sequence\n", sequence)
        print("Rewards\n", rewards)
        print("Feature labels\n", feature_labels)
        print("Global labels\n", global_labels)
        print("States\n", states)
        print("----------------")


def test2():
    mp = MarkovProcess()
    
    n_seqs = 10


    seq_mode = copy(default_sequence_mode)
    seq_mode.window_size = 1
    seq_mode.post_transform = lambda x : x
    dset_mode = copy(default_sequence_dataset_options)
    dset_mode.sequence_mode = seq_mode


    msd = MarkovSequenceDataset(n_seqs, mp, options=dset_mode)

    for i in range(msd.num_sequences()):
        cur_seq = msd.get_sequence(i)
        print(cur_seq)



if __name__ == '__main__':
#    test1()
    test2()












