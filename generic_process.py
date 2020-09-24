import torch 
import torch.nn as nn
import os
import sys
import numpy as np
import time
import argparse
from argparse import Namespace

from old_code.video_loader import Zipindexables



'''
Artificial sequence generators
'''
# For use in derived sequence_object classes
class MarkovProcess:
    def __init__(self):
        pass


class MnistMarkovProcess:
    def __init__(self):
        pass


# Have a sequence generator that loads embeddings too I guess, by using SequenceDataset to get embeddings for a second sequence dataset



'''
Rendering functions:
    For use in artificial sequence generators

'''

def render_sequence_1(states):
    pass

def render_mnist_sequence_1(states):
    pass


'''
Artificial sequence labeling functions (like even only, etc.)
For use in artificial sequence generators
'''
def label_sequence_1(states):
    pass



'''
Utilities: defined already in another file
'''

# imported above

#class ZipIndices:
#    pass

#class ZipIndexables:
#    pass

'''
Batch postprocessing functions:
    convert sequence object batches into correct training format, scale properly, etc.
'''
def postprocess_1(args):
    return [torch.from_numpy(args[0]).float()/255.0] + [torch.from_numpy(np.array(val)).float() for val in args[1:-1]]+[torch.from_numpy(np.array(args[-1]))]

def postprocess_test(args):
    return [torch.from_numpy(args[0]).float()] + [torch.from_numpy(np.array(val)).float() for val in args[1:-1]]+[torch.from_numpy(np.array(args[-1]))]


'''
Reward Schemas:
    funtions that take a series and a label for the whole series, and expand the label into
    rewards spanning the whole
'''
def labelfunc_1(series, label):
    pass


'''
Null schemas:
    functions that take an object or object shape and return a null object of that type
'''
def nullfunc_1(numpy_obj):
    return np.zeros(numpy_obj.shape, dtype=numpy_obj.dtype)


'''
Collating functions
'''
def collatefunc_1(args):
    data = args[0]
    if len(data[0].shape) == 4:
        data = torch.stack(data).transpose(0,1)
    else:
        data = torch.stack(data)
    return [data] + [torch.stack(a) for a in args[1:]]


'''
Sequence objects
'''

default_sequence_mode = Namespace(
        window_size = 10,
        return_transitions = True,
        pad_beginning = True,
        return_global_label = False,
        post_transform = postprocess_1,
        null_transform = nullfunc_1,
        batch_size = 64, # typically unused
        )

class SequenceObject:
    def __init__(self, *args, global_label=None, mode=default_sequence_mode):
        assert(len(args) > 0)
        assert(len(args[0]) > 0)
        for a in args:
            assert(len(a) == len(args[0]))

        self._sequence_length = len(args[0])
        self.sequences = args
        self.global_label = global_label,
        self.set_mode(mode)

    def set_mode(self, mode):
        self.mode = mode
        self.null_object = self.mode.null_transform(np.array(self.sequences[0][0])) # This is stored redundantly across objects; fix later

    def __len__(self):
        if self.mode.pad_beginning:
            return self._sequence_length
        else:
            return max(self._sequence_length - self.mode.window_size + 1, 0)

    def __getitem__(self, i):
        if self.mode.pad_beginning:
            bottom_calculated = i-self.mode.window_size+1
            range_bottom = max(bottom_calculated, 0)
            range_top = i+1
            num_null = abs(bottom_calculated) * (bottom_calculated < 0)
            rest = [self.sequences[k][i] for k in range(1,len(self.sequences))]
        else:
            bottom_calculated = i
            range_bottom = i
            range_top = i+self.mode.window_size
            num_null = 0
            rest = [self.sequences[k][range_top-1] for k in range(1,len(self.sequences))]

        null_list = num_null*[self.null_object]
        data_window = np.concatenate(self.sequences[0][range_bottom:range_top])
        data_window = np.concatenate(null_list+[data_window], axis=0)

        is_terminal = (i == self.__len__()-1)
        
        if self.mode.return_transitions:
            if is_terminal:
                future_data_window = np.concatenate(self.mode.window_size*[self.null_object],axis=0)
            else:
                future_range_bottom = max(bottom_calculated+1, 0)
                future_num_null = max(num_null-1,0)
                future_null_list = future_num_null*[self.null_object]
                future_data_window = np.concatenate(self.sequences[0][future_range_bottom:range_top+1])
                future_data_window = np.concatenate(future_null_list + [future_data_window], axis=0)
            data_window = np.stack([data_window, future_data_window])

        all_results = [data_window] + rest + [is_terminal]

        if self.mode.return_global_label:
            return self.mode.post_transform(all_results), self.global_label
        else:
            return self.mode.post_transform(all_results)


    def get_full_sequence(self):
        pass

    def __iter__(self):
        pass

    def __next__(self):
        pass

    # Used for parallel preloading and such
    def _load(self):
        pass

    def _flush(self):
        pass


# Derive various types of sequence objects
# Implement load and flush on them
# Sequences must be numpy arrays
# Sequence objects can return multiple series, and transitions
# Return a boolean array telling whether a point is terminal or not
# Can load batches via __next__ (for processing the individual sequence during testing)
# Can place null values at beginning of sequence

default_sequence_dataset_options = Namespace(
        sequence_mode = default_sequence_mode,
        collate_fn = collatefunc_1,
        batch_size = 64,
        sample_mode = 'random',
        preload_num = None,
        )

class SequenceDataset:
    def __init__(self, sequence_objects, options=default_sequence_dataset_options): # include curation function and randomization level, e.g. lstm or other
        self.sequences = sequence_objects
        self.options = options
        self.set_modes(self.options.sequence_mode)


    def set_modes(self, mode):
        for s in self.sequences:
            s.set_mode(mode)

        self.zip_sequences = Zipindexables(self.sequences)
        self.indices = np.arange(len(self.zip_sequences))
        # reset here or?

    def _reset(self):
        if self.options.sample_mode == 'random':
            self.indices = np.random.permutation(self.indices)
        self.position = 0
       

    def __iter__(self):
        self._reset()
        return self

    def _preload(self):
        pass

    def __next__(self):
        if self.position == len(self.zip_sequences):
            raise StopIteration
        end = min(len(self.zip_sequences), self.position+self.options.batch_size)
        batch = self.zip_sequences[self.indices[self.position:end]]
        self.position = end
        num_return_seqs = len(batch[0])
        return_seqs = [ [t[i] for t in batch] for i in range(num_return_seqs) ]
        return self.options.collate_fn(return_seqs)

    # Save dataset to file
    def save(self):
        pass

    # Load dataset from file
    def load(self):
        pass

    def num_sequences(self):
        return len(self.sequences)

    # Return whole sequence and the *global sequence labels* if any, e.g. the frame index of a crash, etc.
    def get_sequence(self, i):
        return self.sequences[i]




# Derive various types of sequence dataset here that load videos or something








'''
Generic models:
    Take neural nets, optimizer settings, loss settings, and track them
'''

class GenericModel(nn.Module):
    def __init__(self, network_constructor, network_args, optim_construct, optim_args, apply_func, update_func, device, savename):
        pass

    def update(self, *args):
        pass

    def save(self):
        pass

    def load(self):
        pass


# For later extension to LSTMs
class GenericRecurrentModel(GenericModel):
    pass


# For simple linear classifiers and such
class GenericSimpleModel:
    pass




'''
Time series functions
'''
def process_series():
    pass


def sweep_decision_boundary():
    pass

def auroc():
    pass


'''
Metric trackers
'''



'''
Training, testing, metric extraction, progress saving
'''



'''
Plotting, results viewing, results saving

'''



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
Run from command line
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_sequence_object', action='store_true')
    parser.add_argument('--test_sequence_dataset', action='store_true')
    opt = parser.parse_args()

    if opt.test_sequence_object:
        print("Testing sequence object")
        test_sequence_object()
    elif opt.test_sequence_dataset:
        print("Testing sequence dataset")
        test_sequence_dataset()








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





























