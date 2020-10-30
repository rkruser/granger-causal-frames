import torch 
#import torch.nn as nn
#import os
#import sys
import numpy as np
#import time
import argparse
from argparse import Namespace
import copy
import decord

from helper_functions import *

class Zipindices:
    # length_list : a list of integers specifying lengths of other objects
    def __init__(self, length_list):
        self.llist = np.array(length_list)
        self.clist = np.cumsum(length_list)-1
        
    def __len__(self):
        if len(self.clist)>0:
            return self.clist[-1]+1
        else:
            return 0
        
    def __getitem__(self, key):
        if key < 0:
            key = self.__len__()+key
        if key < 0 or key >= self.__len__():
            raise KeyError('Zipindices key out of bounds')
        place = np.searchsorted(self.clist, key)
        if place==0:
            return place, key
        else:
            return place, key-self.clist[place-1]-1

class Zipindexables:
    # indexables is an iterable of objects with len and getitem defined
    def __init__(self, indexables, slice_is_range=True):
        self.indexables = indexables
        self.slice_is_range = slice_is_range
        self._buildindex()
        
    def _buildindex(self):
        llist = []
        for l in self.indexables:
            llist.append(len(l))
        self.zipindex = Zipindices(llist)       
        
    def __len__(self):
        return len(self.zipindex)
    
    def __getitem__(self, key):
        if isinstance(key,int) or isinstance(key,np.int64):
            return self._access(key)
        elif isinstance(key, slice):
            return self._slice(key)
        elif isinstance(key, np.ndarray):
            if len(key.shape)==1:
                if key.dtype=='bool':
                    key = np.arange(len(key))[key]
                    return self._int_array_index(key)
                elif 'int' in str(key.dtype):
                    return self._int_array_index(key)
                else:
                    raise KeyError('Zipindexables key array has invalid type')
            else:
                raise(KeyError('Zipindexables key array has wrong number of dimensions'))
        else:
            try:
                return self._access(key)
            except KeyError as e:
                raise KeyError('Zipindexables key error') from e

    def __iter__(self):
        self.iter_index=0
        return self

    def __next__(self):
        if self.iter_index < self.__len__():
            item = self.__getitem__(self.iter_index)
            self.iter_index += 1
            return item
        else:
            raise StopIteration
            
    def get_indexable(self, key):
        return self.indexables[key]
    
    def num_indexables(self):
        return len(self.indexables)
    
    def __str__(self):
        return self.__repr__()+', contents = '+str(self.indexables)
        
    def __repr__(self):
        return 'Zipindexables object, indexables {0}, items {1}'.format(
            self.num_indexables(), self.__len__())
            
    def _access(self, key):
        place, ind = self.zipindex[key]
        return self.indexables[place][ind]
        
    # Not maximally efficient, but whatever
    def _slice(self, key):
        start = key.start
        stop = key.stop
        step = key.step
        if start is None:
            start = 0
        if step is None:
            step = 1
        if stop is None:
            stop = self.__len__()
            
        if self.slice_is_range:
            return self._int_array_index(range(start,stop,step)) #changed from np.arange to range
        
        if step < 0:
            print("Warning: negative step size produces undefined behavior when slicing a Zipindexables object")
        
        place_list = []
        place_inds = {}
        for i in range(start, stop, step):
            place, ind = self.zipindex[i]
            if place not in place_inds:
                place_inds[place] = [ind, ind]
                place_list.append(place)
            else:
                place_inds[place][1] = ind
            
        new_items = []
        for j in place_list:
            sl = place_inds[j]
            new_items.append(self.indexables[j][sl[0]:sl[1]+step:step])
                             
        return Zipindexables(new_items)
            
    def _int_array_index(self, key):
        all_items = []
        for i in key:
            all_items.append(self._access(i))
        return all_items
        

'''
Sequence objects
'''




'''
Make a copy of ns1 and update it with ns2, or just return the copy if ns2 is None
'''
def merge_namespaces(ns1, ns2, make_copy=True):
    new_ns = copy.copy(ns1) if make_copy else ns1
    if ns2 is not None:
        new_ns.__dict__.update(ns2.__dict__)
    return new_ns

class SequenceObject:
    default = Namespace(
            window_size = 10,
            return_transitions = True,
            pad_beginning = True,
            return_global_label = False,
            post_transform = postprocess_1,
            collate_fn = collatefunc_1,
            batch_size = 64, # typically unused
            )

    def __init__(self, *args, global_label=None, null_object=None, mode=None):
        assert(len(args) > 0)
        assert(len(args[0]) > 0)
        for a in args:
            assert(len(a) == len(args[0]))

        self._sequence_length = len(args[0])
        self.sequences = args
        self.global_label = global_label
        self.null_object = null_object
        self.set_mode(mode)

    def set_mode(self, mode):
        base_dict = self.__dict__.get('mode')
        make_copy = False
        if base_dict is None:
            base_dict = self.__class__.default
            make_copy = True
        self.mode = merge_namespaces(base_dict, mode, make_copy=make_copy)

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

    
    def __str__(self):
        s = '[\n'
        for i in range(self.__len__()):
            item = self.__getitem__(i)
            s += str(item) + ',\n'
        s += ']\n'
        return s


    def __repr__(self):
        return "SequenceObject_length_{}".format(self.__len__())


    def get_full_sequence(self):
        return self.sequences

    def __iter__(self):
        self.position = 0
        return self

    def __next__(self):
        if self.position == self.__len__():
            raise StopIteration
        end = min(self.__len__(), self.position+self.mode.batch_size)
        batch = [self.__getitem__(k) for k in range(self.position,end)]
        self.position = end
        num_return_seqs = len(batch[0])
        return_seqs = [ [t[i] for t in batch] for i in range(num_return_seqs) ]
        return self.mode.collate_fn(return_seqs)

    # Used for parallel preloading and such
    def _load(self):
        pass

    def _flush(self):
        pass




class VideoSequenceObject(SequenceObject):
    default = Namespace(
            window_size = 10,
            return_transitions = True,
            pad_beginning = True,
            return_global_label = False,
            post_transform = postprocess_1,
            collate_fn = collatefunc_1,
            batch_size = 64, 
            frame_size = 64, 
            frame_interval=3,
            )

    def __init__(self, video_name, video_label, mode=None): #default_sequence_mode):
        self.video_name = video_name
        self.video_label = video_label
        self.mode = mode #Need to init this later

    '''
    Use decord loader to load single video, generate labels, and populate the sequenceObject member variables
    '''
    def _load(self):
        reader = decorder.VideoReader(self.video_name, ctx=decord.cpu(0), width=frame_size, height=frame_size)
        n_frames = len(reader)
        frames = reader.get_batch(np.arange(0,n_frames,self.mode.frame_interval)) #random start location?
        frames = frames.permute(0,3,1,2)
        # ... #call super().__init__() here?

    def _flush(self):
        pass




# Derive various types of sequence objects
# Implement load and flush on them
# Sequences must be numpy arrays
# Sequence objects can return multiple series, and transitions
# Return a boolean array telling whether a point is terminal or not
# Can load batches via __next__ (for processing the individual sequence during testing)
# Can place null values at beginning of sequence




class SequenceDataset:
    default = Namespace(
            sequence_mode = None,
            collate_fn = collatefunc_1,
            batch_size = 64,
            sample_mode = 'random',
            preload_num = None,
            )

    def __init__(self, sequence_objects, options=None): # include curation function and randomization level, e.g. lstm or other
        self.sequences = sequence_objects
        self.options = merge_namespaces(self.__class__.default, options)
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



def default_sample_func(*args):
    return [ a[-1] for a in args ]

# For getting single points from a sequence dataset
# Add ability for models to return embeddings

class SequenceSampleDataset:
    def __init__(self, sequence_dataset, embedding_model=None, sample_func=default_sample_func, collate_fn=collatefunc_2, batch_size=64):
        self.sampled_sequences = []

#        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.collate_fn = collate_fn

        for i in range(sequence_dataset.num_sequences()):
            seq = sequence_dataset.get_sequence(i)
            seq_embeddings = []
            seq_labels = []
            if embedding_model is not None:
                for batch in seq:
                    seq_embeddings.append(embedding_model.embed(batch[0]).detach()) #Should detach
                    seq_labels.append(batch[2]) # batch[2] is conventionally the feature label
            else:
                for batch in seq:
                    seq_embeddings.append(batch[0])
                    seq_labels.append(batch[2]) 
            seq_embeddings = torch.cat(seq_embeddings)
            seq_labels = torch.cat(seq_labels)

            self.sampled_sequences.append(sample_func(seq_embeddings, seq_labels))

            # Need to do something about labels

#        self.sampled_sequences = torch.stack(sampled_sequences)
#        self.sampled_labels = None # Need to write this

    def __iter__(self):
        self.position = 0
        return self

    def __next__(self):
        if self.position == len(self.sampled_sequences):
            raise StopIteration
        end = min(len(self.sampled_sequences), self.position+self.batch_size)
        batch = self.sampled_sequences[self.position:end]
        self.position = end
        num_return_seqs = len(batch[0])
        return_seqs = [ [t[i] for t in batch] for i in range(num_return_seqs) ]
        return self.collate_fn(return_seqs)















