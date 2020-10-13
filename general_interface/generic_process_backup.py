import torch
import torch.nn as nn
import os
import sys
import numpy as np
import time
import argparse
from argparse import Namespace

from loading_utils import Zipindexables


'''
Artificial sequence generators
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
state_mapping_1 = np.array([ 0, 1, 2, 3, 4, 5, -1, -2, 8, 9]) # Come back to the 8,9 later
terminal_states_1 = np.abs(transition_matrix_1.diagonal()-1) < 1e-8
terminal_values_1 = np.array([0.0, 1.0])


'''
Rendering functions:
    For use in artificial sequence generators

'''

def render_sequence_1(states, terminal_label):
    sequence_values = []

    choice_matrix = [[-1,-1],[-1,1],[1,-1],[1,1]] #Can make this correlate with terminal_label instead
    choice = np.random.choice(4)
    feature_label = 1 if choice in [0,3] else 0 #Not linearly separable
    features = choice_matrix[choice]

    for s in states:
        sequence_values.append([s]+features)

    feature_labels = np.empty(len(states))
    feature_labels.fill(feature_label)

    return np.array(sequence_values), feature_labels, feature_label



def render_sequence_2(states, terminal_label):
    sequence_values = []

    choice_matrix = [[-1,-1],[-1,1],[1,-1],[1,1]] #Can make this correlate with terminal_label instead
#    choice = np.random.choice(4)
    if terminal_label == 1:
        choice = np.random.choice(4, p=[0.4, 0.1, 0.1, 0.4])
    else:
        choice = np.random.choice(4, p=[0.1, 0.4, 0.4, 0.1])
    feature_label = 1 if choice in [0,3] else 0 #Not linearly separable
    features = choice_matrix[choice]

    for s in states:
        sequence_values.append([s]+features)

    feature_labels = np.empty(len(states))
    feature_labels.fill(feature_label)

    return np.array(sequence_values), feature_labels, feature_label





'''
Artificial sequence labeling functions (like even only, etc.)
For use in artificial sequence generators
'''
def label_sequence_1(states, terminal_label):
    rewards = np.zeros(len(states))
    rewards[-1] = terminal_label
    return rewards


# For use in derived sequence_object classes
class MarkovProcess:
    def __init__(self, transition_matrix=transition_matrix_1,
                       state_mapping=state_mapping_1,
                       terminal_states = terminal_states_1,
                       terminal_values = terminal_values_1,
                       sequence_render = render_sequence_1,
                       sequence_labeler = label_sequence_1,
                       random_terminal=False,
#                       feature_constructor = feature_constructor_1,
                       ):

        self.transition_matrix = transition_matrix
        self.state_mapping = state_mapping_1
        self.terminal_states = terminal_states
        self.terminal_values = terminal_values
        self.sequence_renderer = sequence_render
        self.sequence_labeler = sequence_labeler
        self.random_terminal = random_terminal
#        self.feature_constructor = feature_constructor_1

        self.n_states = len(self.transition_matrix)
        self.state_indices = np.arange(self.n_states, dtype=int)
        self.terminal_state_labels = self.state_indices[self.terminal_states]
        self.terminal_value_labels = {self.terminal_state_labels[i]:self.terminal_values[i] for i in range(len(self.terminal_values))}

    def sample(self):
        states, terminal_label = self.sample_states()
        rewards = self.sequence_labeler(states, terminal_label)

        sequence, feature_labels, global_feature_label = self.sequence_renderer(states, terminal_label)
        global_labels = (terminal_label, global_feature_label)
        return sequence, rewards, feature_labels, global_labels, states

    def sample_states(self):
        state = 0
        state_sequence = [state]

        while state not in self.terminal_state_labels:
            state = np.random.choice(self.n_states, p=self.transition_matrix[state])
            state_sequence.append(state)

        if self.random_terminal:
            terminal_label = np.random.choice([0.0,1.0])
        else:
            terminal_label = self.terminal_value_labels[state_sequence[-1]]

        mapped_state_sequence = np.array([self.state_mapping[s] for s in state_sequence]) #Map the states

        return mapped_state_sequence, terminal_label


class MnistMarkovProcess:
    def __init__(self):
        pass


# Have a sequence generator that loads embeddings too I guess, by using SequenceDataset to get embeddings for a second sequence dataset









def render_mnist_sequence_1(states):
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

# Normalize first input
def postprocess_1(args):
    return [torch.from_numpy(args[0]).float()/255.0] + [torch.from_numpy(np.array(val)).float() for val in args[1:-1]]+[torch.from_numpy(np.array(args[-1]))]

# No normalizing of input here
def postprocess_2(args):
    return [torch.from_numpy(args[0]).float()] + [torch.from_numpy(np.array(val)).float() for val in args[1:-1]]+[torch.from_numpy(np.array(args[-1]))]

postprocess_test = postprocess_2

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
#    if len(data[0].shape) == 4:
    data = torch.stack(data).transpose(0,1)
#    else:
#        data = torch.stack(data)
    return [data] + [torch.stack(a) for a in args[1:]]

def collatefunc_2(args):
    return [torch.stack(a) for a in args]


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
        collate_fn = collatefunc_1,
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




class MarkovSequenceDataset(SequenceDataset):
    def __init__(self, n_sequences, markov_process, options=default_sequence_dataset_options):
        sequence_objects = []
        for i in range(n_sequences):
            sequence, rewards, feature_labels, global_labels, states = markov_process.sample()
            seq_obj = SequenceObject(sequence, rewards, feature_labels, states, global_label=global_labels, mode=options.sequence_mode)
            sequence_objects.append(seq_obj)

        super().__init__(sequence_objects, options=options)




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



# Derive various types of sequence dataset here that load videos or something








'''
Model functions
'''

class SequenceNet(nn.Module):
    def __init__(self, input_features=10, intermediate_features=256, embedding_features=3):
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
    def __init__(self, input_features=10):
        super().__init__()

        self.net = nn.Linear(input_features,1)

    def embed(self, x):
        pass #Not needed

    def forward(self, x):
        x = x.view(len(x),-1)
        return self.net(x)


def default_network_constructor(network_type='sequence_net', input_features=10, intermediate_features=256, embedding_features=3):
    if network_type == 'sequence_net':
        return SequenceNet(input_features=input_features, intermediate_features=intermediate_features, embedding_features=embedding_features)
    elif network_type == 'linear_net':
        return LinearNet(input_features=input_features)
    else:
        print("Unrecognized network type")
        sys.exit(1)


def default_optim_constructor(network, **kwargs):
    return torch.optim.Adam(network.parameters(), **kwargs)


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


# For simple linear classifiers and such class GenericSimpleModel:
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
