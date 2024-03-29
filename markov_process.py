import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import torchvision.datasets as dset
import torchvision.transforms as transforms

from old_code.video_loader import Zipindexables
from flexible_resnet import resnet18_flexible

import sys
import imageio


#     0       1     2      3      4      5      6   
#transition_matrix = np.array([
#  [ 0.334, 0.333, 0.000, 0.000, 0.000, 0.333, 0.000 ], # 0
#  [ 0.500, 0.000, 0.500, 0.000, 0.000, 0.000, 0.000 ], # 1
#  [ 0.000, 0.500, 0.000, 0.500, 0.000, 0.000, 0.000 ], # 2
#  [ 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000 ], # 3
#  [ 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000 ], # 4
#  [ 0.500, 0.000, 0.000, 0.000, 0.000, 0.000, 0.500 ], # 5
#  [ 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000 ]  # 6
#])

#     0       1     2      3      4      5      6      7
#transition_matrix = np.array([
#  [ 0.334, 0.333, 0.000, 0.000, 0.000, 0.000, 0.333, 0.000 ], # 0
#  [ 0.500, 0.000, 0.500, 0.000, 0.000, 0.000, 0.000, 0.000 ], # 1
#  [ 0.000, 0.500, 0.000, 0.500, 0.000, 0.000, 0.000, 0.000 ], # 2
#  [ 0.000, 0.000, 0.300, 0.000, 0.700, 0.000, 0.000, 0.000 ], # 3
#  [ 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000 ], # 4
#  [ 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000 ], # 5
#  [ 0.500, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.500 ], # 6
#  [ 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000 ]  # 7
#])


tp = 0.05
#     0       1     2      3      4      5      6      7      8      9
transition_matrix = np.array([
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


#terminal_rows = np.array([False, False, False, False, False, False, False, False, True, True])
terminal_rows = np.abs(transition_matrix.diagonal()-1) < 1e-8
terminal_values = np.array([0.0, 1.0])
#gamma = 0.999
gamma = 0.977


def analytical_values(M, gamma, terminal_rows, terminal_values):
#    terminal_rows = (M.diagonal()==1)
    m_hat = M[~terminal_rows]
    v_hat = np.dot(m_hat[:,terminal_rows], terminal_values) # or something; assume terminal state is first
    m_hat = m_hat[:,~terminal_rows]
    x_hat = np.linalg.solve(m_hat-(1.0/gamma)*np.eye(len(m_hat)), -v_hat)
    x = np.empty(len(M))
    x[~terminal_rows] = x_hat
    x[terminal_rows] = terminal_values

    return x
    

# terminal Need to add terminal transition and re-tune markov chain
# Retrain markov chain model
# Add graphing of analytical values




n_states = len(transition_matrix)
#print(n_states)
#print(len(transition_matrix[0]))


# Altered to not use max length
def get_sequence(maxlen = None):
    state = 0
    index = 1 # !!!!!!!!
    states = [state]

    if maxlen is not None:
        while (state <= 7 and index <= maxlen):
            state = np.random.choice(n_states, p=transition_matrix[state])
            states.append(state)
            index += 1
    else:
        while (state <= 7):
            state = np.random.choice(n_states, p=transition_matrix[state])
            states.append(state)
            index += 1

    return np.array(states)

# non-terminal states
state_data_default = np.array([ 0, 1, 2, 3, 4, 5, -1, -2 ])
omega_default = 2*np.pi*4*np.sqrt(2)
amplitude = 0.25

def render_sequence(seq, stepsize=0.1):
    durations = 0.5 + 0.5*np.random.rand(2*(len(seq)-1)) # len(seq)-1 to get rid of terminal endpoint
    durations[np.arange(2,2*(len(seq)-1),2)] *= 0.25
    durations[0] = 0
    cumulative = np.cumsum(durations)
    total = cumulative[-1]
    steps = np.linspace(0, total, num = total/stepsize + 1)

    slices = []
    for i, state in enumerate(seq[:-1]):
        state_data = state_data_default + 0.1*np.random.randn(len(state_data_default))
        omega = omega_default + 0.5*np.random.randn()
        terminal = (i == len(seq)-2) # don't include very last state
        start_state = cumulative[2*i]
        stop_state = cumulative[2*i+1]
        if terminal:
            stop_lin = stop_state
            next_state = None
        else:
            stop_lin = cumulative[2*(i+1)]
            next_state = seq[i+1]

        state_masked = steps[(steps>=start_state)&(steps<=stop_state)]
        lin_masked = steps[(steps>stop_state)&(steps<stop_lin)]
        lin_duration = stop_lin-stop_state

        # Instead of ifs, use an array of sine shifts/params

        # Also, account for case where next state is current state

        offset = state_data[state]
#        state_func = lambda x : offset + amplitude*np.sin(omega*(x-start_state))
#        state_masked_func_applied = state_func(state_masked)
#        slices.append( state_masked_func_applied )       
        state_masked_func_applied = offset + amplitude*np.sin(omega*(state_masked-start_state))
        slices.append(state_masked_func_applied)

#        if next_state == state:
#            slices.append( state_func(lin_masked) )
#        else:
        if next_state is not None:
            next_offset = state_data[next_state]
            intercept = state_masked_func_applied[-1]
            slope = (next_offset-intercept)/lin_duration
            slices.append( intercept + slope*(lin_masked-stop_state) )
        
    return steps, np.concatenate(slices), cumulative



# Maxlen=16 yields roughly 50/50
# With new transition matrix, maxlen=20 yields roughly 50/50
#def get_sequences(N, maxlen=10, plot=False):
#    terminal_count = 0
#    for _ in range(N):
#        seq = get_sequence(maxlen)
#
#        if plot:
#            print(seq)
#            steps, result = render_sequence(seq, stepsize=0.1)
#            result += 0.3*np.random.randn(len(result))
#            plt.plot(steps, result)
#            plt.ylim(-3, 6)
#            plt.show()
#            input("Press enter")
#            plt.close()
#
##        if seq[-1] == 4:
##            terminal_count += 1
#        if seq[-1] == 5:
#            terminal_count += 1
#
#
#    return terminal_count / N


def get_sequences(N, maxlen=None, noise_coeff=0.3):
    ones_count = 0
    sequences = []
    states = []
    steplist = []
    labels = []
    cumulatives = []
    for _ in range(N):
        seq = get_sequence(maxlen)
        steps, result, cumulative = render_sequence(seq, stepsize=0.1)
        result += noise_coeff*np.random.randn(len(result))
        sequences.append(result)
        steplist.append(steps)
        states.append(seq)
        cumulatives.append(cumulative)

        if seq[-1] == 9:
            ones_count += 1
            labels.append(1)
        else:
            labels.append(0)

    return sequences, np.array(labels), steplist, states, cumulatives, ones_count / N


mnist = dset.MNIST('/mnt/linuxshared/data/',
                   transform = transforms.Compose([
                       transforms.Resize(64),
                       transforms.ToTensor()
                       ]))

def render_mnist_movie(steps, seq, max_val=255, data_type=np.uint8, noise=0.1, preappend_num=None):
    frames = []
    digit, label = mnist[np.random.choice(len(mnist))]
    #digit = (maxval*digit.numpy()).astype(data_type)
    digit = digit.squeeze(0).numpy()
    seq = np.clip(seq, -2.5, 5.5)
#    print(seq)
#    sys.exit()
#    inversion = False
#    lower = (5/25)*223-32
    for x, y in zip(steps, seq):
        new_x = int((x/10)*223)
        remainder = new_x % 160
        quotient = new_x // 160
        new_x = 160-remainder if quotient%2 == 1 else remainder
#        if new_x < lower or new_x > 150:
#            inversion = (not inversion)
#        new_x = int(new_x) # Do this afterward to prevent multiple inversion



        new_y = int(223 - ((y+5)/13)*223 - 32)
        canvas = np.zeros((224,224))
#        canvas.fill(1.0)
#        canvas[4:220,4:220] = np.zeros((216,216))
        canvas[new_y:new_y+64, new_x:new_x+64] = digit
        frames.append(canvas)



    frames = np.stack(frames)

    if preappend_num is not None:
        frames = np.concatenate((np.zeros((preappend_num, 224, 224)), frames))

    if noise > 0:
        frames += noise*2*(np.random.rand(*frames.shape)-0.5)

    frames = (max_val*np.clip(frames, 0, 1)).astype(data_type)

    return frames, label


def get_mnist_sequences(N, maxlen=None, only_even_gets_reward=False, digit_reward_only=False, noise=0.4, seq_noise = 0.3,
                        return_digit_label=False, preappend_num=None):
    crash_count = 0
    sequences = []
    states = []
    steplist = []
    labels = []
    digit_labels = []
    cumulatives = []
    sequences_1d = []
    for j in range(N):
        print("Sequence", j, end='\r', flush=True)
        seq = get_sequence(maxlen)
        steps, result, cumulative = render_sequence(seq, stepsize=0.1)
        result += seq_noise*np.random.randn(len(result))
        frames, digit_label = render_mnist_movie(steps, result, noise=noise, preappend_num=preappend_num)
#        frames = np.expand_dims(frames, axis=1)

        sequences.append(frames)
        sequences_1d.append(result)
        steplist.append(steps)
        states.append(seq)
        cumulatives.append(cumulative)
        digit_labels.append(digit_label)

        # Change this to only be 1 for certain digits
        if not digit_reward_only:
            if seq[-1] == 9:
                keep = (digit_label%2 == 0) if only_even_gets_reward else True
                crash_count += 1*keep #yes you can multiply bools and numbers
                labels.append(1*keep)
            else:
                labels.append(0)
        else:
            labels.append(int(digit_label%2==0))


    if return_digit_label:
        return sequences, sequences_1d, np.array(labels), steplist, states, cumulatives, crash_count / N, digit_labels
    else:
        return sequences, sequences_1d, np.array(labels), steplist, states, cumulatives, crash_count / N



#def process_sequences(seqs, labels, window_size=10):

class SequenceWindow:
    def __init__(self, seq, label, window_size, return_transitions=True):
        self.seq = seq
        self.window_size = window_size
        self.label = label
        self.return_transitions = return_transitions

        self._length = max(len(self.seq)-self.window_size+1, 0)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        data = self.seq[i:i+self.window_size]
        if i == self._length-1:
            r = self.label
            is_terminal = True
        else:
            r = 0
            is_terminal = False

        if self.return_transitions:
            if is_terminal:
                future = np.zeros(data.shape)
            else:
                future = self.seq[(i+1):(i+1+self.window_size)]
            data = np.stack((data, future))
            

        return data, r, is_terminal


class SequenceWindowEmbedded:
    def __init__(self, seq, label, window_size, return_transitions=True):
        self.seq = seq
        self.window_size = window_size
        self.label = label
        self.return_transitions = return_transitions

        self._length = max(len(self.seq)-self.window_size+1, 0)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        data = self.seq[i:i+self.window_size]
        if i == self._length-1:
            r = self.label
            is_terminal = True
        else:
            r = self.label
            is_terminal = False

        if self.return_transitions:
            if is_terminal:
                future = np.zeros(data.shape)
            else:
                future = self.seq[(i+1):(i+1+self.window_size)]
            data = np.stack((data, future))
            

        return data, r, is_terminal


class SequenceLoader(nn.Module):
    # Note: steps and states are currently unused
    # Due to preappend, steps/states length may not match sequence length
    def __init__(self, sequences, labels, steps, states, 
                window_size=10, randomize=True, 
                batch_size=128, return_transitions=True,
                post_transform=None,
                windowclass=SequenceWindow):
        self.sequences = Zipindexables([windowclass(sequences[i], labels[i], 
                                        window_size, return_transitions=return_transitions) for i in range(len(sequences))])
        self.labels = labels
        self.steps = steps
        self.states = states

        self.window_size = window_size
        self.batch_size = batch_size
        self.randomize = randomize
        self.return_transitions = return_transitions
        self.post_transform = post_transform

        self.indices = np.arange(len(self.sequences))
        self.position = 0

        self._length = len(self.sequences)//self.batch_size
        if len(self.sequences)%self.batch_size > 0:
            self._length += 1
        

    def _reset(self):
        if self.randomize:
            self.indices = np.random.permutation(self.indices)
        self.position = 0

    def __len__(self):
        return self._length
    
    def __iter__(self):
        self._reset()
        return self

    def __next__(self):
        if self.position == len(self.sequences):
            raise StopIteration
        end = min(len(self.sequences), self.position+self.batch_size)
        batch = self.sequences[self.indices[self.position:end]]
        data_batch, rewards, terminal = zip(*batch)
        self.position = end


        data_batch = torch.from_numpy(np.stack(data_batch)).float()
        
        if self.post_transform is not None:
            data_batch = self.post_transform(data_batch) #for example, divide by 255

        if self.return_transitions:
            data_batch = data_batch.transpose(1,0)

        return data_batch, torch.Tensor(rewards), torch.BoolTensor(terminal)

        
def test_sequence_loader():
    sequences, labels, steps, states, cumulatives, proportion = (100)
    loader = SequenceLoader(sequences, labels, steps, states, batch_size=128)

    print(len(loader))
    for s in loader:
        data, rewards, terminal = s
        print(data.shape)
        print(data)
        print(rewards)
        print(terminal)
        input("enter")

#test_sequence_loader()


def embedded_mnist_loader(N, load_from, device='cuda:0', preappend_num=None, maxlen=10):
    mnist_seqs, _, _, _, _, _, _, digit_labels = get_mnist_sequences(N, maxlen=maxlen, 
                                                                    only_even_gets_reward=True, 
                                                                    noise=0.001, 
                                                                    seq_noise = 0.001,
                                                                    return_digit_label=True,
                                                                    preappend_num=preappend_num)
    loader = SequenceLoader(mnist_seqs, digit_labels, None, None, return_transitions=False, randomize=True, batch_size=64, window_size=10, post_transform=lambda x: x/255.0, windowclass=SequenceWindowEmbedded)

    net = resnet18_flexible(num_classes=1, data_channels=10, return_features=True)
    net = net.to(device)

    if load_from is not None:
        print("loading embedding model from {}".format(load_from))
        net.load_state_dict(torch.load(load_from, map_location=device))
    
    net.eval()

    batch_features, batch_labels = [], []
#    for batch in loader: 
#        x, y, _ = batch
#        x = x.to(device)
#        _, features = net(x)
#        features = features.detach().cpu()
#        batch_features.append(features)
#        batch_labels.append(y)
    for i in range(loader.sequences.num_indexables()):
        batch_window_loader = loader.sequences.get_indexable(i)
        x, y, _ = batch_window_loader[len(batch_window_loader)-1]
        x = torch.from_numpy(x).float().unsqueeze(0) / 255.0

        x = x.to(device)
        _, features = net(x)
        features = features.detach().cpu()
        batch_features.append(features)
        batch_labels.append(y)


    return batch_features, torch.Tensor(batch_labels)


def raw_mnist_frame_loader(N, device='cuda:0', preappend_num=9, maxlen=1):

    mnist_seqs, _, _, _, _, _, _, digit_labels = get_mnist_sequences(N, maxlen=maxlen, 
                                                                    only_even_gets_reward=True, 
                                                                    noise=0.001, 
                                                                    seq_noise = 0.001,
                                                                    return_digit_label=True,
                                                                    preappend_num=preappend_num)
    loader = SequenceLoader(mnist_seqs, digit_labels, None, None, return_transitions=False, randomize=True, batch_size=64, window_size=10, post_transform=lambda x: x/255.0, windowclass=SequenceWindow)

    batch_frames, batch_labels = [], []
    for batch in loader:
        x, y, _ = batch
        batch_size = x.size(0)
        x = x[:,9,:,:].unsqueeze(1)
        x = torch.nn.functional.interpolate(x,64) #Make smaller for tractability
        x = x.view(batch_size,-1) #Only first frame
        x = x.to('cpu') #??
        batch_frames.append(x)
        batch_labels.append(y)

    return batch_frames, batch_labels


#  Can train by iterating through chucks of the above

    # To do:
    #  Add maxlen back into sequence creation because don't want to generate excess frames. Fix up other functions that this disrupts.
    #  Add ability of flexible resnet to return embeddings
    #  Generate lots of mnist vids with with exactly 10 frames and put them in sequence loader;
    #    use the digit_label as the label
    #  Iterate through the vids and get the embeddings and the digit labels
    # Then put the embeddings back into a sequence loader and return than


class ParityNet(nn.Module):
    def __init__(self, input_size=512*4):
        super().__init__()

        self.net = nn.Sequential(
                nn.Linear(input_size, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
                )

    def forward(self, x):
        return self.net(x)

class LinearParityNet(nn.Module):
    def __init__(self, input_size=512*4):
        super().__init__()

        self.net = nn.Linear(input_size,1)

    def forward(self, x):
        return self.net(x)



def train_parity_net(preappend_num=9, max_pt_len=1, parity_model='embedded', embedding_model='mnist_even_only_977_low_noise_15.pth', n_train_pts = 100, parity_net_name='parity_net.pth', nepochs=100):

    if parity_model == 'embedded':
        print("Training parity net on embeddings")

        parity_net = LinearParityNet(2048)
        batch_features, batch_labels = embedded_mnist_loader(n_train_pts, embedding_model,
                                                maxlen=max_pt_len, preappend_num=preappend_num)
    elif parity_model == 'raw':
        print("Training parity net on raw")

        parity_net = LinearParityNet(4096)
        batch_features, batch_labels = raw_mnist_frame_loader(n_train_pts,
                                                maxlen=max_pt_len, preappend_num=preappend_num)
    else:
        print("Unrecognized parity model")
        sys.exit(1)



    optimizer = optim.Adam(parity_net.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    

    for epoch in range(nepochs):
        print("Epoch", epoch)
        total_loss = 0.0
        iteration = 0
        total_accuracy = 0.0
        total_count = 0
        for batch, labels in zip(batch_features, batch_labels):
            iteration += 1
            predictions = parity_net(batch).squeeze()
            labels = (labels%2==0).float()
            loss = criterion(predictions, labels)

            parity_net.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            total_accuracy += ((predictions >= 0.5) == (labels >= 0.5)).sum()
            total_count += len(labels)

            if iteration%100 == 0:
                print("Loss: {0}, accuracy: {1}".format(total_loss / iteration, total_accuracy/total_count))

        print("Saving")
        torch.save(parity_net.state_dict(), parity_net_name)

def test_parity_net(preappend_num=9, max_pt_len=1, parity_model='embedded', embedding_model='mnist_even_only_977_low_noise_15.pth', n_test_pts = 100, parity_net_name='parity_net.pth'):

    if parity_model == 'embedded':
        print("Testing parity net on embeddings")

        parity_net = LinearParityNet(2048)
        parity_net.load_state_dict(torch.load(parity_net_name,map_location='cpu'))

        batch_features, batch_labels = embedded_mnist_loader(n_test_pts, embedding_model,
                                                maxlen=max_pt_len, preappend_num=preappend_num)
    elif parity_model == 'raw':
        print("Testing parity net on raw")

        parity_net = LinearParityNet(4096)
        parity_net.load_state_dict(torch.load(parity_net_name,map_location='cpu'))
        batch_features, batch_labels = raw_mnist_frame_loader(n_test_pts,
                                                maxlen=max_pt_len, preappend_num=preappend_num)
    else:
        print("Unrecognized parity model")
        sys.exit(1)



    iteration = 0
    total_accuracy = 0.0
    total_count = 0
    for batch, labels in zip(batch_features, batch_labels):
        iteration += 1
        predictions = parity_net(batch).squeeze()
        labels = (labels%2==0).float()

        total_accuracy += ((predictions >= 0.5) == (labels >= 0.5)).sum()
        total_count += len(labels)

        print("accuracy: {0}".format(total_accuracy/total_count))

#seq = np.array([0,1,2,3,4])
#render_sequence(seq)


class SequenceNet(nn.Module):
    def __init__(self, input_features=10, intermediate_features=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_features, intermediate_features),
            nn.ReLU(),
            nn.Linear(intermediate_features, intermediate_features),
            nn.ReLU(),
            nn.Linear(intermediate_features, intermediate_features),
            nn.ReLU(),
            nn.Linear(intermediate_features, intermediate_features),
            nn.ReLU(),
            nn.Linear(intermediate_features, 1),
#            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def train_sequence_net(n_epochs, save_every=5, device='cuda:0', rl_gamma=0.999, terminal_weight=1, out_name='seq_net.pth'):
    net = SequenceNet()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    train_sequences, train_labels, train_steps, train_states, _, train_proportion = get_sequences(2000)
#    test_sequences, test_labels, test_steps, test_states, test_proportion = get_sequences(1000)
    train_loader = SequenceLoader(train_sequences, train_labels, train_steps, train_states, batch_size=128)
#    test_loader = SequenceLoader(test_sequences, test_labels, test_steps, test_states, batch_size=128, randomize=False)


    net = net.to(device)
    net.train()
    for epoch in range(n_epochs):
        print("Epoch", epoch)
        
        total_loss = 0.0

        for i,batch in enumerate(train_loader):
            data, rewards, terminal = batch
            data = data.to(device)
            rewards = rewards.to(device)
            terminal = terminal.to(device)

            data_cur = data[0]
            data_fut = data[1]

            q_cur = net(data_cur).squeeze(1)
            

            weights = torch.ones(len(rewards), device=device)
            with torch.no_grad():
                weights[terminal] = terminal_weight
                fut_preds = net(data_fut).squeeze(1)
                fut_preds[terminal] = 0
                q_future = rl_gamma*fut_preds+rewards

            diffs = q_future-q_cur
            loss = (weights*diffs**2).mean()

            net.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i+1)%100 == 0:
                print("Batch", i, "of", len(train_loader), "Average loss:", total_loss/i)


        if (epoch+1)%save_every == 0 or epoch == n_epochs-1:
            print("Saving")
            torch.save(net.state_dict(), out_name)

def train_mnist_sequence_net(n_epochs, save_every=1, device='cuda:0', rl_gamma=0.999, terminal_weight=1, out_name='seq_net.pth', load_from=None, only_even=False, sequence_noise=0.1, image_noise=0.05, preappend_num=None, digit_reward_only = False):
    net = resnet18_flexible(num_classes=1, data_channels=10)
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    if load_from is not None:
        print("loading from {}".format(load_from))
        net.load_state_dict(torch.load(load_from, map_location=device))

    train_sequences, _, train_labels, train_steps, train_states, _, train_proportion = get_mnist_sequences(500, noise=image_noise, seq_noise=sequence_noise, only_even_gets_reward = only_even, 
            digit_reward_only=digit_reward_only, preappend_num=preappend_num)

    if only_even:
        print("Only even rewards")

#    test_sequences, test_labels, test_steps, test_states, test_proportion = get_sequences(1000)
    train_loader = SequenceLoader(train_sequences, train_labels, train_steps, 
                                   train_states, batch_size=64, post_transform=lambda x: x/255.0)
#    test_loader = SequenceLoader(test_sequences, test_labels, test_steps, test_states, batch_size=128, randomize=False)


    net.train()
    for epoch in range(n_epochs):
        print("Epoch", epoch)
        
        total_loss = 0.0

        for i,batch in enumerate(train_loader):
            data, rewards, terminal = batch
            data = data.to(device)
            rewards = rewards.to(device)
            terminal = terminal.to(device)

            data_cur = data[0]
            data_fut = data[1]

            q_cur = net(data_cur).squeeze(1)
            

            weights = torch.ones(len(rewards), device=device)
            with torch.no_grad():
                weights[terminal] = terminal_weight
                fut_preds = net(data_fut).squeeze(1)
                fut_preds[terminal] = 0
                q_future = rl_gamma*fut_preds+rewards

            diffs = q_future-q_cur
            loss = (weights*diffs**2).mean()

            net.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i+1)%10 == 0:
                print("Batch", i, "of", len(train_loader), "Average loss:", total_loss/i)


        if (epoch+1)%save_every == 0 or epoch == n_epochs-1:
            print("Saving")
            torch.save(net.state_dict(), out_name)



def get_prediction(steps, preds, threshold=0.5):
    thresholded = (preds >= threshold)
    if torch.any(thresholded):
        index = torch.nonzero(thresholded)[0].item()
        return steps[index]
    else:
        return None

def get_stats(predicted, actual, labels, true_pos):
    diffs = predicted-actual
    print("Length before", len(diffs))
    diffs = diffs[true_pos]
    print("All positives", np.sum(labels))
    print("True predicted positives", len(diffs))
    print("Recall", len(diffs)/np.sum(labels))
    print("Balance of signs", np.sum(np.sign(diffs)))
    print("Magnitudes", np.abs(diffs).mean())
    

def get_true_plot(states, steps, cumulatives, analytical_vals):
    true_seq = np.zeros(len(steps))
    for i, t in enumerate(cumulatives[:-1]):
        window = (steps >= cumulatives[i]) & (steps < cumulatives[i+1])
        state_index = i//2
        val = analytical_vals[states[state_index]]
        true_seq[window] = val

    return true_seq



def test_sequence_net(model_path='sequence_net.pth', device='cpu', threshold=0.5, gamma=0.977):
    net = SequenceNet()
    net.load_state_dict(torch.load(model_path, map_location=device))

    analytical_vals = analytical_values(transition_matrix, gamma, terminal_rows, terminal_values)


    test_sequences, test_labels, test_steps, test_states, cumulatives, test_proportion = get_sequences(1000)
    test_loader = SequenceLoader(test_sequences, test_labels, test_steps, test_states, 
                                batch_size=128, randomize=False, return_transitions=False)

    print("Test proportion", test_proportion)

    all_predictions = []
    all_terminal = []
    for i,batch in enumerate(test_loader):
        if (i+1)%100 == 0:
            print("Batch", i, "of", len(test_loader))
        data, _, terminal = batch
        predictions = net(data).detach().squeeze(1)
        all_predictions.append(predictions)
        all_terminal.append(terminal)

    predictions = torch.cat(all_predictions)
    terminal = torch.cat(all_terminal)

    split_inds = torch.arange(len(terminal))[terminal]+1
    
    test_predictions = []
    prev_low = 0
    for ind in split_inds:
        test_predictions.append(predictions[prev_low:ind])
        prev_low = ind


    assert(len(test_predictions) == len(test_sequences))
    length_check = True
    for s1, s2 in zip(test_sequences,test_predictions):
        if len(s2) != len(s1)-9:
            length_check = False
    assert(length_check)

#    threshold = 0.5
    predicted_times = np.zeros(len(test_sequences))
    stable_certain_times = np.zeros(len(test_sequences))
    true_pos = np.zeros(len(test_sequences),dtype='bool')
    for i in range(len(test_sequences)):
        pred = get_prediction(test_steps[i][9:], test_predictions[i], threshold=threshold)
        if test_labels[i] and pred:
            predicted_times[i] = pred
            stable_certain_times[i] = cumulatives[i][-4]
            true_pos[i] = True


    get_stats(predicted_times, stable_certain_times, test_labels, true_pos)

    for i in range(20):
        plt.figure(1)
        plt.ylim(-3, 6)
        plt.plot(test_steps[i], test_sequences[i], label='Sequence')
#        plt.title("Sequence")

#        plt.figure(2)
#        plt.ylim(-0.5, 1.5)


        true_vals = get_true_plot(test_states[i], test_steps[i], cumulatives[i], analytical_vals)
        plt.plot(test_steps[i], 5*true_vals, label='Analytical values')

        predicted_time = get_prediction(test_steps[i][9:], test_predictions[i], threshold=threshold)
        if predicted_time is not None:
            plt.vlines(predicted_time, -3, 6, colors='b', label='Predicted')

        if test_labels[i]:
            at_least_70_percent = cumulatives[i][-6]
            earliest_certain = cumulatives[i][-5]
            stable_certain = cumulatives[i][-4]
            true_lines = np.array([at_least_70_percent, earliest_certain, stable_certain])
            plt.vlines(true_lines, -3, 6, linestyles='dashed', colors='r', label='Actual')

        plt.plot(test_steps[i][9:], 5*test_predictions[i].numpy(), label='Predictions')

        plt.legend()
        plt.show()

#        input("enter")

        plt.close(1)
#        plt.close(2)


def test_mnist_sequence_net(model_path='mnist_resnet18.pth', device='cuda:0', threshold=0.9, only_even=False, gamma=0.977, preappend_num=None):
    net = resnet18_flexible(num_classes=1, data_channels=10)
    net.load_state_dict(torch.load(model_path, map_location=device))

    analytical_vals = analytical_values(transition_matrix, gamma, terminal_rows, terminal_values)


    test_sequences, test_sequences_1d, test_labels, test_steps, test_states, cumulatives, test_proportion = get_mnist_sequences(10, noise=0.05, seq_noise=0.1, only_even_gets_reward=only_even, preappend_num=preappend_num)

    if only_even:
        print("Only even rewards")

#    train_sequences, train_labels, train_steps, train_states, _, train_proportion = get_mnist_sequences(1000)

    test_loader = SequenceLoader(test_sequences, test_labels, test_steps, test_states, 
                                batch_size=64, randomize=False, return_transitions=False, post_transform=lambda x: x/255.0)

    print("Test proportion", test_proportion)

    all_predictions = []
    all_terminal = []
    for i,batch in enumerate(test_loader):
        if (i+1)%10 == 0:
            print("Batch", i, "of", len(test_loader))
        data, _, terminal = batch
        predictions = net(data).detach().squeeze(1)
        all_predictions.append(predictions)
        all_terminal.append(terminal)

    predictions = torch.cat(all_predictions)
    terminal = torch.cat(all_terminal)

    split_inds = torch.arange(len(terminal))[terminal]+1
    
    test_predictions = []
    prev_low = 0
    for ind in split_inds:
        test_predictions.append(predictions[prev_low:ind])
        prev_low = ind


    assert(len(test_predictions) == len(test_sequences))
    length_check = True
    for s1, s2 in zip(test_sequences_1d,test_predictions):
        if len(s2) != len(s1)-9:
            length_check = False
    assert(length_check)

#    threshold = 0.5
    predicted_times = np.zeros(len(test_sequences_1d))
    stable_certain_times = np.zeros(len(test_sequences_1d))
    true_pos = np.zeros(len(test_sequences_1d),dtype='bool')
    for i in range(len(test_sequences_1d)):
        pred = get_prediction(test_steps[i][9:], test_predictions[i], threshold=threshold)
        if test_labels[i] and pred:
            predicted_times[i] = pred
            stable_certain_times[i] = cumulatives[i][-4]
            true_pos[i] = True

    get_stats(predicted_times, stable_certain_times, test_labels, true_pos)

    for i in range(10):
        plt.figure(1)
#        plt.ylim(-3, 6)
        plt.ylim(-0.5,1.5)
        imageio.mimsave('test_movie.mp4', test_sequences[i], fps=10)
        plt.title("Label: {}".format(test_labels[i]))

#        plt.plot(test_steps[i], test_sequences_1d[i], label='Sequence')
##        plt.title("Sequence")

        true_vals = get_true_plot(test_states[i], test_steps[i], cumulatives[i], analytical_vals)
#        plt.plot(test_steps[i], 5*true_vals, label='Analytical values')


##        plt.figure(2)
##        plt.ylim(-0.5, 1.5)

#        predicted_time = get_prediction(test_steps[i][9:], test_predictions[i], threshold=threshold)
#        if predicted_time is not None:
#            plt.vlines(predicted_time, -3, 6, colors='b', label='Predicted')

#        if test_labels[i]:
#            at_least_70_percent = cumulatives[i][-6]
#            earliest_certain = cumulatives[i][-5]
#            stable_certain = cumulatives[i][-4]
#            true_lines = np.array([at_least_70_percent, earliest_certain, stable_certain])
#            plt.vlines(true_lines, -3, 6, linestyles='dashed', colors='r', label='Actual')

        # Removed the 5x
        plt.plot(test_steps[i][9:], test_predictions[i].numpy(), label='Predictions')

        plt.legend()
        plt.show()

##        input("enter")

        plt.close(1)
##        plt.close(2)




#train_sequence_net(5, terminal_weight=1, out_name = 'sequence_net_balanced.pth')
#test_sequence_net(model_path='sequence_net_balanced.pth', threshold=0.7)


# Idea: render the above sequences directly into numpy stuff
# Maybe also stipulate that some numbers lead to reward and some don't







# ****** NOTE *****
# Use terminal weights of 1, not 64, because mathematically this is sound (in terms of probabilistic interpretation)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_seq', action='store_true')
    parser.add_argument('--test_seq', action='store_true')
    parser.add_argument('--train_mnist_seq', action='store_true')
    parser.add_argument('--test_mnist_seq', action='store_true')
    parser.add_argument('--test_program', action='store_true')
    parser.add_argument('--terminal_values', action='store_true')
    parser.add_argument('--train_parity_net', action='store_true')
    parser.add_argument('--test_parity_net', action='store_true')

    parser.add_argument('--make_mnist_movie', action='store_true')
    parser.add_argument('--movie_name', type=str, default='mnist_out.mp4')
    parser.add_argument('--model_name', type=str, default='model.pth')
    parser.add_argument('--threshold', type=float, default=0.9)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--nepochs', type=int, default=5)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.977)
    parser.add_argument('--only_even', action='store_true')
    parser.add_argument('--digit_reward_only', action='store_true')
    parser.add_argument('--seq_noise', type=float, default=0.1)
    parser.add_argument('--image_noise', type=float, default=0.05)
    parser.add_argument('--load_from', type=str, default=None)
    parser.add_argument('--preappend_num', type=int, default=9)

    parser.add_argument('--parity_net_max_pt_len', type=int, default=1)
    parser.add_argument('--parity_model', type=str, default='embedded')
    parser.add_argument('--parity_embeddings_model_file', type=str, default=None)
    parser.add_argument('--parity_model_file', type=str, default=None)
    parser.add_argument('--parity_net_train_points', type=int, default=100)
    parser.add_argument('--parity_net_test_points', type=int, default=500)
    parser.add_argument('--parity_net_epochs', type=int, default=100)


    opt = parser.parse_args()
  

    if opt.train_parity_net:
        train_parity_net(preappend_num=opt.preappend_num, 
                        max_pt_len=opt.parity_net_max_pt_len,
                        parity_model = opt.parity_model,
                        embedding_model=opt.parity_embeddings_model_file,
                        n_train_pts=opt.parity_net_train_points,
                        parity_net_name=opt.parity_model_file,
                        nepochs = opt.parity_net_epochs)

    if opt.test_parity_net:
        test_parity_net(preappend_num=opt.preappend_num, 
                        max_pt_len=opt.parity_net_max_pt_len,
                        parity_model = opt.parity_model,
                        embedding_model=opt.parity_embeddings_model_file,
                        n_test_pts=opt.parity_net_test_points,
                        parity_net_name=opt.parity_model_file)

    if opt.make_mnist_movie:
        import imageio
        seq = get_sequence()
        while len(seq) < 50:
            seq = get_sequence()
        steps, result, _ = render_sequence(seq)
        print(seq)
        movie, label = render_mnist_movie(steps, result, noise=0.2)
        print(label)
        print(type(movie), movie.shape)
        imageio.mimsave(opt.movie_name, movie, fps=10)

    if opt.test_program:
        seqs, labels, steps, states, cumulatives, p = get_sequences(100)
        print(p)

    if opt.train_seq:
        train_sequence_net(opt.nepochs, rl_gamma=opt.gamma, save_every=opt.save_every, out_name = opt.model_name)

    if opt.test_seq:
        test_sequence_net(model_path=opt.model_name, threshold=opt.threshold, gamma=opt.gamma)

    if opt.train_mnist_seq:
        train_mnist_sequence_net(opt.nepochs, rl_gamma=opt.gamma, device=opt.device, 
                                save_every=opt.save_every, 
                                out_name=opt.model_name, 
                                load_from=opt.load_from,
                                only_even=opt.only_even,
                                digit_reward_only=opt.digit_reward_only,
                                image_noise=opt.image_noise,
                                sequence_noise=opt.seq_noise,
                                preappend_num=opt.preappend_num)

    if opt.terminal_values:
        vals = analytical_values(transition_matrix, opt.gamma, terminal_rows, terminal_values)
        print(vals)

    if opt.test_mnist_seq:
        test_mnist_sequence_net(model_path=opt.model_name, threshold=opt.threshold, device=opt.device, only_even=opt.only_even, gamma=opt.gamma, preappend_num=opt.preappend_num)


#test_mnist_sequence_net(model_path='mnist_resnet18.pth', threshold=0.9, device='cuda:0')







# preappend_num=9, max_pt_len=1, parity_model='embedded', embedding_model='mnist_even_only_977_low_noise_15.pth', n_test_pts = 100, parity_net_name='parity_net.pth'

















