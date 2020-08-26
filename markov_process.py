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
transition_matrix = np.array([
  [ 0.334, 0.333, 0.000, 0.000, 0.000, 0.000, 0.333, 0.000 ], # 0
  [ 0.500, 0.000, 0.500, 0.000, 0.000, 0.000, 0.000, 0.000 ], # 1
  [ 0.000, 0.500, 0.000, 0.500, 0.000, 0.000, 0.000, 0.000 ], # 2
  [ 0.000, 0.000, 0.300, 0.000, 0.700, 0.000, 0.000, 0.000 ], # 3
  [ 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000 ], # 4
  [ 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000 ], # 5
  [ 0.500, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.500 ],  # 6
  [ 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000 ]  # 7
])



n_states = len(transition_matrix)
#print(n_states)
#print(len(transition_matrix[0]))

def get_sequence(maxlen = 10):
    state = 0
    index = 0
    states = [state]
    while (state != 4) and index <= maxlen:
        state = np.random.choice(n_states, p=transition_matrix[state])
        states.append(state)
        index += 1

#    if states[-1] == 3:
#        states.append(4)
    if states[-1] == 4: # Alternative transition matrix
        states.append(5)


    return np.array(states)


#state_data_default = np.array([ 0, 1, 2, 3, 4, -1, -2 ])
state_data_default = np.array([ 0, 1, 2, 3, 4, 5, -1, -2 ])
omega_default = 2*np.pi*4*np.sqrt(2)
amplitude = 0.25

def render_sequence(seq, stepsize=0.1):
    durations = 0.5 + 0.5*np.random.rand(2*len(seq))
    durations[np.arange(2,2*len(seq),2)] *= 0.25
    durations[0] = 0
    cumulative = np.cumsum(durations)
    total = cumulative[-1]
    steps = np.linspace(0, total, num = total/stepsize + 1)

    slices = []
    for i, state in enumerate(seq):
        state_data = state_data_default + 0.1*np.random.randn(len(state_data_default))
        omega = omega_default + 0.5*np.random.randn()
        terminal = (i == len(seq)-1)
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
        state_func = lambda x : offset + amplitude*np.sin(omega*(x-start_state))
        state_masked_func_applied = state_func(state_masked)
        slices.append( state_masked_func_applied )       

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


def get_sequences(N, maxlen=20):
    terminal_count = 0
    sequences = []
    states = []
    steplist = []
    labels = []
    cumulatives = []
    for _ in range(N):
        seq = get_sequence(maxlen)
        steps, result, cumulative = render_sequence(seq, stepsize=0.1)
        result += 0.3*np.random.randn(len(result))
        sequences.append(result)
        steplist.append(steps)
        states.append(seq)
        cumulatives.append(cumulative)

        if seq[-1] == 5:
            terminal_count += 1
            labels.append(1)
        else:
            labels.append(0)

    return sequences, np.array(labels), steplist, states, cumulatives, terminal_count / N


mnist = dset.MNIST('/mnt/linuxshared/data/',
                   transform = transforms.Compose([
                       transforms.Resize(64),
                       transforms.ToTensor()
                       ]))

def render_mnist_movie(steps, seq, max_val=255, data_type=np.uint8, noise=0.1):
    frames = []
    digit, label = mnist[np.random.choice(len(mnist))]
    #digit = (maxval*digit.numpy()).astype(data_type)
    digit = digit.squeeze(0).numpy()
    seq = np.clip(seq, -2.5, 5.5)
#    print(seq)
#    sys.exit()
    for x, y in zip(steps, seq):
        new_x = int(((x+5)/35)*223 - 32)
        new_y = int(223 - ((y+5)/13)*223 - 32)
        canvas = np.zeros((224,224))
#        canvas.fill(1.0)
#        canvas[4:220,4:220] = np.zeros((216,216))
        canvas[new_y:new_y+64, new_x:new_x+64] = digit
        frames.append(canvas)

    frames = np.stack(frames)
    if noise > 0:
        frames += noise*2*(np.random.rand(*frames.shape)-0.5)

    frames = (max_val*np.clip(frames, 0, 1)).astype(data_type)

    return frames, label


def get_mnist_sequences(N, maxlen=20, only_even_gets_reward=False, noise=0.4):
    terminal_count = 0
    sequences = []
    states = []
    steplist = []
    labels = []
#    digit_labels = []
    cumulatives = []
    sequences_1d = []
    for j in range(N):
        print("Sequence", j, end='\r', flush=True)
        seq = get_sequence(maxlen)
        steps, result, cumulative = render_sequence(seq, stepsize=0.1)
        result += 0.3*np.random.randn(len(result))
        frames, digit_label = render_mnist_movie(steps, result, noise=noise)
#        frames = np.expand_dims(frames, axis=1)

        sequences.append(frames)
        sequences_1d.append(result)
        steplist.append(steps)
        states.append(seq)
        cumulatives.append(cumulative)
#        digit_labels.append(digit_label)

        # Change this to only be 1 for certain digits
        if seq[-1] == 5:
            keep = (digit_label%2 == 0) if only_even_gets_reward else True
            terminal_count += 1*keep #yes you can multiply bools and numbers
            labels.append(1*keep)
        else:
            labels.append(0)

    return sequences, sequences_1d, np.array(labels), steplist, states, cumulatives, terminal_count / N



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

class SequenceLoader(nn.Module):
    def __init__(self, sequences, labels, steps, states, 
                window_size=10, randomize=True, 
                batch_size=128, return_transitions=True,
                post_transform=None):
        self.sequences = Zipindexables([SequenceWindow(sequences[i], labels[i], 
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
    sequences, labels, steps, states, cumulatives, proportion = get_sequences(100)
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
    train_sequences, train_labels, train_steps, train_states, _, train_proportion = get_sequences(10000)
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

def train_mnist_sequence_net(n_epochs, save_every=5, device='cuda:0', rl_gamma=0.999, terminal_weight=1, out_name='seq_net.pth'):
    net = resnet18_flexible(num_classes=1, data_channels=10)
    optimizer = optim.Adam(net.parameters(), lr=0.0002)
    train_sequences, _, train_labels, train_steps, train_states, _, train_proportion = get_mnist_sequences(1000, noise=0.05)
#    test_sequences, test_labels, test_steps, test_states, test_proportion = get_sequences(1000)
    train_loader = SequenceLoader(train_sequences, train_labels, train_steps, 
                                   train_states, batch_size=64, post_transform=lambda x: x/255.0)
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
    

def test_sequence_net(model_path='sequence_net.pth', device='cpu', threshold=0.5):
    net = SequenceNet()
    net.load_state_dict(torch.load(model_path, map_location=device))

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


def test_mnist_sequence_net(model_path='mnist_resnet18.pth', device='cuda:1', threshold=0.9):
    net = resnet18_flexible(num_classes=1, data_channels=10)
    net.load_state_dict(torch.load(model_path, map_location=device))

    test_sequences, test_sequences_1d, test_labels, test_steps, test_states, cumulatives, test_proportion = get_mnist_sequences(20, noise=0.05)
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

    for i in range(20):
        plt.figure(1)
        plt.ylim(-3, 6)
        plt.plot(test_steps[i], test_sequences_1d[i], label='Sequence')
#        plt.title("Sequence")

#        plt.figure(2)
#        plt.ylim(-0.5, 1.5)

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

    parser.add_argument('--make_mnist_movie', action='store_true')
    parser.add_argument('--movie_name', type=str, default='mnist_out.mp4')
    parser.add_argument('--model_name', type=str, default='model.pth')
    parser.add_argument('--threshold', type=float, default=0.9)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--nepochs', type=int, default=5)
    parser.add_argument('--save_every', type=int, default=1)
    opt = parser.parse_args()
  

    if opt.make_mnist_movie:
        import imageio
        seq = get_sequence(maxlen=20)
        steps, result, _ = render_sequence(seq)
        print(seq)
        movie, label = render_mnist_movie(steps, result, noise=0.05)
        print(label)
        print(type(movie), movie.shape)
        imageio.mimsave(opt.movie_name, movie, fps=10)

    if opt.train_seq:
        train_sequence_net(opt.nepochs, save_every=opt.save_every, out_name = opt.model_name)

    if opt.test_seq:
        test_sequence_net(model_path=opt.model_name, threshold=opt.threshold)

    if opt.train_mnist_seq:
        train_mnist_sequence_net(opt.nepochs, save_every=opt.save_every, out_name=opt.model_name)

    if opt.test_mnist_seq:
        test_mnist_sequence_net(model_path=opt.model_name, threshold=opt.threshold, device=opt.device)


#test_mnist_sequence_net(model_path='mnist_resnet18.pth', threshold=0.9, device='cuda:0')

























