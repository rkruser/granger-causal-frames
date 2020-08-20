import torch
import torch.nn as nn
from cnn_lstm import CNNLSTM
from cnn_dataloader import CNNLSTMDataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tqdm
import numpy as np

network_type = 'resnet101'
hidden_channels = 512
num_layers = 3
if_cnn_trainabel = False

batch_size = 2
image_shape = (224,224,3)
frame_interval = 3
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_num = -1
if_shuffle = False

torch.set_printoptions(sci_mode=False)

load_network_path = 'D:/university/summer_research/20_epoch_model.pth'
save_test_graph_path = 'D:/university/summer_research/result_graphs/'

dpath = 'D:/Beamng_research/recordings/Beamng_dataset_30fps'
test_annotation = "test_annotation.txt"
test_dataset = CNNLSTMDataLoader(dpath, test_annotation,
                                 frame_interval=frame_interval,
                                 device=device,
                                 batch_size=batch_size,
                                 if_shuffle=if_shuffle,
                                 data_num=data_num,
                                 if_return_name=True)
test_loader = iter(test_dataset)

class Model:
    def __init__(self):
        self.network_type = network_type
        self.device = device
        self.prob_loss_func = nn.MSELoss()
        self.network = CNNLSTM(network_type, hidden_channels,
                               num_layers, if_cnn_trainabel).cuda()
        self.network.load_state_dict(torch.load(load_network_path))

        if torch.cuda.device_count() > 1:
            print('use', torch.cuda.device_count(), 'gpus')
            self.network = nn.DataParallel(self.network, device_ids=range(torch.cuda.device_count()))

    def no_grad_forward(self, x):
        with torch.no_grad():
            predictions = self.network(x)
        return predictions

    def eval(self):
        self.network.eval()

    def prob_loss(self, predictions, actual):
        return self.prob_loss_func(predictions, actual)

def test(model, data_loader):
    model.eval()
    val_loss = []
    pbar = tqdm.tqdm(total=len(data_loader))

    for i, batch in enumerate(data_loader):
        x, _, _, actual, names, lengths = batch
        predictions = model.no_grad_forward(x)
        curr_loss = model.prob_loss(predictions, actual)
        val_loss.append(curr_loss.item())

        # if i % 2 == 0:
        #     print((actual + 0.4)[:2])
        #     print(predictions.detach().clone()[:2])
        #     print(names)

        # draw:
        predictions = predictions.detach().cpu().numpy()
        actual = actual.detach().cpu().numpy()

        for i in range(len(names)):
            length = lengths[i]
            name = names[i]
            pred = predictions[i][:length]
            true_label = actual[i][:length]
            frame_x = list(range(length))

            fig = plt.figure()
            plt.plot(frame_x, pred, label='pred')
            plt.plot(frame_x, true_label, label='true')
            plt.xlabel('#frame')
            plt.ylabel('probability')
            plt.legend()
            plt.ylim(0, 1)
            plt.savefig(save_test_graph_path + name + '.jpg')
            plt.close(fig)

        pbar.update(1)

    return np.mean(val_loss)

def main():
    model = Model()
    test_loss = test(model, test_loader)
    print('test loss:', test_loss)

if __name__ == '__main__':
    main()
