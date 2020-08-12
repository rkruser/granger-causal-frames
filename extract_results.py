import pickle
import torch
import numpy as np
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--results_file', type=str, required=True)
opt = parser.parse_args()


with open(opt.results_file,'rb') as f:
    results = pickle.load(f)

# "train_fullpaths": train_fullpaths,
# "train_times": train_times,
# "train_labels": train_labels,
# "test_fullpaths": test_fullpaths,
# "test_times": test_times,
# "test_labels": test_labels,
# "train_predictions": all_train_predictions,
# "test_predictions": all_test_predictions,
# "config": cfg


train_predictions = results['train_predictions']
test_predictions = results['test_predictions']
train_times = results['train_times']
test_times = results['test_times']
train_labels = results['train_labels']
test_labels = results['test_labels']


# duration and offset based on 30 fps, sampling every 3rd frame, 10 frame window
def get_times(preds, frame_duration=0.1, offset=0.9, threshold=0.5):
    times = []
    for i in range(len(preds)):
        bool_preds = (preds[i] >= threshold)
        if torch.any(bool_preds):
            time = bool_preds.nonzero()[0,0].item()*frame_duration+offset
        else:
            time = -1
        times.append(time)
    return np.array(times)

train_predicted_times = get_times(train_predictions)
test_predicted_times = get_times(test_predictions, threshold=0.5)



######
'''
print(train_predicted_times)
print("----------------")
print(train_times)
print("##########################################################################")
print(test_predicted_times)
print("----------------")
print(test_times)
'''

#print(test_times)

def get_stats(predicted, actual_labels, actual_times, name):
    print("{} Identification Accuracy".format(name))
    print("Percentage positive correctly identified", np.sum(predicted[actual_labels==1]>=0)/(actual_labels==1).sum())
    print("Percentage negative correctly identified", np.sum(predicted[actual_labels==0]<0)/(actual_labels==0).sum())

    print("Test time accuracy")
    inds = (actual_labels==1)&(predicted>=0)
    print("Num positive correct:", inds.sum())
    diffs = predicted[inds] - actual_times[inds] # Negative = predicted in advance, Positive = predicted afterward
    print(diffs)
    print("Mean", diffs.mean())
    print("Median", np.median(diffs))
    print("Std", diffs.std())
    print("Min", diffs.min())
    print("Max", diffs.max())
    print("Mean absolute", np.abs(diffs).mean())
    print("Median absolute", np.median(np.abs(diffs)))
    print("Std absolute", np.abs(diffs).std())
    print("Min abs", np.abs(diffs).min())
    print("Max abs", np.abs(diffs).max())

get_stats(train_predicted_times, train_labels, train_times, "Train")
print("------------------------")
get_stats(test_predicted_times, test_labels, test_times, "Test")
