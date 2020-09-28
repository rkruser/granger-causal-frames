import pickle
import torch
import numpy as np
import argparse
import sys




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
    print("Number identified in advance:", (diffs < 0).sum())
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_file', type=str, required=True)
    parser.add_argument('--sigmoid', action='store_true')
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
    print(type(test_predictions))
    print(len(test_predictions))
    print(test_predictions[0].size())

#    sys.exit()

    if opt.sigmoid:
        print("Applying sigmoid")
        for i in range(len(train_predictions)):
            train_predictions[i] = torch.sigmoid(train_predictions[i])
        for i in range(len(test_predictions)):
            test_predictions[i] = torch.sigmoid(test_predictions[i])

    train_times = results['train_times']
    test_times = results['test_times']
    train_labels = results['train_labels']
    test_labels = results['test_labels']    
    train_predicted_times = get_times(train_predictions)
    test_predicted_times = get_times(test_predictions, threshold=0.5)
    get_stats(train_predicted_times, train_labels, train_times, "Train")
    print("------------------------")
    get_stats(test_predicted_times, test_labels, test_times, "Test")
