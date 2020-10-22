import numpy as np
from argparse import Namespace
import matplotlib.pyplot as plt
import os

# Take a numpy array of predictions and find the first index meeting
#  or exceeding the threshold. Return -1 if none found.
def get_predicted_index(predictions, threshold):
    indices = np.nonzero(predictions >= threshold)[0]
    if len(indices) > 0:
        return indices[0]
    return -1

def array_stats(arr):
    if len(arr) > 0:
        avg = np.mean(arr)
        std = np.std(arr)
        mean_abs_deviation = np.mean(np.abs(arr-avg))
        amin, q1, med, q2, amax = np.min(arr), np.percentile(arr,25), np.median(arr), np.percentile(arr, 75), np.max(arr)

        return avg, std, mean_abs_deviation, amin, q1, med, q2, amax

    return None, None, None, None, None, None, None, None

# predictions is a list of 1d numpy arrays
# event_indices is an array of event indices
# final_labels is an array of final labels
def series_dataset_scores(predictions, event_indices, final_labels, threshold):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    advance_magnitudes = []
    lagged_magnitudes = []

    predicted_indices = []

    for i, preds in enumerate(predictions):
        predicted_index = get_predicted_index(preds, threshold)
        predicted_indices.append(predicted_index)
        true_index = event_indices[i]
        true_label = final_labels[i]

        predicted_positive = (predicted_index >= 0)
        actual_positive = (true_label >= 0.5)

        true_positives += (predicted_positive and actual_positive)
        false_positives += (predicted_positive and not actual_positive)
        true_negatives += (not predicted_positive and not actual_positive)
        false_negatives += (not predicted_positive and actual_positive)

        is_true_positive = (predicted_positive and actual_positive)
        is_advance = (predicted_index <= true_index)
        magnitude = abs(true_index-predicted_index)

        if is_true_positive:
            if is_advance:
                advance_magnitudes.append(magnitude)
            else:
                lagged_magnitudes.append(magnitude)


    accuracy_denom = (true_positives + false_positives + true_negatives + false_negatives)
    precision_denom = (true_positives + false_positives)
    recall_denom = (true_positives + false_negatives)

    accuracy = None if (accuracy_denom==0) else (true_positives + true_negatives) / accuracy_denom
    precision = None if (precision_denom==0) else true_positives / precision_denom
    recall = None if (recall_denom==0) else true_positives / recall_denom

    advance_magnitudes = np.array(advance_magnitudes)
    lagged_magnitudes = np.array(lagged_magnitudes)
    predicted_indices = np.array(predicted_indices)
    all_magnitudes = np.concatenate((advance_magnitudes, lagged_magnitudes))    
    all_signed = np.concatenate((-advance_magnitudes, lagged_magnitudes))

    proportion_advanced = None if (true_positives == 0) else len(advance_magnitudes)/true_positives

    all_stats = Namespace(
            predicted_indices = predicted_indices,

            accuracy = accuracy,
            precision = precision,
            recall = recall,
            tp = true_positives,
            fp = false_positives,
            tn = true_negatives,
            fn = false_negatives,

            num_adv = len(advance_magnitudes),
            prop_adv = proportion_advanced,

            avg_mag_adv = None,
            std_mag_adv = None,
            mean_abs_dev_adv = None,
            min_adv = None,
            q1_adv = None,
            med_adv = None,
            q2_adv = None,
            max_adv = None,

            avg_mag_lag = None,
            std_mag_lag = None,
            mean_abs_dev_lag = None,
            min_lag = None,
            q1_lag = None,
            med_lag = None,
            q2_lag = None,
            max_lag = None,

            avg_mag_all = None,
            std_mag_all = None,
            mean_abs_dev_all = None,
            min_all = None,
            q1_all = None,
            med_all = None,
            q2_all = None,
            max_all = None,

            avg_mag_all_sign = None,
            std_mag_all_sign = None,
            mean_abs_dev_all_sign = None,
            min_all_sign = None,
            q1_all_sign = None,
            med_all_sign = None,
            q2_all_sign = None,
            max_all_sign = None,
            )

        
    all_stats.avg_mag_adv, all_stats.std_mag_adv, all_stats.mean_abs_dev_adv, all_stats.min_adv, all_stats.q1_adv, all_stats.med_avd, all_stats.q2_adv, all_stats.max_adv = array_stats(advance_magnitudes)

    all_stats.avg_mag_lag, all_stats.std_mag_lag, all_stats.mean_abs_dev_lag, all_stats.min_lag, all_stats.q1_lag, all_stats.med_lag, all_stats.q2_lag, all_stats.max_lag = array_stats(lagged_magnitudes)

    all_stats.avg_mag_all, all_stats.std_mag_all, all_stats.mean_abs_dev_all, all_stats.min_all, all_stats.q1_all, all_stats.med_all, all_stats.q2_all, all_stats.max_all = array_stats(all_magnitudes)

    all_stats.avg_mag_all, all_stats.std_mag_all_sign, all_stats.mean_abs_dev_all_sign, all_stats.min_all_sign, all_stats.q1_all_sign, all_stats.med_all_sign, all_stats.q2_all_sign, all_stats.max_all_sign = array_stats(all_signed)

    return all_stats
    

def sweep_decision_boundary(predictions, event_indices, final_labels, step_size = 0.01):
    all_predictions = np.concatenate(predictions)
    min_pred = np.min(all_predictions)
    max_pred = np.max(all_predictions)
    num_points = ((max_pred-min_pred)/step_size) + 1

    thresholds = np.linspace(max(min_pred,-1), min(max_pred,2), num_points) #So we don't get too absurd of results
    
    max_precision = -1
    max_precision_threshold = None
    max_precision_stats = None

    max_recall = -1
    max_recall_threshold = None
    max_recall_stats = None

    max_acc = -1
    max_acc_threshold = None
    max_acc_stats = None

    for th in thresholds:
        stats = series_dataset_scores(predictions, event_indices, final_labels, th)

        if stats.precision is not None and stats.precision > max_precision:
            max_precision_threshold = th
            max_precision_stats = stats

        if stats.recall is not None and stats.recall > max_recall:
            max_recall_threshold = th
            max_recall_stats = stats
            
        if stats.acc is not None and stats.acc > max_acc:
            max_acc_threshold = th
            max_acc_stats = stats

    
    all_results = Namespace(
            min_prediction = min_pred,
            max_prediction = max_pred,

            max_precision = max_precision,
            max_precision_threshold = max_precision_threshold,
            max_precision_stats = max_precision_stats,

            max_recall = max_recall,
            max_recall_threshold = max_recall_threshold,
            max_recall_stats = max_recall_stats,

            max_acc = max_acc,
            max_acc_threshold = max_acc_threshold,
            max_acc_stats = max_acc_stats
            )

    return all_results


def time_average(predictions, method='half_gaussian'):
    pass



def plot_all(folder, name_prefixes, predictions, event_indices, final_labels, index_time_conversion=0.1, step_size=0.01):

    results = sweep_decision_boundary(predictions, event_indices, final_labels, step_size=step_size)

    threshold = results.max_acc_threshold
    acc_stats = results.max_acc_stats
    

    for i,preds in enumerate(predictions):
        plt.clf()

        plt.ylim(max(stats.min_prediction, -2), min(stats.max_prediction,2))
        plt.title(name_prefixes[i]+'_label_'+str(final_labels[i]))
        plt.xlabel('Time')
        plt.ylabel('Predicted value')
        plt.plot(np.arange(len(preds))*index_time_conversion, preds, label='Predicted state values')
        plt.axhline(y=threshold, label='Decision threshold')

        if predicted_indices[i] > 0:
            plt.axvline(x=acc_stats.predicted_indices[i]*index_time_conversion, label='Event first predicted')

        if event_indices[i] > 0:
            plt.axvline(x=event_indices[i]*index_time_conversion, label='Event time')
        plt.legend()

        plt.savefig(os.path.join(folder, name_prefixes[i]+'.png'))


