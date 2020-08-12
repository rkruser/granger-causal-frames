import os
import pickle
from utils import *
from config import modeldir
from utils import best_step_fit
import numpy as np

import matplotlib.pyplot as plt



def joinstrings(*args, sep='_'):
    result = ''
    if len(args)==0:
        return result
    for s in args[:-1]:
        result += s+sep
    result += args[-1]
    return result

def load_model_data(path):
    with open(os.path.join(path,'config_info.pkl'),'rb') as f:
        configinfo = pickle.load(f)
    checkpoint_dict = {}
    for thing in os.listdir(path):
        thingpath = os.path.join(path,thing)
        if os.path.isdir(thingpath):
            splt = thing.split('_')
            if splt[0] == 'checkpoint':
                num = int(splt[1])
                metrics_path = os.path.join(thingpath,'metrics.pkl')
                video_predictions_path = os.path.join(thingpath, 'video_predictions.pkl')
                
                with open(metrics_path,'rb') as f:
                    metrics_obj = pickle.load(f)
                with open(video_predictions_path, 'rb') as f:
                    video_predictions_obj = pickle.load(f)
                checkpoint_dict[num] = (thingpath, metrics_obj, video_predictions_obj)

    return (configinfo, checkpoint_dict)


def plot_data(x, y, xtitle='', ytitle='', title='', vertical_lines=[], save_location=None):
    plt.plot(x,y)
    bottomy, topy = plt.gca().get_ylim()
    if len(vertical_lines) > 0:
        if len(vertical_lines) == 1:
            colors = ['b']
        else:
            colors = ['b', 'r']
        plt.vlines(vertical_lines, bottomy, topy, colors=colors)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.title(title)
    plt.savefig(save_location)
    plt.clf()

def graph_all_videos(models):
    for k in models:
        print("Graphing", k)
        
        checkpoints = models[k][1][1]
        for num in checkpoints:
            if num != 99:
                continue #Takes too long to do all
            print("...for checkpoint", num)
            checkpoint_path, metrics, video_predictions = checkpoints[num]
            train_path = os.path.join(checkpoint_path, 'train')
            test_path = os.path.join(checkpoint_path, 'test')
            if not os.path.isdir(train_path):
                os.makedirs(train_path)
            if not os.path.isdir(test_path):
                os.makedirs(test_path)

            for keys in [('train',train_path), ('test',test_path)]:
                key1, key2 = keys
                print('......'+key1)
                for vidname in video_predictions[key1]:
                    print('.........'+vidname)
                    savename = os.path.join(key2, vidname+'.png')
                    vid_data = video_predictions[key1][vidname]

                    numerical_predictions = vid_data[0]


                    label_info = vid_data[1]
                    playback_info = vid_data[2]
                    crash_predicted = vid_data[3]

                    # Apparently the given predictions are not right, so here are some adjustments
                    if crash_predicted:
                        re_predicted_crashindex = best_step_fit(np.array(numerical_predictions))
                        re_predicted_crashtime = label_info.starttime+playback_info['adjusted_frame_duration']*re_predicted_crashindex


                    time_predicted = vid_data[4]
                    if label_info.crash and crash_predicted:
                        #vertical_lines = [label_info.crashtime, time_predicted]
                        vertical_lines = [label_info.crashtime, re_predicted_crashtime]
                    else:
                        vertical_lines = []
                    name_append = ', actual_crash: '+str(label_info.crash)+', predicted_crash: '+str(crash_predicted)

                    times = np.arange(len(numerical_predictions))*playback_info['adjusted_frame_duration']+label_info.starttime
                    plot_data(times, numerical_predictions, xtitle='Video time (s)', ytitle='Crash potential', title=vidname+name_append, save_location=savename, vertical_lines=vertical_lines)

#                break
#            break
#        break


def print_statistics(models):
    for k in models:
        print("Printing for", k)
        
        checkpoints = models[k][1][1]
        for num in checkpoints:
            if num < 80:
                continue #Takes too long to do all
            print("...for checkpoint", num)
            checkpoint_path, metrics, video_predictions = checkpoints[num]

            train_summary = metrics['summarized_train_video_stats'][-1] #Each is a growing list
            test_summary = metrics['summarized_test_video_stats'][-1]
            print('......Train accuracy', train_summary[1])
#            print('......Train loss', train_summary[0])
#            print('......Train diffs', train_summary[3:])
            print('......Test accuracy', test_summary[1])
#            print('......Test loss', test_summary[0])
#            print('......Test diffs', test_summary[3:])





def main():
    models = {}
    for item in os.listdir(modeldir):
        itempath = os.path.join(modeldir, item)
        if os.path.isdir(itempath):
            splititem = item.split('_')
            timestr = splititem[-1]
            itemname = joinstrings(*splititem[:-1]) # Remove time string
            models[itemname] = (itempath, load_model_data(itempath)) # Maybe re-add timestr later
#            break #for now
    
    allowed_models = ['model_6', 'model_7', 'model_8', 'model_9', 'model_10', 'model_11', 'q_predictor50', 'q_predictor101']
    models = {k:models[k] for k in allowed_models}
#    graph_all_videos(models)
    print_statistics(models)

#    print(models.keys())
#    for k in models:
#        print(models[k][1][1][99][0].keys())
#        print(models[k][1][1][99][0]['summarized_train_video_stats'])
#        print(models[k][1][1][99][1].keys())
#        print(models[k][1][1][99][1]['train'].keys())

        # models[model_name][1][1][checkpoint_num][0=metrics,1=preds][metrickeys or 'train'|'test'][...]


#        break #for now
    


if __name__ == '__main__':
    main()
