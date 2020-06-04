import pickle
import os
import time
import numpy as np

class AverageMeter:
    def __init__(self, init_value=0, init_count=0):
        self.init_value = init_value #in case we init with a different type or something
        self.init_count = init_count
        self.value = init_value
        self.count = init_count
    def update(self, value, count):
        self.value += value
        self.count += count
    def average(self):
        return self.value / self.count
    def reset(self):
        self.value = self.init_value
        self.count = self.init_count

class MeterBox:
    def __init__(self, **kwargs):
        self.meterlist = kwargs.keys()
        for k,v in kwargs.items():
            setattr(self, k, v)
    def reset_all(self):
        for k in self.meterlist:
            getattr(self,k).reset()

class MetricBox:
    def __init__(self, *args):
        self.metriclist = args
        for a in args:
            setattr(self, a, [])
    def save(self, folder):
        data = {a:self.__dict__[a] for a in self.metriclist}
        with open(os.path.join(folder,'metrics.pkl'),'wb') as f:
            pickle.dump(data,f)
    def load(self, folder):
        with open(os.path.join(folder,'metrics.pkl'),'rb') as f:
            data = pickle.load(f)
            self.metriclist = data.keys()
            for k,v in data.items():
                setattr(self,k,v)

class VideoPredictions:
    def __init__(self):
        self.preds = {}
    def add_prediction(self, category, vidname, predictions, labelinfo, playbackinfo, predcrash, predtime):
        if category not in self.preds:
            self.preds[category] = {}
        self.preds[category][vidname] = (predictions, labelinfo, playbackinfo, predcrash, predtime)
    def save(self, path):
        with open(os.path.join(path,'video_predictions.pkl'),'wb') as f:
            pickle.dump(self.preds,f)
    def load(self, path):
         with open(os.path.join(path,'video_predictions.pkl'),'rb') as f:
            self.preds = pickle.load(f)
       


class FolderTracker:
    def __init__(self, root='.', name='model', use_timestamp=True):
        if use_timestamp:
            ts = time.gmtime()
            self.timestamp = time.strftime('_%m-%d-%Y-%H:%M:%S', ts)
        else:
            self.timestamp = ''

        self.basepath = os.path.join(root, name+self.timestamp)
        if not os.path.isdir(self.basepath):
            os.makedirs(self.basepath)

    def folder(self):
        return self.basepath

    def subfolder(self, name, number=None):
        if number is not None:
            name = name+'_'+str(number)
        path = os.path.join(self.basepath,name)
        if not os.path.isdir(path):
            os.makedirs(path)
        return path


############################################
    
# Fits a 0-1 step function to the series
# Problem: this presumes you know the future; can't really be used online 
# assumes series is numpy array
def best_step_fit(series):
    l = len(series)
    scores = np.zeros(l+1)
    for i in range(l+1):
        before = series[:i]
        after = 1-series[i:]
        scores[i] = np.sum(before*before) + np.sum(after*after)

    argmin = np.argmin(scores)
    if argmin >= l:
        return -1
    else:
        return argmin


#def calculate_stats(picklefile, seriesEval, sampleRate=10, overlap=True):
#    stats = pickle.load(open(picklefile,'rb'))
#    differentials = []
## (xvals, qvals, labelinfo, playbackinfo)
#    for vidname in stats:
#        xvals, qvals, labelinfo, playbackinfo = stats[vidname]
#        predicted_position = seriesEval(qvals)
#        if predicted_position < 0:
#            print("No crash predicted for", vidname)
#            continue
#        if overlap:
#            predicted_time = (predicted_position+sampleRate-1)/playbackinfo['fps']+labelinfo.starttime
#        else:
#            pass
#
#        difference = predicted_time-labelinfo.crashtime
#        print(vidname, difference)
#        differentials.append(difference)
#
#    differentials = np.array(differentials)
#    mean = differentials.mean()
#    std = differentials.std()
#
#    newFileName = os.path.splitext(picklefile)
#    newFileName = newFileName[0]+'_processed'+newFileName[1]
#
#    print("Mean time difference", mean, "Std time difference", std)
#
