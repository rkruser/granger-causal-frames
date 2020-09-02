import torch 
import pickle
import os
import numpy as np
import argparse
import torch.nn as nn

from utils import AverageMeter, MeterBox, MetricBox, FolderTracker, VideoPredictions
from utils import best_step_fit
from flexible_resnet import resnet50_flexible, resnet101_flexible, resnet18_flexible
#from video_loader import BeamNG_FileTracker, VideoDataset, label_func_1, label_func_prediction_only frame_transform_1
from config import get_config #, trainvids, testvids

# Add better_decord_loader imports
from decord_loader import VideoFrameLoader, get_label_data

import time


# Later make this more flexible class Model:
class Model:
    def __init__(self, cfg):
        self.cfg = cfg

        if cfg.use_color:
            ncolors = 3
        else:
            ncolors = 1

        if cfg.network_type == 'resnet50':
            self.network = resnet50_flexible(num_classes=1, data_channels=cfg.frames_per_datapoint*ncolors)
        elif cfg.network_type == 'resnet101':
            self.network = resnet101_flexible(num_classes=1, data_channels=cfg.frames_per_datapoint*ncolors)
        elif cfg.network_type == 'resnet18':
            self.network = resnet18_flexible(num_classes=1, data_channels=cfg.frames_per_datapoint*ncolors)
        else:
            raise ValueError("Unknown network_type option")

        self.device = cfg.device
        if cfg.use_data_parallel:
            self.network = nn.DataParallel(self.network)
        self.network = self.network.to(self.device)

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=cfg.learning_rate,
                                          betas=(cfg.beta1, cfg.beta2),
                                          eps=cfg.eps, weight_decay=cfg.weight_decay,
                                          amsgrad=cfg.amsgrad)

        if cfg.use_q_loss:
            self.update_func = self.q_update
        else:
            self.update_func = self.prob_update
       

    def q_loss(self, q_current, q_future, weights):
        diff = q_future-q_current
        loss = (weights*(diff**2)).mean()
        return loss
        
    def q_update(self, batch):
        x, y, is_terminal, _ = batch
        x = x.to(self.device)
        y = y.to(self.device)
        is_terminal = is_terminal.to(self.device)

        x_current = x[0]
        x_future = x[1]

        weights = torch.ones(len(x_current)).to(self.device)
        with torch.no_grad():
            q_future = self.cfg.rl_gamma * self.network(x_future).squeeze(1)
#            inds = torch.abs(y_future+2)<0.0001 # replaced by is_terminal
            q_future[is_terminal] = 0
            weights[is_terminal] = self.cfg.terminal_weight
            q_future = q_future+y

        q_current = self.network(x_current).squeeze(1)

        loss = self.q_loss(q_current, q_future, weights)

        self.network.zero_grad()
        loss.backward()
        self.optimizer.step()
        return q_current, loss.item()
       
    def prob_loss(self, predictions, actual):
        return torch.nn.functional.binary_cross_entropy_with_logits(predictions, actual)

    def prob_update(self, batch):
        x, y, _, _ = batch
        x = x.to(self.device)
        y = y.to(self.device)

        predictions = self.network(x).squeeze(1)
        loss = self.prob_loss(predictions, y)
        self.network.zero_grad()
        loss.backward()
        self.optimizer.step()
        return predictions, loss.item()

    def update(self, batch):
#        x, y = x.to(self.device), y.to(self.device)
        return self.update_func(batch)

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()

    # Video is tensor with entire video, no transitions
    def predict_video(self, video, y_vals=None):
        video = video.to(self.device)
        input_size = self.cfg.frames_per_datapoint

        predictions = []
        with torch.no_grad():
            if self.cfg.overlap_datapoints:
                for chunk in torch.chunk(video, (len(video)//self.cfg.batch_size)+1):
                    prediction = self.network(chunk)
                    predictions.append(prediction.squeeze(1))
#@                for i in range(len(video)-input_size+1):
#@                    prediction = self.network(video[i:i+input_size].unsqueeze(0))
#@                    predictions.append(prediction.item())
            else:
                pass

        predictions = torch.cat(predictions) # (len(video)-window_size+1,)
        if not self.cfg.use_q_loss:
            predictions = torch.sigmoid(predictions)
#        predictions = torch.Tensor(predictions)
        loss = None
        if y_vals is not None:
            if self.cfg.use_q_loss:
                weights = torch.ones(len(predictions))
                weights[-1] = self.cfg.terminal_weight
                q_future = torch.zeros(len(predictions))
                q_future += y_vals
                q_future[:-1] += predictions[1:]
                q_future = q_future
                loss = self.q_loss(predictions, q_future, weights).item()
            else:
                loss = -10

        return predictions, loss

    def filter_predictions(self, 
                            time_series, 
                            video_length, 
                            adjusted_frame_duration, #after accounting for sample rate
                            start_time): #crash_time
#        time_series = time_series.numpy()
        crash_position = best_step_fit(time_series)

#        adjusted_frame_duration = sample_every*orig_fm_duration
        future_adjustment = video_length-len(time_series)+1
        predicted_time = adjusted_frame_duration*(crash_position+future_adjustment-1)+start_time

        is_crash = (crash_position >= 0)
#        if is_crash:
#            difference = predicted_time-crash_time
#        else:
#            difference = None

        return is_crash, predicted_time
        
        

    # Change this later to a better save function?
    def save(self, path):
        with open(os.path.join(path,'model.th'), 'wb') as f:
            #pickle.dump(self, f)
            torch.save({'model_state_dict':self.network.state_dict(), 'optimizer_state_dict':self.optimizer.state_dict(), 'config':self.cfg}, f)

    def load_from(self, model_save_path):
        with open(model_save_path,'rb') as f:
            print("Loading model from", model_save_path)
            print("Loading onto", self.device)
            params = torch.load(f, map_location=self.device)
            self.network.load_state_dict(params['model_state_dict'])
            self.optimizer.load_state_dict(params['optimizer_state_dict'])



def compute_summary(video_stats):
    losses, predictions_correct, predictions, actual, diffs = zip(*video_stats)
    losses, predictions_correct, predictions, actual, diffs = np.array(losses), np.array(predictions_correct), \
                                                              np.array(predictions), np.array(actual), \
                                                              np.array(diffs)
    # (avg loss, crash existence accuracy, num_correct_crash_preds, mean diff from crashtime, std of diffs, mean abs diff, std abs diff)
    diffs = diffs[predictions&actual] #Only count the errors on crash predictions that were true

    # Quick hack to prevent exceptions
    diff_length = len(diffs)
    if diff_length == 0:
        diffs = np.array([-100])

    return (losses.mean(), predictions_correct.mean(), diff_length, diffs.mean(), diffs.std(), np.abs(diffs).mean(), np.abs(diffs).std())


def train_and_test(model, train_dataset, test_dataset, cfg, args):
    # Set model to training mode
    model.train() 

    # Names of things to track about the results
#    tracker_names = ['all_train_batch_loss', 
#                     'summarized_train_batch_loss',
#
#                     'epoch_train_video_stats',
#                     'summarized_train_video_stats',
#
#                    'epoch_test_video_stats',
#                    'summarized_test_video_stats',
#                    ]

    # Meters are real-time scores extracted
    meters = MeterBox(train_batch_loss=AverageMeter())
    # Metrics are processed scores
#    metrics = MetricBox(*tracker_names) 
    folders = FolderTracker(root=cfg.model_directory, name=cfg.model_name)

    # Save the config parameters for later reference
    with open(os.path.join(folders.folder(),'config_info.pkl'),'wb') as f:
        pickle.dump((cfg, args), f) # Save config for parsing

    # Training loop
    for epoch in range(cfg.n_epochs):
        print("Epoch", epoch)
        meters.train_batch_loss.reset()

        # Iterate over dataset
        time1 = time.time()
        for i, batch in enumerate(train_dataset):
            _, loss = model.update(batch) #Do backprop and optimization step
            batch_size = len(batch[0])
            meters.train_batch_loss.update(batch_size*loss, batch_size)

            if i%10 == 0:
                print("  batch {0}, loss {1}".format(i,meters.train_batch_loss.average()))

            if i==30: # for testing
              break

#            metrics.all_train_batch_loss.append(loss)

        print("Epoch average loss", meters.train_batch_loss.average())

#        metrics.summarized_train_batch_loss.append(meters.train_batch_loss.average())
        
        time2 = time.time()
        print("Epoch time:", time2-time1)

        # Every few epochs, save a checkpoint and run model on all train and test data
        if ( (epoch+1) % cfg.checkpoint_every == 0 ) or ( (epoch+1)==cfg.n_epochs ):
            print("Checkpointing")
#            model.eval()
            path = folders.subfolder('checkpoint',epoch)
            if cfg.overwrite_last:
                model.save(folders.folder())
            else:
                model.save(path)
            
#            vp = VideoPredictions()
            
#            metrics.epoch_train_video_stats = []
#            metrics.epoch_test_video_stats = []
#
#            # Train video benchmarking
#            for video in train_dataset.video_iter():
#                print("Benchmarking", video.labelinfo.name)
#                if len(video) == 0:
#                    print("Skipping zero vid")
#                    continue
#                numerical_predictions, loss = model.predict_video(torch.from_numpy(video.array).float()/255.0, 
#                                                                    y_vals=torch.Tensor(video.labels))
#                if not cfg.use_q_loss:
#                    numerical_predictions = torch.sigmoid(numerical_predictions)
#                predicted_crash, predicted_time = model.filter_predictions(numerical_predictions.numpy(),
#                                                           video.playback_info['num_sampled_frames'],
#                                                           video.playback_info['adjusted_frame_duration'],
#                                                           video.labelinfo.starttime
#                                                           )
#
#                actual_crash, actual_time = video.labelinfo.crash, video.labelinfo.crashtime
#                vp.add_prediction('train',
#                                  video.labelinfo.name,
#                                  numerical_predictions,
#                                  video.labelinfo,
#                                  video.playback_info,
#                                  predicted_crash,
#                                  predicted_time
#                                  )
#                # Then update meters / metrics
#                metrics.epoch_train_video_stats.append((loss, predicted_crash==actual_crash, predicted_crash, actual_crash, predicted_time-actual_time))
#
#            metrics.summarized_train_video_stats.append(compute_summary(metrics.epoch_train_video_stats)) 
#
#            # Test video benchmarking
#            for video in test_dataset.video_iter():
#                print("Benchmarking", video.labelinfo.name)
#                if len(video) == 0:
#                    print("Skipping zero vid")
#                    continue
#                numerical_predictions, loss = model.predict_video(torch.from_numpy(video.array).float()/255.0, 
#                                                                  y_vals=torch.Tensor(video.labels))
#                if not cfg.use_q_loss:
#                    numerical_predictions = torch.sigmoid(numerical_predictions)
#
#                predicted_crash, predicted_time = model.filter_predictions(numerical_predictions.numpy(),
#                                                           video.playback_info['num_sampled_frames'],
#                                                           video.playback_info['adjusted_frame_duration'],
#                                                           video.labelinfo.starttime
#                                                           )
#                actual_crash, actual_time = video.labelinfo.crash, video.labelinfo.crashtime
#                vp.add_prediction('test',
#                                  video.labelinfo.name,
#                                  numerical_predictions,
#                                  video.labelinfo,
#                                  video.playback_info,
#                                  predicted_crash,
#                                  predicted_time
#                                  )
#
#                metrics.epoch_test_video_stats.append((loss, predicted_crash==actual_crash, predicted_crash, actual_crash, predicted_time-actual_time))
#
#
#            metrics.summarized_test_video_stats.append(compute_summary(metrics.epoch_test_video_stats)) 
#
#
#            metrics.save(path) 
#            vp.save(path)

#            model.train()



# Default is always alternate truncation
def construct_dataset_from_config(cfg, vidlist, label_list, shuffle_files=True):
    dataset = VideoFrameLoader(vidlist,
                               label_list,
                               batch_size = cfg.batch_size,
                               image_shape = cfg.image_shape,
                               shuffle_files = shuffle_files,
                               preload_num = cfg.preload_num,
                               frame_interval = cfg.frame_sample_freq,
                               frames_per_point = cfg.frames_per_datapoint,
                               overlap_points=cfg.overlap_datapoints,
                               return_transitions=cfg.use_transitions,
                               parallel_processes=cfg.num_data_workers,
                               randomize_start_frame=cfg.randomize_start_frame
                               )

    return dataset


def construct_model_from_config(cfg):
    return Model(cfg)   


def random_seed(seed):
    torch.manual_seed(seed) #I think this works for the gpus too?
    np.random.seed(seed+1)



def run_job(trainvids, trainlabels, testvids, testlabels, cfg_str = None, jobid=None):
    cfg, args = get_config(stringargs=cfg_str)
    if jobid is not None:
        cfg.model_name += '_'+str(jobid)
    if cfg.use_q_loss:
        cfg.use_transitions=True
    print(args)
    print(cfg)
    random_seed(cfg.random_seed)
    train_set = construct_dataset_from_config(cfg, trainvids, trainlabels, shuffle_files=True)
    test_set = construct_dataset_from_config(cfg, testvids, testlabels, shuffle_files=False)
    model = construct_model_from_config(cfg)
    train_and_test(model, train_set, test_set, cfg, args)


def train_standard(cfg_str = None, job_id=None):
    train, test = get_label_data()
    train_fullpaths, train_times, train_labels = train
    test_fullpaths, test_times, test_labels = test

    run_job(train_fullpaths, train_labels, test_fullpaths, test_labels, cfg_str=cfg_str, jobid=job_id)


def test_model(model_path, cfg, savename='results.pkl'):
    print("Testing")
    print(cfg)

    train, test = get_label_data()
    train_fullpaths, train_times, train_labels = train
    test_fullpaths, test_times, test_labels = test
    train_set = construct_dataset_from_config(cfg, train_fullpaths, train_labels, shuffle_files=False)
    test_set = construct_dataset_from_config(cfg, test_fullpaths, test_labels, shuffle_files=False)
    model = construct_model_from_config(cfg)
    model.load_from(model_path)
    model.eval()

    all_train_predictions = []
    for i in range(train_set.num_videos()):
        print("Train video", i)
        frames, label = train_set.get_video(i)
#        print("...label",label)
        predictions, _ = model.predict_video(frames)    
        all_train_predictions.append(predictions)
        

    all_train_predictions = np.array(all_train_predictions)


    all_test_predictions = []
    for i in range(test_set.num_videos()):
        print("Test video", i)
        frames, label = test_set.get_video(i)
#        print("...label",label)
        predictions, _ = model.predict_video(frames)    
        all_test_predictions.append(predictions)
        

    all_test_predictions = np.array(all_test_predictions)

    collected_results = {
        "train_fullpaths": train_fullpaths,
        "train_times": train_times,
        "train_labels": train_labels,
        "test_fullpaths": test_fullpaths,
        "test_times": test_times,
        "test_labels": test_labels,
        "train_predictions": all_train_predictions,
        "test_predictions": all_test_predictions,
        "config": cfg
    }

    with open(savename,'wb') as fobj:
        pickle.dump(collected_results, fobj)



if __name__ == '__main__':
    default_model_dir = '/mnt/linuxshared/phd-research/better_causalFrames/models/model_on_new_dataset_08-09-2020-05:48:37/'

#    parser = argparse.ArgumentParser()
#    parser.add_argument('--train', action='store_true')
#    parser.add_argument('--test', action='store_true')
#    parser.add_argument('--load_model_dir', type=str, default=default_model_dir)
#    parser.add_argument('--load_model_num', type=int, default=-1)
#    parser.add_argument('--test_results_savename', type=str, default='results.pkl')
#    opt, _ = parser.parse_known_args()
    
#    if opt.train:
#        cfg_str = '--model_name ryen_sanity_check_model'
#        train_standard(cfg_str)
    train_standard()

#    if opt.test:
#        print("in opt.test")
#        model_path = os.path.join(opt.load_model_dir, 'model.th')
#        config_path = os.path.join(opt.load_model_dir, 'config_info.pkl')
#        model_cfg, _ = pickle.load(open(config_path,'rb'))
#        update_cfg, _ = get_config(default_dict=model_cfg.__dict__)
#        test_model(model_path, update_cfg, savename=opt.test_results_savename)        



