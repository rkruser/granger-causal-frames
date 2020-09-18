import decord
import torch
import numpy as np
import pickle
import os
import cv2
#from config import datadir as data_directory #, trainvids, testvids
from config import get_config
#from video_loader import play_video

# Load videos in parallel if possible
from multiprocessing import Pool
import time


#label_file = './annotation/full_annotation.txt'
#split_file = './annotation/traintest_split.pkl'
#from config import label_file, split_file

decord.bridge.set_bridge('torch') 

def split_num(num, parts):
    div = num // parts
    rem = num % parts
    split = np.empty(parts,dtype='int64')
    split.fill(div)
    split[:rem] += 1
    return split

# Given size of batch, allocate items from batch according to counts
def split_batch_between(size, counts):
    counts = np.array(counts)
    total = np.sum(counts)
    if size >= total: # Last batch
        return counts

    split = split_num(size, len(counts))
    split = np.minimum(split, counts)
    split_total = np.sum(split)
    diffs = counts-split
    gap = size - split_total
    while gap > 0:
        mask = diffs>0
        newsplit = split_num(gap, np.sum(mask))
        split[mask] += newsplit
        split = np.minimum(split, counts)
        split_total = np.sum(split)
        diffs = counts-split
        gap = size-split_total
    return split
    
# Assume patch has already been frame subsampled and scaled to correct size
def process_patch(arr, begin_point, end_point, frames_per_point, use_transitions=True):
    begin_ind = begin_point
    end_ind = end_point+frames_per_point-1
    terminal = False
    if use_transitions:
        end_ind += 1
    if end_ind == len(arr)+1:
        end_ind = len(arr)
        terminal = True

    patch = arr[begin_ind:end_ind]

    # Get sets of frames
    size = len(patch)
    cur_array = [ patch[i:(size-frames_per_point+i+1)] for i in range(frames_per_point) ]
    cur_array = np.concatenate(cur_array, axis=1)
    size = len(cur_array)

    if use_transitions:
        result_array = np.stack([cur_array[:(size-1)], cur_array[1:]])
        if terminal:
            result_array = np.concatenate([result_array, np.stack([np.expand_dims(cur_array[-1],axis=0), 
                                                            np.zeros((1,)+cur_array[-1].shape)])], axis=1)

            # final shape: (2, #points, 3*frames_per_point, rows, cols)
    else:
        result_array = cur_array

    return result_array, terminal

def rl_label_func(label, begin, end, length):
    l = np.zeros(end-begin)
    if end == length:
        l[-1] = label
    return l

def regular_label_func(label, begin, end, length):
    l = np.empty(end-begin)
    l.fill(label)
    return l

class VideoFrameLoader:
    # randomization_level = decord's shuffle
    # inter_batch_skip = decord's skip
    # intra_batch_skip = decord's interval
    # post_transform : either None, or a function that takes an image batch in numpy form and transforms it
    # frames_per_point : number of past frames per point to use
    # overlap_points : overlap points with multiple frames or not
    # return_transitions : obtain video transitions
    def __init__(self, vid_list, 
                       label_list, #list of vid labels (crash or no crash)
                       batch_size=64, 
                       image_shape=(224,224,3), 
#                       post_transform = None,
                       shuffle_files=True,
                       preload_num=16,
#                       randomization_level=1,
#                       inter_batch_skip=0,
#                       intra_batch_skip=0,
                       frame_interval=5,
                       frames_per_point=3,
                       overlap_points=True,
                       return_transitions=True,
                       parallel_processes=4, # Seems pretty good on my desktop
                       randomize_start_frame=True
                       ):


        self.shuffle_files = shuffle_files
        self.file_shuffle = np.arange(len(vid_list))
        self.file_list = vid_list
        self.label_list = label_list
        self.image_shape = image_shape
        self.frame_interval = frame_interval
        self.batch_size = batch_size
        self.parallel_processes = parallel_processes
        self.randomize_start_frame = randomize_start_frame

#        assert(randomization_level == 0 or randomization_level == 1)
#        self.loader = decord.VideoLoader(vid_list, 
#                                         ctx=decord.cpu(), 
#                                         shape=(batch_size,)+image_shape, 
#                                         interval=intra_batch_skip,
#                                         skip=inter_batch_skip,
#                                         shuffle=randomization_level)
#        self.post_transform = post_transform
        self.overlap_points = overlap_points
        self.frames_per_point = frames_per_point
        self.return_transitions = return_transitions

        if self.return_transitions:
            self.label_func = rl_label_func
        else:
            self.label_func = regular_label_func

        self._preload_num = preload_num

        self._reset()


    def num_videos(self):
        return len(self.file_list)

    def get_video(self, i):
        reader = decord.VideoReader(self.file_list[i], ctx=decord.cpu(0), width=self.image_shape[0], height=self.image_shape[1])
        n_frames = len(reader)
        frames = reader.get_batch(np.arange(0,n_frames,self.frame_interval))
        frames = frames.permute(0,3,1,2)
        
        frames, _ = process_patch(frames.numpy(), 0, len(frames), self.frames_per_point, use_transitions=False)
        frames = torch.from_numpy(frames) # Do any reshaping?
        frames = frames.float()
        frames = frames/255.0
        return frames, self.label_list[i]


    def _reset(self):
        if self.shuffle_files:
            self.file_shuffle = np.random.permutation(self.file_shuffle)
        
        self._file_pointer = 0
        self._preloaded = []
        self._preloaded_indices = []
        self._preloaded_labels = []
        self._preloaded_lengths = np.array([])
        self._preloaded_counters = np.array([])
#        self._length = len(self.loader)
#        self._loader_iter = None
#        self._current_index = 0
#        self._current_batch = None



    # Preload several videos for training
    def _preload_next(self):
        print("Preloading", self.file_shuffle[self._file_pointer:self._file_pointer+self._preload_num])
        if self._file_pointer == len(self.file_list):
            self._reset()
            raise StopIteration

        self._preloaded = []
        next_pointer = min(self._file_pointer + self._preload_num, len(self.file_list))
        self._preloaded_indices = self.file_shuffle[self._file_pointer:next_pointer]
        self._preloaded_labels = [self.label_list[i] for i in self._preloaded_indices]

        while self._file_pointer < next_pointer:
            index = self.file_shuffle[self._file_pointer]
            reader = decord.VideoReader(self.file_list[index], ctx=decord.cpu(0), width=self.image_shape[0], height=self.image_shape[1])
            n_frames = len(reader)

            # Randomly choose an initial start point!
        # Randomly choose an initial start point!
            if self.randomize_start_frame:
                startframe = np.random.randint(self.frame_interval) #doesn't matter super much
            else:
                startframe = 0

            frames = reader.get_batch(np.arange(startframe, n_frames, self.frame_interval)) # randomly change start frame to increase effective data
            frames = frames.permute(0,3,1,2)
            self._preloaded.append(frames.numpy())
            self._file_pointer += 1

        self._preloaded_lengths = np.array([len(arr)-self.frames_per_point+1 for arr in self._preloaded],dtype=int)
        self._preloaded_counters = np.zeros(len(self._preloaded),dtype=int)


    def _load_single_video(self, video_file_name):
        reader = decord.VideoReader(video_file_name, ctx=decord.cpu(0), width=self.image_shape[0], height=self.image_shape[1])
        n_frames = len(reader)

        # Randomly choose an initial start point!
        if self.randomize_start_frame:
            startframe = np.random.randint(self.frame_interval) #doesn't matter super much
        else:
            startframe = 0

        frames = reader.get_batch(np.arange(startframe, n_frames, self.frame_interval)) # randomly change start frame to increase effective data
        frames = frames.permute(0,3,1,2)
        return frames.numpy()


    def _preload_next_parallel(self):
        print("Preloading", self.file_shuffle[self._file_pointer:self._file_pointer+self._preload_num])
        if self._file_pointer == len(self.file_list):
            self._reset()
            raise StopIteration

        self._preloaded = []
        next_pointer = min(self._file_pointer + self._preload_num, len(self.file_list))
        self._preloaded_indices = self.file_shuffle[self._file_pointer:next_pointer]
        preloaded_names = [self.file_list[ind] for ind in self._preloaded_indices]
        self._preloaded_labels = [self.label_list[i] for i in self._preloaded_indices]

        with Pool(processes=self.parallel_processes) as pool:
            self._preloaded = pool.map(self._load_single_video, preloaded_names)
            self._file_pointer = next_pointer

        self._preloaded_lengths = np.array([len(arr)-self.frames_per_point+1 for arr in self._preloaded],dtype=int)
        self._preloaded_counters = np.zeros(len(self._preloaded),dtype=int)



    def _next_batch(self):
        remaining = self._preloaded_lengths - self._preloaded_counters
        if np.all(remaining == 0):
#            time1 = time.time()

            if self.parallel_processes > 1:
#                print("Parallel preload")
                t1=time.time()
                self._preload_next_parallel()
                print("Preload time", time.time()-t1)
            else:
#                print("Normal preload")
                t1 = time.time()
                self._preload_next()
                print("Preload time", time.time()-t1)

#            time2 = time.time()
#            print("Preload time:", time2-time1)
            remaining = self._preloaded_lengths - self._preloaded_counters

        mask = remaining>0
        split = np.zeros(len(remaining),dtype=int)
        split[mask] += split_batch_between(self.batch_size, remaining[mask])

        patches = []
        labels = []
        terminals = []
        vid_inds = [] # [(vid_id, begin, end)]
        for i, arr in enumerate(self._preloaded):
            if split[i] == 0:
                continue
            begin = self._preloaded_counters[i]
            end = begin+split[i]
            length = self._preloaded_lengths[i]
            vid_inds.append((self._preloaded_indices[i], begin, end, length))
            label = self._preloaded_labels[i]
            arr_labels = self.label_func(label, begin, end, length) #Need to implement label_func
            process_arr, is_terminal = process_patch(arr, begin, end, self.frames_per_point, self.return_transitions)
            terminal_array = np.zeros(end-begin,dtype=bool)
            terminal_array[-1] = is_terminal
            terminals.append(terminal_array)
            patches.append( process_arr )
            labels.append(arr_labels)

        cat_axis = 1 if self.return_transitions else 0
        batch = np.concatenate(patches, axis=cat_axis) # Axis 1 for transitions, 0 otherwise
        labels = np.concatenate(labels,axis=0)
        terminals = np.concatenate(terminals,axis=0)
        self._preloaded_counters += split

        return batch, labels, terminals, vid_inds


    def __iter__(self):
        self._reset()  # This probably works
        return self

    def __next__(self):
        batch, labels, terminals, vid_inds = self._next_batch()
        batch_torch = torch.from_numpy(batch)
        batch_torch = batch_torch.float()
        batch_torch = batch_torch/255.0
        
        return batch_torch, torch.from_numpy(labels), torch.BoolTensor(terminals), torch.Tensor(vid_inds)
        # Question! Does decord return 3xhxw or hxwx3?
        # Does it normalize the values to 0,1? Or have right data type?



def get_label_data(data_directory, label_file, split_file=None):
#    with open(os.path.join(data_directory, label_file), 'r') as labfile:
    with open(os.path.join(label_file), 'r') as labfile:

        lines = labfile.readlines()
        split = [l.split() for l in lines]

        annotations = [ (os.path.join(data_directory,l[0].lstrip('./')), 
                        float(l[1]), 
                        1 if float(l[1])>=0 else 0) for l in split ]
        fullpaths, times, labels = zip(*annotations)

        if split_file:
#            train_inds, test_inds = pickle.load(open(os.path.join(data_directory,split_file),'rb'))
            train_inds, test_inds = pickle.load(open(os.path.join(split_file),'rb'))
       
            fullpaths, times, labels = np.array(fullpaths), np.array(times), np.array(labels)
            train_fullpaths, train_times, train_labels = fullpaths[train_inds], times[train_inds], labels[train_inds]
            test_fullpaths, test_times, test_labels = fullpaths[test_inds], times[test_inds], labels[test_inds]

            return (train_fullpaths, train_times, train_labels), (test_fullpaths, test_times, test_labels)
        else:
            return fullpaths, times, labels



def test_loader():
    cfg = get_config()
    fullpaths, times, labels = get_label_data(cfg.data_directory, cfg.label_file, cfg.split_file)

    vidloader = VideoFrameLoader(fullpaths,labels,
                                preload_num=10,
                                shuffle_files=True, 
                                batch_size=256, 
                                frame_interval=5,
                                return_transitions=True,
                                parallel_processes=4)

    for i, k in enumerate(vidloader):
        batch, labels, terminals, vid_inds = k
        print("Batch", i)
        print(batch.size())
#        print(batch.dtype)
#        print(batch[0,0,0,0,:3])
#        print(batch[0,0,0,:3])

#        print(labels)
#        print(terminals)
#        print(vid_inds)


def test2():
    cfg = get_config()
    fullpaths, times, labels = get_label_data(cfg.data_directory, cfg.label_file, cfg.split_file)
    vidloader = VideoFrameLoader(fullpaths[:5],labels[:5],
                                preload_num=2,
                                shuffle_files=True, 
                                batch_size=64, 
                                frame_interval=3,
                                return_transitions=True,
#                                parallel_processes=1
                                )


    for i in range(vidloader.num_videos()):
        vid,label = vidloader.get_video(i)
        print(vid.shape, vid.dtype, label)


if __name__=='__main__':
    test_loader()
#    test2()































