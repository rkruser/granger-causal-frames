import decord
import torch
import numpy as np
import os
import cv2
from config import datadir, trainvids, testvids
from video_loader import play_video


train_fullpaths = [os.path.join(datadir, vid) for vid in trainvids[:10]]

decord.bridge.set_bridge('torch')

# Maybe use every 3rd frame instead of every 5th

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
    
def process_patch(path, frames_per_point, use_transitions=True):
    pass


class VideoFrameLoader:
    # randomization_level = decord's shuffle
    # inter_batch_skip = decord's skip
    # intra_batch_skip = decord's interval
    # post_transform : either None, or a function that takes an image batch in numpy form and transforms it
    # frames_per_point : number of past frames per point to use
    # overlap_points : overlap points with multiple frames or not
    # return_transitions : obtain video transitions
    def __init__(self, vid_list, 
                       batch_size=64, 
                       image_shape=(224,224,3), 
                       post_transform = None,
                       randomization_level=1,
#                       inter_batch_skip=0,
#                       intra_batch_skip=0,
                       frame_interval=5,
                       frames_per_point=3,
                       overlap_points=True,
                       return_transitions=True
                       ):

        self.file_shuffle = np.random.permutation(np.arange(len(vid_list)))
        self.file_list = vid_list
        self.image_shape = image_shape
        self.frame_interval = frame_interval
        self.batch_size = batch_size

#        assert(randomization_level == 0 or randomization_level == 1)
#        self.loader = decord.VideoLoader(vid_list, 
#                                         ctx=decord.cpu(), 
#                                         shape=(batch_size,)+image_shape, 
#                                         interval=intra_batch_skip,
#                                         skip=inter_batch_skip,
#                                         shuffle=randomization_level)
        self.post_transform = post_transform
        self.overlap_points = overlap_points
        self.frames_per_point = frames_per_point
        self.return_transitions = return_transitions


        self._preloaded = []
        self._preloaded_lengths = np.array([])
        self._preloaded_counters = np.array([])
        self._file_pointer = 0
        self._preload_num = 8
        self._length = len(self.loader)
        self._loader_iter = None
        self._current_index = 0
        self._current_batch = None
        self._next_batch = None


    # Preload several videos for training
    def _preload_next(self):
        print("Preloading", self.file_shuffle[self._file_pointer:self._file_pointer+self._preload_num])
        self._preloaded = []
        next_pointer = min(self._file_pointer + self._preload_num, len(self.file_list))
        while self._file_pointer < next_pointer:
            index = self.file_shuffle[self._file_pointer]
            reader = decord.VideoReader(self.file_list[index], ctx=cpu(0), width=self.image_shape[0], height=self.image_shape[1])
            n_frames = len(reader)
            frames = reader.get_batch(np.arange(0, n_frames, self.frame_interval))
            self._preloaded.append(frames.numpy())
            self._file_pointer += 1

        self._preloaded_lengths = np.array([len(i)-self.frames_per_point+1 for i in self._preloaded])
        self._preloaded_counters = np.zeros(len(self._preloaded))


    def _next_batch(self):
        remaining = self._preloaded_lengths - self._preloaded_counters
        if np.all(remaining == 0):
            self._preload_next()
            remaining = self._preloaded_lengths - self._preloaded_counters

        mask = remaining>0
        split = np.zeros(len(remaining))
        split[mask] += split_batch_between(self.batch_size, remaining[mask])

        patches = []
        for i, arr in enumerate(self._preloaded):
            if split[i] == 0:
                continue
            begin = self._preloaded_counters[i]
            end = begin+split[i]
            patches.append( process_patch(arr[begin:end+self.frames_per_point-1], self.frames_per_point) )

        batch = np.concatenate(patches, axis=1) # Axis 1 for transitions, 0 otherwise
        self._preloaded_counters += split

        return batch


    def __iter__(self):
        pass

    def __next__(self):
        pass
