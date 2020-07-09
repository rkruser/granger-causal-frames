import decord
import decord# import VideoLoader, VideoReader
#from decord# import cpu, gpu
import torch
import torchvision.transforms.functional as tfunc
import numpy as np

import os
import cv2
from config import datadir, trainvids, testvids
from video_loader import play_video

train_fullpaths = [os.path.join(datadir,vid) for vid in trainvids[:10]]

decord.bridge.set_bridge('torch')


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
                       inter_batch_skip=0,
                       intra_batch_skip=0,

                       frames_per_point=3,
                       overlap_points=True,
                       return_transitions=True
                       ):

        assert(randomization_level == 0 or randomization_level == 1)
        self.loader = decord.VideoLoader(vid_list, 
                                         ctx=decord.cpu(), 
                                         shape=(batch_size,)+image_shape, 
                                         interval=intra_batch_skip,
                                         skip=inter_batch_skip,
                                         shuffle=randomization_level)
        self.post_transform = post_transform
        self.overlap_points = overlap_points
        self.frames_per_point = frames_per_point
        self.return_transitions = return_transitions


        self._length = len(self.loader)
        self._loader_iter = None
        self._current_index = 0
        self._current_batch = None
        self._next_batch = None

        

    def __len__(self):
        return self._length

    def __iter__(self):
        self._current_index = 0
        self._loader_iter = iter(self.loader)
        return self

    def __next__(self):
        if self._current_index >= self._length:
            raise StopIteration

        if self._current_index==0:
            self._current_batch = next(self._loader_iter) # Is tuple of torch tensor and indices
        else:
            self._current_batch = self._next_batch

        if self._current_index < self._length-1:
            self._next_batch = next(self._loader_iter)
        else:
            self._next_batch = None

        self._current_index += 1
        return self._transform(self._current_batch, self._next_batch)

    # Apply all transforms to data batch
    # Presume shuffle level 0 or 1, otherwise this doesn't work
    def _transform(self, cur_batch, next_batch):
        # cur_batch is never None when this function is called, but next_batch could be


        cur_array = cur_batch[0]#.numpy()
        if self.frames_per_point > len(cur_array):
            return None, None
        cur_inds = cur_batch[1]
        next_size = 1 if self.overlap_points else self.frames_per_point #The else may not be right, not that it matters

        if next_batch is not None and self.return_transitions:
            next_array = next_batch[0]#.numpy()
            next_inds = next_batch[1]
            cur_array = torch.cat([cur_array, next_array[:next_size]])
            cur_inds = cur_inds + next_inds[:next_size]
        else:
            next_inds = None
            if self.return_transitions:
                cur_array = torch.cat([cur_array, torch.zeros((next_size,)+cur_array.size()[1:], dtype=cur_array.dtype)])
                cur_inds = cur_inds + [[-1,-1] for _ in range(next_size)]

        cur_array = self.post_transform(cur_array.numpy()) if self.post_transform is not None else cur_array
        cur_array = torch.Tensor(cur_array)
        size = len(cur_array) #including extra point on end

        # Don't need whole next_array, just next few points from it


        # Todo:
        #  Reshape into torch rgb pane format [3, x, y]
        cur_array = cur_array.permute(0,3,1,2)

        #  Reshape being mindful of frames_per_point and overlap_points
        # need torch.gather I think
        # or not
        if self.overlap_points:
            cur_array = [ cur_array[i:(size-self.frames_per_point+i+1)] for i in range(self.frames_per_point) ]
            cur_array = torch.cat(cur_array, dim=1)
            cur_inds = cur_inds[(self.frames_per_point-1):] #identify multiframes with index of last frame
        else:
            pass

        #  Reshape being mindful of transitions
        if self.return_transitions:
            cur_array = torch.stack([ cur_array[:(size-1)], cur_array[1:] ])#.transpose(1,0) #don't really need transpose
            cur_inds = cur_inds[:(size-1)] #Label of next transition not counted

        #  Change to float
        cur_array = cur_array.float()

        #  Return
        return cur_array, cur_inds


















# Todo: wrap decord loader in class that can be used for training with transitions


def torch_tensor_transform(t, batch=True):
    # transform rgb uint8
    if batch:
        t = t.permute(0,3,1,2)
    else:
        t = t.permute(2,0,1)
    t = tfunc.to_grayscale(t)
    t = t.float()

    return t

def tensor2cv(t, batch=True):
    if batch:
        t = t.permute(0,2,3,1)
    else:
        t = t.permute(1,2,0)
    t = t.numpy().astype('uint8')
#    t = t.numpy()
##    t = t.transpose(0,3,1,2)
#    t = cv2.cvtColor(t, cv2.COLOR_RGB2BGR)
    return t





def test1():
    vidloader = decord.VideoLoader(train_fullpaths, ctx=decord.cpu(), shape=(8,224,224,3), interval=0, skip=-1, shuffle=0)
    print(vidloader)
    print(type(vidloader))
    print(len(vidloader))
    
    for batch in vidloader:
        data, inds = batch
        print(type(data))
        print(data.shape)
        print(data.dtype)
        #data = np.array(data)
        data = data.numpy()
        print(type(data))
        print(data.dtype)
        print(data.shape)
        print(inds)
#        print( 
        input("continue")


def test2():
    vr = VideoReader(os.path.join(datadir,trainvids[0]), ctx=cpu(0), width=224, height=224)
    print(len(vr))
    print(vr.get_avg_fps())
    inds = [0,1,2,3,4,1,2,3,4,5]
    frames = vr.get_batch(inds)
    print(frames.size())
    print(frames.dtype)
    tstamps = vr.get_frame_timestamp(inds)
    print(type(tstamps))
    print(tstamps.shape)
    print(tstamps)

def test3():
    vr = VideoReader(os.path.join(datadir,trainvids[0]), ctx=cpu(0), width=224, height=224)
    frames = vr.get_batch(np.arange(len(vr)))
    newframes = torch_tensor_transform(frames)
    print(newframes.shape)
    transformed_back = tensor2cv(newframes)
    print(transformed_back.shape)
#    play_video(frames.numpy())
#    play_video(transformed_back)


# Note: get images as numpy, use cv2 to transform them to avoid the hassle of converting to PIL image

test1()

