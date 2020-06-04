# memory.py
# Maintain a "replay" memory for Q-learning
# Input: A list of sequence lengths in frames
#  [ vid1length, vid2length, vid3length, ..., vidnlength]


# videofile: mp4 file name
# videolabel: a named tuple consisting of attributes of the video, like crash time, etc.
# 


# Somewhere I need a loader function to truncate vids, etc.
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import namedtuple
import torch
from torch.utils.data import Dataset
import os

#T = 14

class Zipindices:
    # length_list : a list of integers specifying lengths of other objects
    def __init__(self, length_list):
        self.llist = np.array(length_list)
        self.clist = np.cumsum(length_list)-1
        
    def __len__(self):
        if len(self.clist)>0:
            return self.clist[-1]+1
        else:
            return 0
        
    def __getitem__(self, key):
        if key < 0:
            key = self.__len__()+key
        if key < 0 or key >= self.__len__():
            raise KeyError('Zipindices key out of bounds')
        place = np.searchsorted(self.clist, key)
        if place==0:
            return place, key
        else:
            return place, key-self.clist[place-1]-1

class Zipindexables:
    # indexables is an iterable of objects with len and getitem defined
    def __init__(self, indexables, slice_is_range=True):
        self.indexables = indexables
        self.slice_is_range = slice_is_range
        self._buildindex()
        
    def _buildindex(self):
        llist = []
        for l in self.indexables:
            llist.append(len(l))
        self.zipindex = Zipindices(llist)       
        
    def __len__(self):
        return len(self.zipindex)
    
    def __getitem__(self, key):
        if isinstance(key,int) or isinstance(key,np.int64):
            return self._access(key)
        elif isinstance(key, slice):
            return self._slice(key)
        elif isinstance(key, np.ndarray):
            if len(key.shape)==1:
                if key.dtype=='bool':
                    key = np.arange(len(key))[key]
                    return self._int_array_index(key)
                elif 'int' in str(key.dtype):
                    return self._int_array_index(key)
                else:
                    raise KeyError('Zipindexables key array has invalid type')
            else:
                raise(KeyError('Zipindexables key array has wrong number of dimensions'))
        else:
            try:
                return self._access(key)
            except KeyError as e:
                raise KeyError('Zipindexables key error') from e

    def __iter__(self):
        self.iter_index=0
        return self

    def __next__(self):
        if self.iter_index < self.__len__():
            item = self.__getitem__(self.iter_index)
            self.iter_index += 1
            return item
        else:
            raise StopIteration
            
    def get_indexable(self, key):
        return self.indexables[key]
    
    def num_indexables(self):
        return len(self.indexables)
    
    def __str__(self):
        return self.__repr__()+', contents = '+str(self.indexables)
        
    def __repr__(self):
        return 'Zipindexables object, indexables {0}, items {1}'.format(
            self.num_indexables(), self.__len__())
            
    def _access(self, key):
        place, ind = self.zipindex[key]
        return self.indexables[place][ind]
        
    # Not maximally efficient, but whatever
    def _slice(self, key):
        start = key.start
        stop = key.stop
        step = key.step
        if start is None:
            start = 0
        if step is None:
            step = 1
        if stop is None:
            stop = self.__len__()
            
        if self.slice_is_range:
            return self._int_array_index(range(start,stop,step)) #changed from np.arange to range
        
        if step < 0:
            print("Warning: negative step size produces undefined behavior when slicing a Zipindexables object")
        
        place_list = []
        place_inds = {}
        for i in range(start, stop, step):
            place, ind = self.zipindex[i]
            if place not in place_inds:
                place_inds[place] = [ind, ind]
                place_list.append(place)
            else:
                place_inds[place][1] = ind
            
        new_items = []
        for j in place_list:
            sl = place_inds[j]
            new_items.append(self.indexables[j][sl[0]:sl[1]+step:step])
                             
        return Zipindexables(new_items)
            
    def _int_array_index(self, key):
        all_items = []
        for i in key:
            all_items.append(self._access(i))
        return all_items
        
#    def _bool_array_index(self, key):
#        if len(key) != self.__len__():
#            raise KeyError('Zipindexables numpy boolean key has wrong length')
#        all_items = []
#        for i in range(len(key)):
#            if key[i]:
#                all_items.append(self._access(i))
#        return all_items
    



'''
video_object is a sliceable object with the interface of the Video class.
videoinfo is a named tuple with external attributes of the video
'''
def vid_filter(video_object, videoinfo):
    pass


def ftransform(frame):
    pass


# Returns a generator that takes a valid videocapture object
#  and iterates through its frames, returning frame number and frame
#def video_frame_generator(videocapture_object):
#    pass


# Instead, have video file wrapper that provides an interface for
#  loading frames. Can swap between cv2 and ffmpeg more easily that way
#  Video class below uses this wrapper
class VideoFile:
    class VideoLoadError(Exception):
        def __init__(self, message='Video did not load'):
            super().__init__(message)
    
    # Need absolute path of file apparently
    def __init__(self, filename):
        self.filename = filename
        self.video_object = cv2.VideoCapture(self.filename)
        if not self.video_object.isOpened():
            raise VideoFile.VideoLoadError()
        self.frame_pos = 0
        
        self.num_frames = int(self.video_object.get(cv2.CAP_PROP_FRAME_COUNT))
        self.framerate = self.video_object.get(cv2.CAP_PROP_FPS)
        self.width = self.video_object.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.video_object.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.frame_duration = 1.0/self.framerate       
        self.duration = self.num_frames*self.frame_duration
        
    def __len__(self):
        return self.num_frames
    
    # This getitem is slow and is not recommended for use in a performance loop
    # Instead, use the iterator
    # Slicing is implemented, but not numpy array indexing (not really necessary though)
    def __getitem__(self, n):
        if isinstance(n,slice):
            return self._slice(n)
        if n < 0:
            n = self.__len__()+n
        if n < 0 or n >= self.__len__():
            raise KeyError('VideoFile key out of bounds')
        ret, frame = self._nthframe(n)
        if not ret:
            raise KeyError('Could not return frame {}'.format(n))
        return frame

    def _slice(self, n):
        start = n.start
        stop = n.stop
        step = n.step
        if start is None:
            start = 0
        if stop is None:
            stop = self.__len__()
        if step is None:
            step = 1

        fms = []
        for i in range(start, stop, step):
            fms.append(self.__getitem__(i))
        return np.array(fms)
    
    def _nthframe(self, n):
        self._setframe(n)
        return self._read_cur_frame()
    
    def _setframe(self, n):
        self.video_object.set(cv2.CAP_PROP_POS_FRAMES, n)
        self.frame_pos = n
        
    def _read_cur_frame(self):
        ret, frame = self.video_object.read()
        self.frame_pos += 1
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # OpenCV is BGR by default
        return ret, frame
    
    # Time in seconds
    def framenum_at_time(self, time):
        return int(time*self.framerate)
    
    def reset_video(self):
        self._setframe(0)
    
    def __iter__(self):
        self.reset_video()
        return self
    
    def __next__(self):
        ret, frame = self._read_cur_frame()
        if ret:
            return frame
        else:
            if self.frame_pos < self.num_frames:
                print("Warning: VideoFile iterator exited before end of video")
            raise StopIteration
            
    def play(self, scale=0.25, playback_speed=1.0, window_name=None):
        if window_name is None:
            window_name = self.filename
        play_video(self, frame_duration=self.frame_duration, scale=scale, playback_speed=playback_speed,
                 window_name=window_name)
            
    def _cleanup(self):
        self.video_object.release()
            
#    def __del__(self):
#        self._cleanup()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exception_type, exception_value, traceback):
        self._cleanup()
        

        
def play_video(frames, frame_duration=40, fps=None, scale=1.0, playback_speed=1.0, window_name='video',
             playback_transform = lambda f: cv2.cvtColor(f, cv2.COLOR_RGB2BGR) ):
    print("Press q to stop playback")

    playback_multiplier = 1.0/playback_speed
    
    if fps is not None:
        frame_duration = 1.0/fps

    #If duration is too short then video won't work
    frame_duration_ms = int(frame_duration*1000*playback_multiplier) 

    continue_playing = True
    while continue_playing:
        for frame in frames:
            frame = playback_transform(frame)
            frame = cv2.resize(frame, (int(np.size(frame,1)*scale), int(np.size(frame,0)*scale)))
            #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) #undo RGB for playing
            cv2.imshow(window_name, frame)
            if cv2.waitKey(frame_duration_ms) & 0xFF == ord('q'):
                continue_playing = False
                break
    cv2.destroyAllWindows()       
        







        
# Treat a numpy array as a video file
# Or possibly just an object that can be indexed like a numpy array
# Possibly just merge this with vidLoader instead of separating them
# Have a base class

# If we want to save memory, possibly use the VideoFile object as the self.array
#  rather than loading the whole video into memory (not sure what cv2 does though)
class VideoArray:
    '''
    videofile is a string with a path to the file (must be absolute)
    videolabel is a named tuple with external attributes of the video
    '''
    
    # Also store playback properties somewhere in here
    #  (default ones if no access to video itself)
    
    # frame_subtraction not implemented
    def __init__(self, 
                 videofile=None, 
                 videoarray=None,
                 labels=None,
                 labelinfo=None, # named tuple or other accessible object
                 label_func=lambda key, size, labelinfo : 0, #?
                 frames_per_datapoint=1, 
                 overlap_datapoints=True, #only relevant if more than 1 frame per datapoint
                 frame_subtraction=False,
                 return_transitions=False,
                 frame_transform=lambda x : x, #Transformation to apply per-frame
                 playback_info = {'fps':25},
                 sample_every=1, #Rate to keep frames
                 start_frame=None,
                 end_frame=None,
                 start_time=0,
                 end_time=-1,
                 TERMINAL_LABEL=-2,
                 verbose=False
                ):
        
        
        self.playback_info = playback_info
        self.verbose = verbose
        self.sample_every = sample_every
        
        # Consider moving these into helper functions
        if videoarray is not None:
            if start_frame is None:
                start_frame = int(start_time*playback_info['fps'])
            if end_frame is None:
                if end_time < 0:
                    end_frame = -1
                else:
                    end_frame = int(end_time*playback_info['fps'])

            if end_frame < 0:
                end_frame = len(videoarray)
            videoarray = videoarray[start_frame:end_frame:sample_every]

            if frame_transform is not None:
                all_frames = []
                for frame in videoarray:
                    all_frames.append(frame_transform(frame))
                videoarray = np.stack(all_frames)

            self.array = videoarray
                
        elif videofile is not None:
            with VideoFile(videofile) as vid:
                all_frames = []

                if start_frame is None:
                    start_frame = vid.framenum_at_time(start_time)
                if end_frame is None:
                    if end_time < 0:
                        end_frame=-1
                    else:
                        end_frame = vid.framenum_at_time(end_time)

                if end_frame < 0: 
                    end_frame = len(vid)

                count = 0
                for i, frame in enumerate(vid):
                    if self.verbose:
                        print(i)
                    if i < start_frame:
                        continue
                    if i >= end_frame:
                        break
                    if count%sample_every == 0:
                        all_frames.append(frame_transform(frame))
                    count += 1


                if len(all_frames)>0:
                    self.array = np.stack(all_frames)
                else:
                    # ****** !!!! Below should work but gives "float object cannot be interpreted as integer"
                    #dummy = frame_transform(np.zeros([vid.height, vid.width, 3]).astype('uint8')) # dtype???
                    #dims = [0]+list(dummy.shape)
                    #self.array = np.zeros(dims).astype(dummy.dtype)

                    #self.array = np.stack([frame_transform(np.zeros([vid.height, vid.width, 3]))])
                    self.array = np.zeros([0,224,224,3]) # Change size later

                self.playback_info.update({'frame_duration':vid.frame_duration,
                                            'num_sampled_frames':len(self.array),
                                            'adjusted_frame_duration':self.sample_every*vid.frame_duration,
                                            'sample_every':self.sample_every,
                                            'fps':vid.framerate,
                                            'window_name': vid.filename}) #Why is this the same for all videos?
            
        else:
            raise ValueError('VideoArray constructor received only None objects')
                
        self.frames_per_datapoint = frames_per_datapoint
        self.overlap_datapoints = overlap_datapoints
        self.frame_subtraction = frame_subtraction
        self.return_transitions = return_transitions
        self.labelinfo = labelinfo
        self.label_func = label_func       
        self.labels=labels
        self.TERMINAL_LABEL = TERMINAL_LABEL
        
        # These need to be rethought ***
        # Python apparently looks ahead when initializing, leading to dependency errors
#        self.length = VideoArray.compute_length(len(self.array), overlap_datapoints, frames_per_datapoint)
#        self.labels = VideoArray.extract_labels(self.length, label_func, labelinfo)
        
        self._compute_length()
        self._set_accessor_func()

        if self.labels is None:
            self._extract_labels() #Get array of labels for each frame
            self.given_labels = False
        else:
            self.given_labels = True

    
#    @staticmethod
#    def compute_length(array_len, overlap_datapoints, frames_per_datapoint):
#        if overlap_datapoints:
#            array_len = array_len-frames_per_datapoint+1
#        else:
#            array_len = array_len // frames_per_datapoint
#        return array_len
#    
#    @staticmethod
#    def extract_labels(length, label_func, labelinfo):
#        all_labels = []
#        for key in range(length):
#            all_labels.append(self.label_func(key,length,labelinfo))
#        labels = np.stack(all_labels)       
#        return labels
    
     # provide label for key value
    def _extract_labels(self):
#        self.labels = VideoArray.extract_labels(self.__len__(), self.label_func, self.labelinfo)
        all_labels = []
        for key in range(self.__len__()):
            all_labels.append(self.label_func(key,self.__len__(),self.labelinfo)) #depends on size too, maybe on other parameters
        if len(all_labels)>0:
            self.labels = np.stack(all_labels)
        else:
            self.labels=np.array([]) #improve later?
    
    def _compute_length(self):
#        self.length = VideoArray.compute_length(len(self.array), self.overlap_datapoints, self.frames_per_datapoint)
        l = len(self.array)
        if self.overlap_datapoints:
            l = l-self.frames_per_datapoint+1
        else:
            l = l // self.frames_per_datapoint
        
        if l >= 0:
            self.length = l
        else:
            self.length = 0
        
    def _set_accessor_func(self):
        if self.frames_per_datapoint > 1:
            if self.return_transitions:
                self.accessor_func = self.getbundletransition
            else:
                self.accessor_func = self.getbundle
        else:
            if self.return_transitions:
                self.accessor_func = self.getframetransition
            else:
                self.accessor_func = self.getframe
        
#      Use object.__setattr__() or something to get around this
#    def __setattr__(self, name, val):
#        super().__setattr__(name, val)
#        if name in ['overlap_datapoints', 'frames_per_datapoint']:
#            self._compute_length()
##        if name in ['frames_per_datapoint', 'return_transitions']:
##            self._set_accessor_func()
#        if name in ['labelinfo', 'label_func']:
#            self._extract_labels()
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            start = key.start
            stop = key.stop
            step = key.step
            if start is None:
                start = 0
            if stop is None:
                stop = self.__len__()
            if step is None:
                step = 1
            return self._slice(range(start,stop,step))
        elif isinstance(key, np.ndarray):
            if len(key.shape)==1:
                if key.dtype=='bool':
                    key = np.arange(len(key))[key]
                    return self._slice(key)
                elif 'int' in str(key.dtype):
                    return self._slice(key)
                else:
                    raise KeyError('Zipindexables key array has invalid type')
            else:
                raise(KeyError('Zipindexables key array has wrong number of dimensions'))       
        else:
            if key < 0:
                key = key+self.__len__()
            if key < 0 or key > self.__len__():
                raise KeyError('VideoArray key out of bounds')
            return self._retrieve_item(key)
        
    def _slice(self, slice_range):
        all_items = []
        for i in slice_range:
            all_items.append(self._retrieve_item(i))
        if len(all_items) > 0:
            array, labels = zip(*all_items)
            array = np.stack(array)
            labels = np.stack(labels)
        else:
            array = np.array([])
            labels = np.array([])

        return array, labels
        
    def _retrieve_item(self, key):
#        if self.frames_per_datapoint > 1:
#            if self.return_transitions:
#                return self.get_bundle_transition(key)
#            else:
#                return self.get_bundle(key)
#        else:
#            if self.return_transitions:
#                return self.get_frame_transitions(key)
#            else:
#                return self.get_frame(key)
        return self.accessor_func(key)
        
    def getframe(self, key):
        return self.array[key], self.labels[key]
    
    def getbundle(self, key):
        if self.overlap_datapoints:
            return self.array[key:key+self.frames_per_datapoint], self.labels[key]
        else:
            index = key*self.frames_per_datapoint
            return self.array[index:index+self.frames_per_datapoint], self.labels[key]
        return
    
    # Right now slicing just doesn't work with terminal states
    # I guess that's fine for now
    def getframetransition(self, key):
        cur, cur_label = self.getframe(key)
        if key == self.__len__()-1:
            target = np.zeros(cur.shape) #Need to make this zeros in shape of other thign
            target_label = self.TERMINAL_LABEL # Need to use standard terminal label value like Nan
        else:
            target, target_label = self.getframe(key+1)
        return np.stack([cur, target]), np.stack([cur_label, target_label])
    
    # Right now slicing just doesn't work with terminal states
    # I guess that's fine for now
    def getbundletransition(self, key):
        cur, cur_label = self.getbundle(key)
        if key == self.__len__()-1:
            target = np.zeros(cur.shape) # Need to make this zeros in shape of other thing
            target_label = self.TERMINAL_LABEL #Need to use standard terminal label value
        else:
            target, target_label = self.getbundle(key+1)
        return np.stack([cur, target]), np.stack([cur_label, target_label])
    
    def play(self):
        play_video(self.array, **self.playback_info)
    
    


######## Stuff about loading datasets below ##############

def frame_transform_1(frame):
    frame = cv2.resize(frame, (224,224))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return frame

def frame_transform_2(frame):
    frame = cv2.resize(frame, (224,224))
    #frame = np.transpose(frame, (0,3,1,2))
    return frame


InfoType1 = namedtuple('InfoType1', 'name crash crashtime startframe endframe starttime endtime')

def label_func_1(key, size, info):
    if key == size-1:
        if info.crash:
            return 1.0
    return 0.0

def label_func_2(key, size, info):
    if not info.crash:
        return 0.0
    index = size-1-key
    return -0.858**index #Chosen so that 5 seconds back, sampling every 5th frame, has near zero reward


def label_func_prediction_only(key, size, info):
    if info.crash:
        return 1.0
    return 0.0


class FileTracker:
    def __init__(self):
        pass
    def file_list(self):
        pass
    def file_ids(self):
        pass
    def basenames(self):
        pass
    def file_info(self):
        pass

class BeamNG_FileTracker(FileTracker):
    def __init__(self, datapath, basename_list=None, crash_truncate_list=None, manifest='manifest.txt', info_file='times.txt'):
        self.datapath = datapath
        self.basename_list = basename_list
        self.crash_truncate_list = crash_truncate_list
        self.manifest = os.path.join(datapath, manifest)
        self.info_file = os.path.join(datapath, info_file)

        if self.basename_list is None:
            print("Loading basenames from manifest is unimplemented")
            pass # Do something with the manifest
        if self.crash_truncate_list is None:
            self.crash_truncate_list = np.zeros(len(basename_list)).astype('bool')
        
        self.file_id_list = [os.path.splitext(fname)[0] for fname in basename_list]
        self.fullpaths = [os.path.abspath(os.path.join(self.datapath,fname)) for fname in basename_list]

        self.id_to_fullpath = {self.file_id_list[i]:self.fullpaths[i] for i in range(len(basename_list))}
        self.id_to_crash_truncate = {self.file_id_list[i]:self.crash_truncate_list[i] for i in range(len(basename_list))}


        self._load_info_file()


    def _load_info_file(self):
        self.info = {}
        with open(self.info_file, 'r') as fobj:
            lines = fobj.readlines()
            for line in lines:
                fid, starttime, crashtime = line.split()[:3]

                if fid in self.id_to_fullpath:
                    starttime = float(starttime)
                    crashtime = float(crashtime)

                    if self.id_to_crash_truncate[fid]:
                        endtime=crashtime-0.5
                        crash=False
                    else:
                        endtime=-1
                        crash=True

                    self.info[self.id_to_fullpath[fid]] = InfoType1(name=fid,
                                                          crash=crash,
                                                          crashtime=crashtime,
                                                          starttime=starttime,
                                                          endtime=endtime,
                                                          startframe=None,
                                                          endframe=None)

            # Need to map file ids to full paths and nametuples

    # Return list of fullpath filenames
    def file_list(self):
        return self.fullpaths

    def basenames(self):
        return self.basename_list

    def file_ids(self):
        return self.file_id_list

    # Return dict of file names associated with named tuples, based on manifest
    def file_info(self):
        return self.info


def random_proportion_true(size, proportion=0.5):
    arr = np.zeros(size).astype('bool')
    inds = np.random.permutation(np.arange(size))[:int(size*proportion)]
    arr[inds] = True
    return arr

#datapath = '/mnt/linuxshared/phd-research/data/beamng_vids/all_vids'
#basename_list = [ 'v1_1.mp4', 'v1_2.mp4', 'v1_3.mp4', 'v1_4.mp4' ]


# VideoArray can't handle empty arrays yet, but it's fine for now


class VideoDataset(Dataset):
    '''
    Vidfiles is a list of full video file strings
    videoinfo is a dict associating file basenames (no extension) with named tuples of video info.
    vid_filter takes video objects and processes them based on videoinfo
    '''
    def __init__(self, 
                vidfiles, # List of full video paths
                videoinfo, # Dict associating files with named tuples of labels
                frame_transform=frame_transform_1, 
                label_func=label_func_1, 
                frames_per_datapoint=1, 
                frame_subtraction=False,
                return_transitions=False,
                sample_every=1,
                TERMINAL_LABEL=-2,
                verbose=False,
                overlap_datapoints=True,
                is_color=False,
                label_postprocess=True,
                batch_access=False
                ):
        all_video_arrays = []
        for vidfile in vidfiles:
            print("Loading", vidfile)
            labelinfo = videoinfo[vidfile]
            all_video_arrays.append(VideoArray(videofile=vidfile, 
                                       labelinfo=labelinfo,
                                       label_func=label_func, 
                                       frames_per_datapoint=frames_per_datapoint,
                                       overlap_datapoints=overlap_datapoints, 
                                       frame_subtraction=frame_subtraction,
                                       return_transitions=return_transitions,
                                       frame_transform=frame_transform,
                                       sample_every=sample_every,
                                       start_time=labelinfo.starttime, #named tuple must include start and end frames
                                       end_time=labelinfo.endtime, #To do: add support for start time and end time for easy truncation
                                       TERMINAL_LABEL=TERMINAL_LABEL,
                                       verbose=verbose))

        self.data = Zipindexables(all_video_arrays)

        if label_postprocess:
            self.post_process_labels = lambda y : torch.Tensor(y)
        else:
            self.post_process_labels = lambda y : y

        if batch_access: #Ugh, I hate all these if statements
            if is_color:
                if return_transitions:
                    if frames_per_datapoint>1:
                        self.post_process_array = lambda x : torch.Tensor(x).permute(0,1,2,5,3,4).float()/255.0
                    else:
                        self.post_process_array = lambda x : torch.Tensor(x).permute(0,1,4,2,3).float()/255.0
                else:
                    if frames_per_datapoint>1:
                        self.post_process_array = lambda x : torch.Tensor(x).permute(0,1,4,2,3).float()/255.0
                    else:
                        self.post_process_array = lambda x : torch.Tensor(x).permute(0,3,1,2).float()/255.0
            else:
                self.post_process_array = lambda x : torch.Tensor(x).float()/255.0
        else:
            if is_color:
                if return_transitions:
                    if frames_per_datapoint>1:
                        self.post_process_array = lambda x : torch.Tensor(x).permute(0,1,4,2,3).float()/255.0
                    else:
                        self.post_process_array = lambda x : torch.Tensor(x).permute(0,3,1,2).float()/255.0
                else:
                    if frames_per_datapoint>1:
                        self.post_process_array = lambda x : torch.Tensor(x).permute(0,3,1,2).float()/255.0
                    else:
                        self.post_process_array = lambda x : torch.Tensor(x).permute(2,0,1).float()/255.0
            else:
                self.post_process_array = lambda x : torch.Tensor(x).float()/255.0

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        if isinstance(i, np.ndarray): #Theoretically this allows indexing via numpy array
            xypairs = self.data[i]
            x, y = zip(*xypairs)
        else:
            if i < 0:
                i = i+self.__len__()
            if i < 0 or i > self.__len__():
                raise KeyError('VideoDataset key out of range')
            x, y = self.data[i]
        return self.post_process_array(x), self.post_process_labels(y)

    def num_videos(self):
        return self.data.num_indexables()
    
    def get_video(self,i):
        return self.data.get_indexable(i)

    def access_video(self, i, j):
        x, y = self.get_video(i)[j]
        return self.post_process_array(x), self.post_process_labels(y)
    
    def play_video(self, i):
        vid = self.get_video(i)
        vid.play()

    def video_iter(self):
        return self.__class__.VideoIter(self.data)

    class VideoIter:
        def __init__(self, data):
            self.data = data
        def __iter__(self):
            self.state = 0
            return self
        def __next__(self):
            if self.state >= self.data.num_indexables():
                raise StopIteration
            vid = self.data.get_indexable(self.state)
            self.state += 1
            return vid #torch.from_numpy(vid.array), torch.Tensor(vid.labels), vid.labelinfo, vid.playback_info
            
    
#    Need to think more about this one. (Careful of colorspaces, array reps)
#    def play_all(self):
#        play_video(self.data)

# Next: a quick RGB / Gray / tensor transform
# Then: Quick training of a large resnet
