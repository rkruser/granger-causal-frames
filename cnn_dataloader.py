from random import shuffle
from decord import VideoReader
from decord import cpu
import numpy as np
import torch
import six

class CNNLSTMDataLoader:
    def __init__(self, annotation,
                       frame_interval=3,
                       image_shape=(224,224,3),
                       device='cuda',
                       batch_size=64,
                       if_shuffle=False):

        self.annotation = annotation
        self.frame_interval = frame_interval
        self.image_shape = image_shape
        self.device = device
        self.batch_size = batch_size
        self.if_shuffle = if_shuffle

        with open(self.annotation) as f:
            annotations = f.readlines()

        if self.if_shuffle:
            annotations = shuffle(annotations)

        annotations = [i.strip() for i in annotations]
        self.n_samples = len(annotations)
        self.pathes = [i.split()[0] for i in annotations]
        self.labels = [float(i.split()[1]) for i in annotations]
        self._file_pointer = 0

    def __iter__(self):
        self._file_pointer = 0
        return self

    def __next__(self):
        next_pointer = min(self._file_pointer + self.batch_size, len(self.pathes))
        outputs = []
        lengths = []
        while self._file_pointer < next_pointer:
            vr = VideoReader(self.pathes[self._file_pointer], ctx=cpu(0), width=self.image_shape[0], height=self.image_shape[1])
            n_frames = len(vr)
            frames = vr.get_batch(np.arange(0, n_frames, self.frame_interval)).asnumpy()
            frames = np.transpose(frames, (0, 3, 1, 2))
            outputs.append(frames)
            self._file_pointer += 1

        outputs = pad_sequences(outputs, padding='post')
        outputs = torch.from_numpy(outputs).float().to(self.device)
        # what is y
        # return type?
        return outputs

    def __len__(self):
        return self.n_samples

# From Keras
def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='post', truncating='post', value=0.):
    """Pads sequences to the same length.
    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.
    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the beginning or the end
    if padding='post.
    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.
    Pre-padding is the default.
    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.
    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`
    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    sample_shape = ()
    flag = True

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.

    for x in sequences:
        try:
            lengths.append(len(x))
            if flag and len(x):
                sample_shape = np.asarray(x).shape[1:]
                flag = False
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x
