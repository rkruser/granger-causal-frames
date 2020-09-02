# All config parameters go here with their default values
import argparse
import sys
import os

from socket import gethostname# as hostname
hostname = gethostname()
if 'LV426' in hostname:
    datadir = '/mnt/linuxshared/data/BeamNG'
    modeldir = './models'
    label_file = os.path.join(datadir, 'full_annotation.txt')
    split_file = os.path.join(datadir, 'traintest_split.pkl')
elif 'vulcan' in hostname:
    modeldir = '/cfarhomes/krusinga/storage/research/causality/granger-causal-frames/models'
#    datadir = '/vulcan/scratch/ywen/car_crash/BeamNG_dataset'
    datadir = '/scratch0/krusinga/BeamNG_dataset'
    label_file = './annotation/full_annotation.txt'
    split_file = './annotation/traintest_split.pkl'

elif 'jacobswks20' in hostname:
    modeldir = './models'
    datadir = '/scratch0/datasets/BeamNG_dataset'
    label_file = './annotation/full_annotation.txt'
    split_file = './annotation/traintest_split.pkl'
#    label_file = os.path.join(datadir, 'full_annotation.txt')
#    split_file = os.path.join(datadir, 'traintest_split.pkl')

else:
    print("Unknown hostname")
    datadir = './all_vids'
    modeldir = './models'
    print("Default to", datadir, modeldir)

default_config_params = {
    # Training
    'batch_size': 128,
    'n_epochs': 50,
    'checkpoint_every': 1,
    'train_print_every': 1, #currently unused
    'num_data_workers':4,
    'shuffle_train_data':True,

    # Model / Data
    'network_type': 'resnet50',
    'frames_per_datapoint': 10,
    'use_transitions': True,
    'use_color': True,
    'image_shape': (224,224,3),
    'rl_gamma': 0.977, #Chosen to yield 0.5 discount 3 seconds into the past
    'use_q_loss': True,
    'terminal_weight':1, #weight of terminal states in loss
    'use_data_parallel':True,
    'randomize_start_frame':True,

    # Data preprocessing
    'frame_sample_freq': 3, #every 5th frame
    'overlap_datapoints': True,
    'frame_subtraction': False,
    'preload_num':50,

    # Optimizer
    'learning_rate': 0.0002,
    'beta1': 0.9,
    'beta2': 0.999,
    'eps': 1e-08,
    'weight_decay':0,
    'amsgrad':False,

    # General
    'model_name': 'model_terminal_1_gamma_977',
    'model_directory': modeldir,
    'model_checkpoint_number': -1,
    'data_directory': datadir,
    'device':'cuda:0',
    'random_seed': 53,
    'overwrite_last': True
}

# Items can repeat here
#partitions = {
#    'model_args': ['frames_per_datapoint', 'use_transitions', 'use_color'],
#    'optimizer_args': ['learning_rate', 'beta1', 'beta2', 'eps', 'weight_decay', 'amsgrad'],
#    'training_args': ['batch_size', 'n_epochs', 'checkpoint_every', 'train_print_every'],
#    'dataset_args': ['use_transitions', 'use_color', 'frame_sample_freq', 'overlap_datapoints',
#                     'frame_subtraction'],
#    'dataloader_args': ['num_data_workers'],
#    'general_args':['model_name', 'model_directory', 'model_checkpoint_number', 'data_directory',
#                    'device', 'random_seed']
#}


def construct_parser(cfg_dict):
    parser = argparse.ArgumentParser()
    for param, val in cfg_dict.items():
        valtype = type(val)
        if valtype is bool:
            valtype = int
        parser.add_argument('--'+param, type=valtype, default=val)

    return parser

        
def get_config(stringargs=None, default_dict = default_config_params): #Maybe add optional argument so can change configs from command line
    parser = construct_parser(default_dict)
    if stringargs is None:
        args = sys.argv[1:]
    else:
        args = stringargs.split() + sys.argv[1:] # Allow both command line and string args

    cfg, _ = parser.parse_known_args(args)
    return cfg, args





