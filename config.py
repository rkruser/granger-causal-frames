# All config parameters go here with their default values
import argparse
import sys

import socket
hostname = socket.gethostname()
if 'LV426' in hostname:
    datadir = '/mnt/linuxshared/phd-research/data/beamng_vids/all_vids'
    modeldir = './models'
elif 'vulcan' in hostname:
    datadir = '/vulcan/scratch/krusinga/BeamNG/all_vids'
    modeldir = '/vulcan/scratch/krusinga/BeamNG/models'
else:
    print("Unknown hostname")
    datadir = './all_vids'
    modeldir = './models'
    print("Default to", datadir, modeldir)

all_unique_default_config_params = {
    # Training
    'batch_size': 64,
    'n_epochs': 100,
    'checkpoint_every': 5,
    'train_print_every': 1, #currently unused
    'num_data_workers':4,
    'shuffle_train_data':True,


    # Model / Data
    'network_type': 'resnet50',
    'frames_per_datapoint': 3,
    'use_transitions': True,
    'use_color': False,
    'rl_gamma': 0.999,
    'use_q_loss': True,
    'terminal_weight':64, #weight of terminal states in loss

    # Data preprocessing
    'frame_sample_freq': 5, #every 5th frame
    'overlap_datapoints': True,
    'frame_subtraction': False,


    # Optimizer
    'learning_rate': 0.0002,
    'beta1': 0.9,
    'beta2': 0.999,
    'eps': 1e-08,
    'weight_decay':0,
    'amsgrad':False,

    # General
    'model_name': 'model',
    'model_directory': modeldir,
    'model_checkpoint_number': -1,
    'data_directory': datadir,
    'device':'cuda:0',
    'random_seed': 53,
    'overwrite_last': False
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

        
def get_config(stringargs=None): #Maybe add optional argument so can change configs from command line
    parser = construct_parser(all_unique_default_config_params)
    if stringargs is None:
        args = sys.argv[1:]
    else:
        args = stringargs.split() + sys.argv[1:] # Allow both command line and string args

    cfg, _ = parser.parse_known_args(args)
    return cfg, args



trainvids = [
    'v1_1.mp4',
    'v1_2.mp4',
    'v1_3.mp4',
    'v1_4.mp4',
    'v1_5.mp4',
    'v1_6.mp4',
    'v1_7.mp4',
    'v1_8.mp4',
    'v1_9.mp4',
    'v1_10.mp4',
    'v1_11.mp4',
    'v1_12.mp4',
    'v1_13.mp4',
    'v1_14.mp4',
    'v2_2.mp4',
    'v2_4.mp4',
    'v2_7.mp4',
    'v2_9.mp4',
    'v2_11.mp4',
    'v3_1.mp4',
    'v3_2.mp4',
    'v3_3.mp4',
    'v3_4.mp4',
    'v3_5.mp4',
    'v3_7.mp4',
    'v4_1.mp4',
    'v4_3.mp4',
    'v4_4.mp4',
    'v4_5.mp4',
    'v4_6.mp4',
    'v4_8.mp4',
    'v4_9.mp4',
    'v4_10.mp4',
    'v4_13.mp4',
    'v4_14.mp4',
    'v4_15.mp4',
    'v4_17.mp4',
    'v4_18.mp4',
    'v5_1.mp4',
    'v5_2.mp4',
    'v5_3.mp4',
    'v5_4.mp4',
    'v5_5.mp4',
    'v5_6.mp4',
    'v5_7.mp4',
    'v5_8.mp4',
    'v6_1.mp4',
    'v6_2.mp4',
    'v6_3.mp4',
    'v6_4.mp4',
    'v6_5.mp4',
    'v7_1.mp4',
    'v7_2.mp4',
    'v7_3.mp4',
    'v7_4.mp4',
    'v7_5.mp4',
    'v7_6.mp4',
    'v7_8.mp4',
    'v7_9.mp4',
    'v7_11.mp4',
    'v7_14.mp4',
    'v7_15.mp4',
    'v7_17.mp4',
    'v7_18.mp4',
    'v8_1.mp4',
    'v8_2.mp4',
    'v8_3.mp4',
    'v8_4.mp4',
    'v8_5.mp4',
    'v8_6.mp4',
    'v9_1.mp4',
    'v9_3.mp4',
    'v9_4.mp4',
    'v10_2.mp4',
    'v10_3.mp4',
    'v10_4.mp4',
    'v11_1.mp4',
    'v11_2.mp4',
    'v11_3.mp4',
    'v11_4.mp4',
    'v11_5.mp4',
    'v11_6.mp4',
    'v12_1.mp4',
    'v13_1.mp4',
    'v13_2.mp4',
    'v13_3.mp4',
    'v13_4.mp4',
    'v13_5.mp4',
    'v13_6.mp4',
    'v13_7.mp4',
    'v14_1.mp4',
    'v14_2.mp4',
    'v14_3.mp4',
    'v14_4.mp4',
    'v14_6.mp4',
    'v14_7.mp4',
    'v15_1.mp4',
    'v15_2.mp4',
    'v15_3.mp4',
    'v15_4.mp4',
    'v15_5.mp4',
    'v15_7.mp4',
    'v15_8.mp4',
    'v15_9.mp4',
    'v15_11.mp4',
    'v15_12.mp4',
    'v15_13.mp4',
    'v16_1.mp4',
    'v16_2.mp4',
    'v16_3.mp4',
    'v16_4.mp4',
    'v16_6.mp4',
    'v16_7.mp4',
    'v16_9.mp4',
    'v16_10.mp4',
#    'v17.mp4', #Beginning of multicar
#    'v18_1.mp4',
#    'v19.mp4',
#    'v20.mp4',
#    'v22.mp4',
#    'v23.mp4',
#    'v25.mp4',
#    'v26.mp4',
#    'v28.mp4',
#    'v29.mp4',
#    'v31.mp4',
#    'v32.mp4',
#    'v34.mp4',
#    'v35.mp4',
#    'v37.mp4',
#    'v38.mp4',
#    'v40.mp4',
#    'v41.mp4',
#    'v43.mp4',
#    'v44.mp4',
#    'v46.mp4',
#    'v47.mp4'
]

testvids = [
    'v12_2.mp4',
    'v4_12.mp4',
    'v15_10.mp4',
    'v7_16.mp4',
    'v10_1.mp4',
    'v4_19.mp4',
    'v4_7.mp4',
    'v2_6.mp4',
    'v13_8.mp4',
    'v3_6.mp4',
    'v4_16.mp4',
    'v7_13.mp4',
    'v7_10.mp4',
    'v5_9.mp4',
    'v4_11.mp4',
    'v7_7.mp4',
    'v9_2.mp4',
    'v6_6.mp4',
    'v2_5.mp4',
    'v16_5.mp4',
    'v7_12.mp4',
    'v4_2.mp4',
    'v2_3.mp4',
    'v14_5.mp4',
    'v2_1.mp4',
    'v16_8.mp4',
    'v15_6.mp4',
    'v2_8.mp4',
    'v2_10.mp4',
#    'v18_2.mp4', #Beginning of multicar
#    'v21.mp4',
#    'v24.mp4',
#    'v27.mp4',
#    'v30.mp4',
#    'v33.mp4',
#    'v36.mp4',
#    'v39.mp4',
#    'v42.mp4',
#    'v45.mp4',
#    'v48.mp4'
]


