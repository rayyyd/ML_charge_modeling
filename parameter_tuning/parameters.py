import numpy as np
import torch

# Removed loader import to avoid circular dependency

import sys
sys.path.append('../libs/')
import shjnn



# configure initial state here.


# dictionary of hyper params

#contains the dataset. to be loaded in my the dataloader or init.py.
dataset = {
    'trajs': None,       # Will store original trajectories
    'times': None,       # Will store original time points
    'y': None,           # intensity, voltage, delay, thickness.
    'cut_zero': True,    # whether to delete all data before t=0
    
    # Will store trajectories with missing data points
    'train_trajs': None,
    # Will store time points with missing data points
    'train_times': None,
    
    # set indices of missing data here. [1, 70].
    # if both options are populated, random drops will be prioritised with drop_number.
    'missing_idx': None,
    'drop_number': 20,
    'stochastic_level': 0.0, # stochastic noise level static.
    
}

#sijin
dataset_PV = {
    'elec_df': r'C:\Users\z5183876\Documents\GitHub\PV-syst-data\Bomen\data_from_server\2021_elec_df.pkl'
    }

#choose between options 'B-VAE', 'RNN', 'LSTM', 'RNN-VAE', 'CNN-VAE', 'LSTM-VAE'
trainer = 'B-VAE'
model = None
rnn = None
b_vae = None
load_check = False


#tune hyper-params here.
model_params = {
    # hyper params

    #NOTE: reccomended to increase rnn_nhidden size.
    'nhidden': 512,
    
    'rnn_linear1': 256,
    'rnn_linear2': 128,
    'rnn_linear3': 64,
    'rnn_linear4': 128,
    'rnn_linear5': 256,
    
    
    'rnn_nhidden': 512,
    'obs_dim': 6,  # Changed from 1 to 6 to match input data dimensions

    'latent_dim': 8,
        
    'b_vae_adaptive': True,
    
    'lr':1e-3,
    'lr_decay': 0.5,  # learning rate decay
    'lr_min': 1e-7,
    
    'beta': 0.05,
    'beta_gain': 2,  # beta gain for adaptive training
    'beta_max': 4,  # maximum beta value
    
    'n_batch': 16,  #batch size

    'optim': None,
    'device': None,
    'func': None,
    'rec': None,
    'dec': None,
    
    'stochastic_level_dynamic': 5, # stochastic noise level dynamic.

    # training params
    # TODO: doesn't make too much sense for separate epochs per train and total epochs.
    # NOTE: due a bug, total_epochs_Train must be greater than 14.
    'total_epochs_train': 1000,
    'epochs_per_train': 100,
    'epochs': 0, # a record of the epochs
    'loss': [], # loss = KL_loss + MSE loss
    'loss_thresh': 0.00001,
    'MSE_loss': [],
    'KL_loss': [],
    
    #specifying which trajectories to use (0 - 149 are available)
    'plot': [0, 1, 2, 3, 4],
    'random': False,
    

    #labels
    'name': "default",
    'desc': "default_desc",
    'folder': "saves",

    #automated
    'sequence_size': 70,
    
    #free to use memory for debugging only
    'debug': 1,
    'debug_list': [],
    'debug_list2': [],
    
    #adpative settings:
    'lr_track': None,
    'beta_track': None   
}


vae_params = {
    #rnn properties
    'rnn_nhidden': 2048,
    'input_size': 2,
    
}

records = {
    'loss': [],
    'MSE_loss': [],
    'KL_loss': [],

    #RNN specific
    'predictions': [],
    'targets': []
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

amp = 2
freq = 1
adaptive_training = {
    # change the training parameters during the training procedure, all lists has to have the same length.
    # 'lr': np.logspace(-2, -3, 20),
    'lr': np.logspace(-2.7, -4, 20),
    # 'lr': np.linspace(2e-3, 1e-4, 20),
    'beta': amp*np.sin(np.linspace(0, 5/2*np.pi, 20)) + 2,
    'epochs': [50]*20
}

# options for data visualisation
save = False                     # haven't figured this out yet.
split = False                    #same as above.

# options for grid search

class gridsearch:

    learning_rates = [1e-2]
    n_batches = [16]
    latent_dims = [2]
    betas = [4]

    run_description = "search2"

    data_record = {"learning rate": [],
                    "n_batch": [],
                    "latent_dim": [],
                    "beta": [],
                    "lowest_loss": []}

    loss_record = {"learning rate": [],
                    "n_batch": [],
                    "latent_dim": [],
                    "beta": [],
                    "loss": []}
    
    excel_folder_created = False
    excel_folder_path = ""


# latent_dim, nhidden, rnn_nhidden, obs_dim, lr, n_batch, beta = get_hyper_params(trajs)
# model_params = get_hyper_params(dataset, model_params)
# # func, rec, dec, optim, device, loss, epochs = init_model(latent_dim, nhidden, rnn_nhidden, obs_dim, n_batch, lr)
# model_params = init_model(model_params)
# training_loop(func, rec, dec, optim, trajs, times, n_batch, device, beta)

#clear saves folder
# clear_saves()