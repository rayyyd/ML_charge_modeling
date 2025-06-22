import parameters
import loader
from serial import RNN, LSTM
from autoencoders import vae
import b_vae
import torch
import numpy

import data_dropout

import sys
sys.path.append('../../libs/')
import shjnn



def init_shjnn(model_params=parameters.model_params):
    """
        inits models - always to CPU.
        args: model_params
        return: model_params (updated)
    """
    print("Logs: Init: Initialising SHJNN library")
    latent_dim = model_params['latent_dim']
    nhidden = model_params['nhidden']
    rnn_nhidden = model_params['rnn_nhidden']
    obs_dim = model_params['obs_dim']
    lr = model_params['lr']
    n_batch = model_params['n_batch']

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    func, rec, dec, optim, _ = shjnn.init_model(latent_dim, nhidden, rnn_nhidden, obs_dim, n_batch, lr, device=device)
    loss: list = []
    epochs = 0
    # Print device info for debugging
    print(f"Logs: Init: Using device: {device}")

    model_params['epochs'] = epochs

    model_params['func'] = func
    model_params['rec'] = rec
    model_params['dec'] = dec
    model_params['optim'] = optim
    model_params['device'] = device

    print("Logs: Init: Finished initialising SHJNN library")


    return model_params

'''
must be called only after init_shjnn called.
'''
def init_RNN(model_params=parameters.model_params):
    epochs = 0
    model_params['epochs'] = epochs
    #check if cuda is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_params['device'] = device
    return RNN(model_params)

def init_LSTM(model_params=parameters.model_params):
    epochs = 0
    model_params['epochs'] = epochs
    #check if cuda is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_params['device'] = device
    return LSTM(model_params)

def init_B_VAE(model_params=parameters.model_params):
    epochs=0
    return b_vae.B_VAE(model_params)

def init_autoencoder(model_params=parameters.model_params):
    #check if cuda is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_params['device'] = device
    # Get the VAE class and use its create method
    from autoencoders.vae import VAE
    return VAE.create(model_params, parameters.vae_params)



def load_data(drop_data=False):
    
    print("Logs: Init: Initialising data from dataloader")
    loader.load_data()
    trajs = parameters.dataset['trajs']
    obs_dim = trajs[0].size()[1]
    parameters.model_params['obs_dim'] = obs_dim
    print("Logs: Init: Finished initialising data from dataloader")
    
    # Store copies of original data before introducing missing data
    if drop_data:
        # Deep copy original data to train_trajs and train_times
        parameters.dataset['train_trajs'] = parameters.dataset['trajs'].clone()
        parameters.dataset['train_times'] = parameters.dataset['times'].clone()
        
        # Modify the training data to have missing points
        data_dropout.modify_data()
        print("Logs: Init: Modified data to have missing data.")
    else:
        # If no missing data, training data is the same as original data
        parameters.dataset['train_trajs'] = parameters.dataset['trajs'].clone()
        parameters.dataset['train_times'] = parameters.dataset['times'].clone()
    


