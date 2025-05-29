import torch
import matplotlib.pyplot as plt
import numpy as np
import time, os, pickle
import pandas as pd


import init
import loader
import visualisation
import parameters
import ipynb_utils
import sys
from serial import RNN
from b_vae import B_VAE
from autoencoders import vae
sys.path.append('../../libs/')
import shjnn

def get_sweep_index(reference_index, sweep):
    """Get the sweep index based on the reference index and sweep type.
    Args:
        reference_index (int): The reference index to base the sweep on.
        sweep (str): The sweep type ('intensity', 'voltage', or 'delay').
    Returns:
        list: A list of indices corresponding to the sweep.
    """
    assert reference_index < len(parameters.dataset['y']), f"Reference index {reference_index} is out of bounds for dataset length {len(parameters.dataset['y'])}."
    INTENSITY = 0
    VOLTAGE = 1
    DELAY = 2
    output_list = []
    
    reference_metadata = parameters.dataset['y'][reference_index]
    if sweep == 'intensity':
        out_list = [i for i, meta in enumerate(parameters.dataset['y']) if meta[VOLTAGE] == reference_metadata[VOLTAGE] and meta[DELAY] == reference_metadata[DELAY]]
        output_list = sorted(out_list, key=lambda i: parameters.dataset['y'][i][INTENSITY])
    elif sweep == 'voltage' or sweep == 'bias':
        out_list = [i for i, meta in enumerate(parameters.dataset['y']) if meta[INTENSITY] == reference_metadata[INTENSITY] and meta[DELAY] == reference_metadata[DELAY]]
        output_list = sorted(out_list, key=lambda i: parameters.dataset['y'][i][VOLTAGE])
    elif sweep == 'delay':
        out_list = [i for i, meta in enumerate(parameters.dataset['y']) if meta[INTENSITY] == reference_metadata[INTENSITY] and meta[VOLTAGE] == reference_metadata[VOLTAGE]]
        output_list = sorted(out_list, key=lambda i: parameters.dataset['y'][i][DELAY])
    else:
        raise ValueError("Invalid sweep type. Choose 'intensity', 'voltage', or 'delay'.")
    return output_list

def get_latent_vectors(model_params, dataset, traj_idx=None):
    # Extract data from dataset
    trajectories = dataset['trajs']
    time_points = dataset['times']
    metadata = dataset['y']

    # Extract model components
    model_func = model_params['func']
    encoder = model_params['rec']
    decoder = model_params['dec']
    optimizer = model_params['optim']
    device = model_params['device']
    epoch_num = model_params['epochs']
    latent_dims = model_params['latent_dim']

    # Create inference function for trajectory encoding
    infer_step_encode = shjnn.make_infer_step(
        model_func, encoder, decoder, optimizer, device, 
        input_mode='traj', sample=False
    )

    # Get indices of all trajectories
    if traj_idx is None:
        sample_indices = list(range(len(trajectories)))
    else:
        sample_indices = traj_idx
        
    # Arrays to store latent vectors
    latent_vectors = []      # First timestep only
    all_latent_vectors = []  # All timesteps
    
    # Process each trajectory to collect latent representations
    for idx in sample_indices:
        # Prepare trajectory tensor
        traj_tensor = trajectories[idx].view(1, *trajectories[idx].size()).to(device)
        
        # Create time points tensor
        pred_times = np.linspace(0, 2.5, 1000)
        time_tensor = torch.Tensor(pred_times).to(device)

        # Get model prediction and latent vectors
        pred_x, pred_z = infer_step_encode(traj_tensor, time_tensor)
        
        # Store latent vectors
        latent_vectors.append(pred_z[0, 0, ...].detach().cpu().numpy())  # First timestep only
        all_latent_vectors.append(pred_z[0, ...].detach().cpu().numpy())  # All timesteps
        
    # Convert lists to numpy arrays
    latent_vectors = np.stack(latent_vectors)
    all_latent_vectors = np.stack(all_latent_vectors)
    
    print("Latent vectors shape:", latent_vectors.shape, 
          "All timesteps shape:", all_latent_vectors.shape)
    
    return latent_vectors, all_latent_vectors

def collapse_cells():
    # ─────── Collapse All Cells Metadata ───────
    # Paste this in the FIRST cell of your notebook, then run it.

    import nbformat
    from pathlib import Path

    # ←─── CHANGE this to your notebook’s filename
    NOTEBOOK_FILENAME = "interpretability.ipynb"

    nb_path = Path(NOTEBOOK_FILENAME)
    if not nb_path.exists():
        raise FileNotFoundError(f"Cannot find {nb_path!r} in the current directory.")

    # Read, modify, and overwrite the notebook
    nb = nbformat.read(nb_path, as_version=4)
    for cell in nb.cells:
        cell.metadata['collapsed'] = True
    nbformat.write(nb, nb_path)

    print(f"✅ All cells in {NOTEBOOK_FILENAME} marked collapsed in metadata.")

if __name__ == "__main__":
    collapse_cells()