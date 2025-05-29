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
import matplotlib.colors as colors
import matplotlib.cm as cmx
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
    

def sweep_latent_adaptive(model_params, dataset, latent_dim_number, latent_vectors, all_latent_vectors, specific_traj_list=None, save=False, show=False):
    """
    Visualize the effect of varying a specific latent dimension.
    
    This function generates predictions by varying the value of a single latent
    dimension while keeping others fixed at their mean values. This helps understand
    what feature each latent dimension encodes.
    
    Parameters
    ----------
    model_params : dict
        Dictionary containing model parameters and components
        Required keys: 'func', 'rec', 'dec', 'optim', 'device', 'epochs', 
                        'latent_dim', 'folder'
    dataset : dict
        Dictionary containing dataset components
        Required keys: 'trajs', 'times', 'y'
    latent_dim_number : int
        Index of the latent dimension to vary
    
    Returns
    -------
    None
        The function creates and saves a plot but doesn't return any values
    """
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
    
    print("Latent vectors shape:", latent_vectors.shape, 
          "All timesteps shape:", all_latent_vectors.shape)

    # Set up figure for latent dimension sweep visualization
    num_dims = trajectories[0].shape[-1]
    fig_width = 7
    fig_height = 4 * num_dims
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Create subplot for plotting the predictions
    ax = fig.add_subplot(1, 1, 1)

    # Create inference function for latent space decoding
    infer_step_decode = shjnn.make_infer_step(
        model_func, encoder, decoder, optimizer, device, 
        input_mode='latent'
    )

    # Define range of values to test for the selected latent dimension
    range_size = 3  # +/- 3 standard deviations
    test_values = np.linspace(-range_size, range_size, 10)

    # Create colormap for the different test values
    color_norm = colors.Normalize(vmin=0, vmax=len(test_values))
    color_map = cmx.ScalarMappable(norm=color_norm, cmap='cividis')

    # Test each value in the range
    for i, test_value in enumerate(test_values):
        # Get color for this test value
        color = color_map.to_rgba(i)

        # Start with the mean latent vector from the dataset
        base_latent = np.expand_dims(np.mean(latent_vectors, 0), 0)
        
        # Modify the target dimension with the test value
        base_latent[..., latent_dim_number] += test_value
        
        # Convert to tensor and move to device
        latent_tensor = torch.Tensor(base_latent).to(device)

        # Create time points tensor
        pred_times = np.linspace(0, 2.5, 1000)
        time_tensor = torch.Tensor(pred_times).to(device)

        # Get model prediction from the modified latent vector
        pred_x, pred_z = infer_step_decode(latent_tensor, time_tensor)

        # Convert prediction to numpy for plotting
        pred_x_np = pred_x.detach().cpu().numpy()[0]

        # Plot the prediction for the first dimension
        label = 'z{}, {:.1f} + {:.1f}'.format(
            latent_dim_number, 
            np.mean(latent_vectors, 0)[latent_dim_number],
            test_value
        )
        ax.plot(pred_times, pred_x_np[:, 0], '-', 
               label=label, alpha=0.6, color=color, linewidth=2)
            
    # Add labels and formatting
    plt.xlabel('Time [10$^{-7}$ + -log$_{10}(t)$ s]')
    plt.ylabel('Charge [mA]')

    # Add horizontal line at y=0
    plt.hlines(0., -.1, 2.6, colors='k', linestyle='--', alpha=0.5)
    plt.xlim(-.1, 2.6)
            
    plt.legend()
    plt.tight_layout()

    # Save the figure
    if show:
        plt.show()
    if save:
        save_dir = model_params['folder'] + '/latent_dims'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig(save_dir + f'/epoch_{epoch_num}_dim_{latent_dim_number}.png', dpi=300)


def latent_trajectory(all_latent_vectors, sample_list=None, show=True, save=False, save_path=None):
    """
    Plot how every latent dimension's mean evolves over timesteps.
    
    Parameters
    ----------
    all_latent_vectors : numpy.ndarray
        Array of shape (n_samples, n_timesteps, n_latent_dims) containing
        latent vectors for all samples and timesteps
    sample_list : list or None, optional
        List of sample indices to include in mean calculation. 
        If None, uses all samples. Default is None.
    show : bool, optional
        Whether to display the plot. Default is True.
    save : bool, optional
        Whether to save the plot. Default is False.
    save_path : str or None, optional
        Path to save the plot. If None and save=True, uses default path.
        
    Returns
    -------
    None
        Creates and optionally displays/saves the plot
    """
    
    # Select samples to include in mean calculation
    if sample_list is None:
        selected_vectors = all_latent_vectors  # Use all samples
    else:
        selected_vectors = all_latent_vectors[sample_list]  # Use specified samples
    
    # Calculate mean across samples for each timestep and latent dimension
    # Shape: (n_timesteps, n_latent_dims)
    mean_latent_trajectory = np.mean(selected_vectors, axis=0)
    
    n_timesteps, n_latent_dims = mean_latent_trajectory.shape
    timesteps = np.arange(n_timesteps)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create colormap for different latent dimensions
    colors_list = plt.cm.tab20(np.linspace(0, 1, n_latent_dims))
    
    # Plot each latent dimension
    for dim in range(n_latent_dims):
        ax.plot(timesteps, mean_latent_trajectory[:, dim], 
               label=f'Latent Dim {dim}', 
               color=colors_list[dim], 
               linewidth=2, 
               alpha=0.8)
    
    # Formatting
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Mean Latent Value')
    ax.set_title('Evolution of Mean Latent Dimensions Over Time')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add info about sample selection
    if sample_list is None:
        sample_info = f"All samples (n={len(all_latent_vectors)})"
    else:
        sample_info = f"Selected samples (n={len(sample_list)})"
    
    ax.text(0.02, 0.98, sample_info, transform=ax.transAxes, 
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save if requested
    if save:
        if save_path is None:
            save_path = 'latent_trajectory_evolution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    # Show if requested
    if show:
        plt.show()
    
    print(f"Plotted {n_latent_dims} latent dimensions over {n_timesteps} timesteps")
    if sample_list is not None:
        print(f"Used samples: {sample_list}")

if __name__ == "__main__":
    collapse_cells()