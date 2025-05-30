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
from matplotlib.colors import LinearSegmentedColormap

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
    ref = None
    mse_list = []
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
        
        #calculate MSE against reference.
        if ref is None:
            ref = pred_x
        else:
            mse_list.append(torch.nn.MSELoss()(pred_x, ref).cpu().item())
            
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
        
    return sum(mse_list)

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
    # colors_list = plt.colours.tab20(np.linspace(0, 1, n_latent_dims))
    
    blue_red = LinearSegmentedColormap.from_list(
        'blue_red', ['blue','red'], N=n_latent_dims
    )
    colors_list = [blue_red(i) for i in range(n_latent_dims)]
    
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

def run_and_save_inference_all_trajectories(model_params, dataset, save_dir=None, save_individual=True, save_batch=True):
    """
    Run and save inference for every trajectory in the dataset.
    
    Parameters
    ----------
    model_params : dict
        Dictionary containing model parameters and components
        Required keys: 'func', 'rec', 'dec', 'optim', 'device', 'epochs', 'folder'
    dataset : dict
        Dictionary containing dataset components
        Required keys: 'trajs', 'times', 'y'
    save_dir : str, optional
        Directory to save results. If None, uses model_params['folder'] + '/inference_results'
    save_individual : bool, default=True
        Whether to save individual trajectory inference results
    save_batch : bool, default=True
        Whether to save batch inference results
        
    Returns
    -------
    dict
        Dictionary containing inference results for all trajectories
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
    
    # Set up save directory
    if save_dir is None:
        save_dir = model_params['folder'] + '/inference_results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Create inference function
    infer_step = shjnn.make_infer_step(
        model_func, encoder, decoder, optimizer, device, 
        input_mode='traj', sample=False
    )
    
    # Create time points for prediction
    pred_times = np.linspace(0, 2.5, 1000) + 1  # +1 accounts for time bias
    time_tensor = torch.Tensor(pred_times).to(device)
    
    # Storage for results
    all_predictions = []
    all_latent_vectors = []
    all_losses = []
    trajectory_indices = list(range(len(trajectories)))
    
    print(f"Running inference on {len(trajectories)} trajectories...")
    
    # Process each trajectory
    for idx in trajectory_indices:
        # Prepare trajectory tensor
        traj_tensor = trajectories[idx].view(1, *trajectories[idx].size()).to(device)
        
        # Run model inference for prediction
        pred_x, pred_z = infer_step(traj_tensor, time_tensor)
  
        # Convert to numpy for storage
        pred_x_np = pred_x.detach().cpu().numpy()[0]
        pred_z_np = pred_z.detach().cpu().numpy()[0]
        loss_value = loss.item()
        
        # Store results
        all_predictions.append(pred_x_np)
        all_latent_vectors.append(pred_z_np)
        all_losses.append(loss_value)
        
        
    
    # Convert lists to numpy arrays
    all_predictions = np.stack(all_predictions)
    all_latent_vectors = np.stack(all_latent_vectors)
    all_losses = np.array(all_losses)
    
    print(f"Inference complete. Shapes - Predictions: {all_predictions.shape}, Latent: {all_latent_vectors.shape}")
    
    return all_predictions
    

def get_mean_property_plot(model_params, dataset, show=True):
    """
    Get a mapping of mean property values for each sweep axis 'intensity', 'voltage', or 'delay' on 3 different plots.
    """
    val_map = {
            # intensity in uJ
            'intensity': {'source': 'int',
                        'dark': 0., '32uJ': 32., '10uJ': 10., '3uJ': 3., '1uJ': 1., '03uJ': .3},
            # voltage in V
            'voltage': {'source': 'vlt',
                        '05V': .5, '0V': 0., '15V': 1.5, '1V': 1., '2V': 2.},
            # delay time in log10(s)
            'delay': {'source': 'del',
                    '100ns': 1e-7, '100us': 1e-4, '10ms': 1e-2, '10us': 1e-5, '1ms': 1e-3, '1us': 1e-6,
                    '200ns': 2e-7, '200us': 2e-4, '20ms': 2e-2, '20us': 2e-5, '2ms': 2e-3, '2us': 2e-6,
                    '500ns': 5e-7, '500us': 5e-4, '50ms': 5e-2, '50us': 5e-5, '5ms': 5e-3, '5us': 5e-6,
                    },
        }
    
    # Use run_and_save_inference_all_trajectories for comprehensive inference
    all_predictions = run_and_save_inference_all_trajectories(model_params, dataset)
    all_predictions = all_predictions.squeeze()
    # for each value in one sweep axis, get the mean trajectory of all trajectories with that value in the sweep axis.
    mean_map = {
            # intensity in uJ
            'intensity': {
                        0: [], 32: [], 10: [], 3: [], 1: [], 0.3: []},
            # voltage in V
            'voltage': {'source': 'vlt',
                        0.5: [], 0: [], 1.5: [], 1: [], 2: []},
            # delay time in log10(s)
            'delay': {'source': 'del',
                    1e-7: [], 1e-4: [], 1e-2: [], 1e-5: [], 1e-3: [], 1e-6: [],
                    2e-7: [], 2e-4: [], 2e-2: [], 2e-5: [], 2e-3: [], 2e-6: [],
                    5e-7: [], 5e-4: [], 5e-2: [], 5e-5: [], 5e-3: [], 5e-6: []
                    },
        }
    # append 
    for i, meta in enumerate(dataset['y']):
        meta_cpu = meta.cpu().item()
        mean_map['intensity'][meta_cpu[0]].append(all_predictions[i])
        mean_map['voltage'][meta_cpu[1]].append(all_predictions[i])
        mean_map['delay'][meta_cpu[2]].append(all_predictions[i])
    
    # plot each.
    
        
    

if __name__ == "__main__":
    collapse_cells()