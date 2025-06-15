import torch
import matplotlib.pyplot as plt
import numpy as np
import time, os, pickle
import pandas as pd
import shap


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
    
def integrate_wrt_time(timestamps, values):
    """
    Numerically integrate a 1-D signal with respect to time.

    Parameters
    ----------
    timestamps : list or 1-D array-like (length N)
        Monotonically increasing time points (e.g. produced by np.linspace).
    values : list or 1-D array-like (length N)
        Signal values sampled at the corresponding timestamps.

    Returns
    -------
    float
        The integral ∫ value(t) dt over the full time span.
    """
    # Convert to NumPy arrays for vectorised operations
    t = np.asarray(timestamps, dtype=float)
    y = np.asarray(values, dtype=float)

    # Basic validity checks
    if t.shape != y.shape:
        raise ValueError("timestamps and values must have the same length.")
    if t.ndim != 1:
        raise ValueError("Inputs must be one-dimensional sequences.")
    if not np.all(np.diff(t) > 0):
        raise ValueError("timestamps must be strictly increasing.")

    # Trapezoidal rule (works for uniform or non-uniform spacing)
    integral = np.trapz(y, t)
    return float(integral)

def steepest_descending_gradient(timestamps, values, window=50):
    """
    Find the steepest negative (descending) gradient that spans a fixed
    number of consecutive samples.

    Parameters
    ----------
    timestamps : 1-D array-like
        Monotonically increasing time points (length N ≥ window+1).
    values : 1-D array-like
        Signal values sampled at the corresponding timestamps.
    window : int, default 50
        Width of the sliding window (in *samples*, not seconds).

    Returns
    -------
    grad_min : float
        The most negative slope over any `window`-wide segment:
        (y[i+window] - y[i]) / (t[i+window] - t[i]).
    start_idx : int
        Index i at which that steepest segment begins.
    end_idx : int
        Index i+window at which that segment ends.
    """
    # Convert to NumPy arrays
    t = np.asarray(timestamps, dtype=float)
    y = np.asarray(values, dtype=float)

    if t.shape != y.shape:
        raise ValueError("timestamps and values must have the same length.")
    if len(t) <= window:
        raise ValueError("Input length must exceed the window size.")
    if not np.all(np.diff(t) > 0):
        raise ValueError("timestamps must be strictly increasing.")

    # Vectorised two-point slope over each window-wide span
    dt = t[window:] - t[:-window]      # size N-window
    dy = y[window:] - y[:-window]      # size N-window
    slopes = dy / dt                   # slope for each window start

    # The steepest *descending* gradient = most negative slope
    start_idx = int(np.argmin(slopes))
    end_idx = start_idx + window
    grad_min = slopes[start_idx]

    return grad_min

def sweep_latent_adaptive(model_params, dataset, latent_dim_number, latent_vectors, all_latent_vectors, specific_traj_list=None, save=False, show=False, show_integrals=False):
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
        scale_factor = 50 * 1e2 / 1e3
        # Plot the prediction for the first dimension
        if show_integrals:
            integral = integrate_wrt_time(pred_times, pred_x_np[:, 0]/scale_factor)
            derivative = steepest_descending_gradient(pred_times, pred_x_np[:, 0]/scale_factor)
            label = 'z{}, {:.1f} + {:.1f}, ∫: {:.1f}, ∇: {:.1f} '.format(
                latent_dim_number, 
                np.mean(latent_vectors, 0)[latent_dim_number],
                test_value,
                integral,
                derivative
            )
        else:
            label = 'z{}, {:.1f} + {:.1f}'.format(
                latent_dim_number, 
                np.mean(latent_vectors, 0)[latent_dim_number],
                test_value
            )
        ax.plot(pred_times, pred_x_np[:, 0] / scale_factor, '-', 
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
        
        # Store results
        all_predictions.append(pred_x_np)
        all_latent_vectors.append(pred_z_np)
        
        
    
    # Convert lists to numpy arrays
    all_predictions = np.stack(all_predictions)
    all_latent_vectors = np.stack(all_latent_vectors)
    
    print(f"Inference complete. Shapes - Predictions: {all_predictions.shape}, Latent: {all_latent_vectors.shape}")
    
    return all_predictions
    
def get_mean_trajectories(model_params, dataset):
    all_predictions = ipynb_utils.run_and_save_inference_all_trajectories(model_params, dataset)
    
    
    all_predictions = all_predictions.squeeze()
    # for each value in one sweep axis, get the mean trajectory of all trajectories with that value in the sweep axis.
    mean_map = {
    'intensity': {0:[], 0.3:[], 1:[], 3:[], 10:[], 32:[]},
    'voltage':   {0:[], 0.5:[], 1:[], 1.5:[], 2:[]},
    'delay':     {
                    0.0000001:[],  # was 1e-7
                    0.000001:[],   # was 1e-6
                    0.00001:[],    # was 1e-5
                    0.0001:[],     # was 1e-4
                    0.001:[],      # was 1e-3
                    0.01:[],       # was 1e-2
                    0.02:[],       # was 2e-2
                    0.05:[]       # was 5e-2
                }
    }
    
    mean_map_original_data = {
    'intensity': {0:[], 0.3:[], 1:[], 3:[], 10:[], 32:[]},
    'voltage':   {0:[], 0.5:[], 1:[], 1.5:[], 2:[]},
    'delay':     {
                    0.0000001:[],  # was 1e-7
                    0.000001:[],   # was 1e-6
                    0.00001:[],    # was 1e-5
                    0.0001:[],     # was 1e-4
                    0.001:[],      # was 1e-3
                    0.01:[],       # was 1e-2
                    0.02:[],       # was 2e-2
                    0.05:[]       # was 5e-2
                }
    }
    
    mean_map_indices = {
    'intensity': {0:[], 0.3:[], 1:[], 3:[], 10:[], 32:[]},
    'voltage':   {0:[], 0.5:[], 1:[], 1.5:[], 2:[]},
    'delay':     {
                    0.0000001:[],  # was 1e-7
                    0.000001:[],   # was 1e-6
                    0.00001:[],    # was 1e-5
                    0.0001:[],     # was 1e-4
                    0.001:[],      # was 1e-3
                    0.01:[],       # was 1e-2
                    0.02:[],       # was 2e-2
                    0.05:[]       # was 5e-2
                }
    }
    
    # append 
    def canonical(x, ndp=7):
        x = x.item()
        return round(float(x), ndp)

    for i, meta in enumerate(dataset['y']):
        meta_cpu = meta.detach().cpu().numpy()
       
        int_key   = canonical(meta_cpu[0])      # 0, 32, 10, …
        volt_key  = canonical(meta_cpu[1])      # 0, 0.5, 1, …
        delay_key = canonical(meta_cpu[2])      # 0.01, 1e-3, …
        
        mean_map['intensity'][int_key].append(all_predictions[i])
        mean_map_indices['intensity'][int_key].append(i)
        mean_map_original_data['intensity'][int_key].append(dataset['trajs'][i].detach().cpu())
        mean_map['voltage'][volt_key].append(all_predictions[i])
        mean_map_indices['voltage'][volt_key].append(i)
        mean_map_original_data['voltage'][volt_key].append(dataset['trajs'][i].detach().cpu())
        mean_map['delay'][delay_key].append(all_predictions[i])
        mean_map_indices['delay'][delay_key].append(i)
        mean_map_original_data['delay'][delay_key].append(dataset['trajs'][i].detach().cpu())
        
    return mean_map, mean_map_indices, mean_map_original_data
      
    

def get_mean_property_plot(model_params, dataset, mean_map, mean_map_orig, show=True):
    """
    Get a mapping of mean property values for each sweep axis 'intensity', 'voltage', or 'delay' on 3 different plots.
    """
    time_points = dataset['times']
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


    
    # plot each.
    cmap = plt.get_cmap('viridis')
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    pred_times = np.linspace(0, 2.5, 1000) + 1  # +1 accounts for time bias
    scale_factor = 50 * 1e2 / 1e3
    orig_time = time_points[0].detach().cpu()
    # for each attribute
    for i, (key, value) in enumerate(mean_map.items()):
        ax = axs[i]
        n_groups = len(value)
        colors = cmap(np.linspace(0, 1, n_groups))
        # for each value in the attribute
        for j, (sub_key, trajectories) in enumerate(value.items()):
            if len(trajectories) > 0:
                mean_trajectory = np.mean(trajectories, axis=0)
                # print("trajectory: ", trajectories.shape)
                # print(mean_trajectory.shape)
                ax.plot(
                    pred_times - 1,
                    mean_trajectory / scale_factor,
                    label=f"{sub_key} {val_map[key]['source']}",
                    color=colors[j],        # explicit color
                    linewidth=2,
                    alpha=0.8
                )
                mean_orig_trajectory = np.mean(mean_map_orig[key][sub_key], axis=0)
                ax.plot(
                    orig_time,
                    mean_orig_trajectory / scale_factor,
                    '.',
                    color=colors[j],
                    linewidth=1,
                    alpha=0.5
                )
        
        ax.set_title(f"Mean Trajectories for {key.capitalize()}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Charge [mA]")
        ax.legend()
        
def latent_means_for_parameter(mean_map_indices, latent):
    # plot means of latent dimensions for each parameter value
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    intensity_dict = {}
    for (key, intensities) in mean_map_indices['intensity'].items():
        latent_total = []
        if len(intensities) == 0:
            continue
        for idx in intensities:
            latent_total.append(latent[idx, :])
        latent_mean = np.array(latent_total).mean(axis=0)
        intensity_dict[key] = latent_mean

    filtered = {k: v for k, v in intensity_dict.items() if v is not []}
    keys = sorted(filtered.keys())
    series_count = max(len(v) for v in filtered.values())
    for i in range(series_count):
        xs = []
        ys = []
        for k in keys:
            values = filtered[k]
            # Ensure the current series index exists in the list
            if len(values) > i:
                xs.append(k)
                ys.append(values[i])
        axes[0].plot(xs, ys, marker='o', label=f'Latent Dim {i}')
    axes[0].set_title('Latent Mean by Intensity')
    axes[0].set_xlabel('Intensity')
    axes[0].set_ylabel('Latent Mean Value')
    axes[0].legend()


    voltage_dict = {}
    for (key, voltages) in mean_map_indices['voltage'].items():
        latent_total = []
        if len(voltages) == 0:
            continue
        for idx in voltages:
            latent_total.append(latent[idx, :])
        latent_mean = np.array(latent_total).mean(axis=0)
        voltage_dict[key] = latent_mean
    filtered = {k: v for k, v in voltage_dict.items() if v is not []}
    keys = sorted(filtered.keys())
    series_count = max(len(v) for v in filtered.values())
    for i in range(series_count):
        xs = []
        ys = []
        for k in keys:
            values = filtered[k]
            # Ensure the current series index exists in the list
            if len(values) > i:
                xs.append(k)
                ys.append(values[i])
        axes[1].plot(xs, ys, marker='o', label=f'Latent Dim {i}')
    axes[1].set_title('Latent Mean by Voltage')
    axes[1].set_xlabel('Voltage')
    axes[1].set_ylabel('Latent Mean Value')
    axes[1].legend()

    delay_dict = {}
    for (key, delays) in mean_map_indices['delay'].items():
        latent_total = []
        if len(delays) == 0:
            continue
        for idx in delays:
            latent_total.append(latent[idx, :])
        latent_mean = np.array(latent_total).mean(axis=0)
        delay_dict[key] = latent_mean
    filtered = {k: v for k, v in delay_dict.items() if v is not []}
    keys = sorted(filtered.keys())
    series_count = max(len(v) for v in filtered.values())
    for i in range(series_count):
        xs = []
        ys = []
        for k in keys:
            values = filtered[k]
            # Ensure the current series index exists in the list
            if len(values) > i:
                xs.append(k)
                ys.append(values[i])
        axes[2].plot(xs, ys, marker='o', label=f'Latent Dim {i}')
    axes[2].set_title('Latent Mean by Delay')
    axes[2].set_xlabel('Delay')
    axes[2].set_xscale('log')
    axes[2].set_ylabel('Latent Mean Value')
    axes[2].legend()





def plot_shap_analysis(shap_values, feature_importance, model_params, 
                      save=True, show=True):
    """
    Create comprehensive SHAP visualization plots.
    
    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP values from the analysis
    feature_importance : numpy.ndarray
        Mean absolute SHAP values for each latent dimension
    model_params : dict
        Dictionary containing model parameters
    save : bool, default=True
        Whether to save the plots
    show : bool, default=True
        Whether to display the plots
    """
    
    epoch_num = model_params['epochs']
    folder = model_params['folder']
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Feature importance bar plot
    ax1 = plt.subplot(2, 3, 1)
    latent_dim_names = [f'Latent Dim {i}' for i in range(len(feature_importance))]
    bars = ax1.bar(range(len(feature_importance)), feature_importance, 
                   color='steelblue', alpha=0.7)
    ax1.set_xlabel('Latent Dimension')
    ax1.set_ylabel('Mean |SHAP Value|')
    ax1.set_title('Latent Dimension Importance')
    ax1.set_xticks(range(len(feature_importance)))
    ax1.set_xticklabels([f'Z{i}' for i in range(len(feature_importance))], 
                       rotation=45)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, feature_importance)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 2. SHAP summary plot (beeswarm plot)
    ax2 = plt.subplot(2, 3, 2)
    plt.sca(ax2)
    shap.plots.beeswarm(shap_values[:, :, 0], max_display=len(feature_importance), 
                       show=False)
    ax2.set_title('SHAP Values Distribution (First Time Point)')
    
    # 3. SHAP waterfall plot for a representative sample
    ax3 = plt.subplot(2, 3, 3)
    plt.sca(ax3)
    # Use the first sample for waterfall plot
    shap.plots.waterfall(shap_values[0, :, 0], max_display=len(feature_importance), 
                        show=False)
    ax3.set_title('SHAP Waterfall (Sample 0, t=0)')
    
    # 4. Heatmap of SHAP values across time for top dimensions
    ax4 = plt.subplot(2, 3, 4)
    # Get top 5 most important dimensions
    top_dims = np.argsort(feature_importance)[-5:]
    
    # Average SHAP values across samples for visualization
    mean_shap_over_time = np.mean(np.abs(shap_values.values), axis=0)
    heatmap_data = mean_shap_over_time[top_dims, ::50]  # Subsample time points
    
    im = ax4.imshow(heatmap_data, aspect='auto', cmap='RdBu_r', 
                   interpolation='nearest')
    ax4.set_xlabel('Time Point (subsampled)')
    ax4.set_ylabel('Latent Dimension')
    ax4.set_title('SHAP Values Over Time (Top 5 Dimensions)')
    ax4.set_yticks(range(len(top_dims)))
    ax4.set_yticklabels([f'Z{i}' for i in top_dims])
    plt.colorbar(im, ax=ax4, label='Mean |SHAP Value|')
    
    # 5. Correlation between latent dimensions and output variance
    ax5 = plt.subplot(2, 3, 5)
    # Calculate variance of SHAP values for each dimension across samples
    shap_variance = np.var(shap_values.values, axis=(0, 2))
    ax5.scatter(feature_importance, shap_variance, alpha=0.7, s=50)
    ax5.set_xlabel('Mean |SHAP Value|')
    ax5.set_ylabel('SHAP Value Variance')
    ax5.set_title('Importance vs Variability')
    
    # Add labels for each point
    for i, (imp, var) in enumerate(zip(feature_importance, shap_variance)):
        ax5.annotate(f'Z{i}', (imp, var), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8)
    
    # 6. Top dimensions detailed analysis
    ax6 = plt.subplot(2, 3, 6)
    top_3_dims = np.argsort(feature_importance)[-3:]
    
    for i, dim in enumerate(top_3_dims):
        dim_shap_over_time = np.mean(shap_values.values[:, dim, :], axis=0)
        time_points = np.linspace(0, 2.5, len(dim_shap_over_time))
        ax6.plot(time_points, dim_shap_over_time, 
                label=f'Z{dim} (imp: {feature_importance[dim]:.3f})',
                linewidth=2, alpha=0.8)
    
    ax6.set_xlabel('Time [log scale]')
    ax6.set_ylabel('Mean SHAP Value')
    ax6.set_title('SHAP Evolution for Top 3 Dimensions')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    if save:
        save_dir = folder + '/shap_analysis'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig(save_dir + f'/shap_analysis_epoch_{epoch_num}.png', 
                   dpi=300, bbox_inches='tight')
        print(f"SHAP analysis plots saved to: {save_dir}")
    
    # Show the plot
    if show:
        plt.show()
    
    # Print summary statistics
    print("\n=== SHAP Analysis Summary ===")
    print(f"Most important latent dimensions:")
    importance_ranking = np.argsort(feature_importance)[::-1]
    for i, dim_idx in enumerate(importance_ranking[:5]):
        print(f"  {i+1}. Latent Dim {dim_idx}: {feature_importance[dim_idx]:.4f}")
    
    print(f"\nTotal explained variance: {np.sum(feature_importance):.4f}")
    print(f"Top 3 dimensions explain {np.sum(feature_importance[importance_ranking[:3]]):.1%} of importance")

def shap_analysis(model_params, dataset, latent_vectors, 
                                        background_size=100, num_samples=150, 
                                        save=True, show=True):
    """
    Use SHAP to analyze how each latent dimension relates to the model's output.
    
    This function creates a wrapper around the decoder to enable SHAP analysis,
    calculates SHAP values for each latent dimension, and creates visualizations
    showing which latent dimensions are most important for the model's predictions.
    
    Parameters
    ----------
    model_params : dict
        Dictionary containing model parameters and components
        Required keys: 'func', 'rec', 'dec', 'optim', 'device', 'epochs', 'folder'
    dataset : dict
        Dictionary containing dataset components
    latent_vectors : numpy.ndarray
        Array of latent vectors from the encoded trajectories
    background_size : int, default=100
        Number of background samples for SHAP analysis
    num_samples : int, default=200
        Number of samples to analyze with SHAP
    save : bool, default=True
        Whether to save the plots
    show : bool, default=True
        Whether to display the plots
        
    Returns
    -------
    shap_values : numpy.ndarray
        SHAP values for each latent dimension
    feature_importance : numpy.ndarray
        Mean absolute SHAP values for each latent dimension
    """
    
    # Extract model components
    model_func = model_params['func']
    encoder = model_params['rec']
    decoder = model_params['dec']
    optimizer = model_params['optim']
    device = model_params['device']
    epoch_num = model_params['epochs']
    latent_dims = model_params['latent_dim']
    
    # Create inference function for latent space decoding
    infer_step_decode = shjnn.make_infer_step(
        model_func, encoder, decoder, optimizer, device, 
        input_mode='latent'
    )
    
    # Create time points for prediction
    pred_times = np.linspace(0, 2.5, 1000)
    time_tensor = torch.Tensor(pred_times).to(device)
    
    def model_wrapper(latent_input):
        """
        Wrapper function for the decoder that SHAP can use.
        
        Parameters
        ----------
        latent_input : numpy.ndarray
            Batch of latent vectors to decode
            
        Returns
        -------
        numpy.ndarray
            Model predictions as numpy array
        """
        predictions = []
        
        for latent_vec in latent_input:
            # Convert to tensor and move to device
            latent_tensor = torch.Tensor(latent_vec.reshape(1, -1)).to(device)
            
            # Get model prediction from latent vector
            pred_x, pred_z = infer_step_decode(latent_tensor, time_tensor)
            
            # Convert to numpy and extract first dimension (charge)
            pred_x_np = pred_x.detach().cpu().numpy()[0, :, 0]  # Shape: (time_steps,)
            predictions.append(pred_x_np)
        
        return np.array(predictions)
    
    # Prepare data for SHAP analysis
    # Use subset of latent vectors for efficiency
    analysis_indices = np.random.choice(len(latent_vectors), 
                                      min(num_samples, len(latent_vectors)), 
                                      replace=False)
    analysis_latents = latent_vectors[analysis_indices]
    
    # Create background dataset for SHAP
    background_indices = np.random.choice(len(latent_vectors), 
                                        min(background_size, len(latent_vectors)), 
                                        replace=False)
    background_latents = latent_vectors[background_indices]
    
    print(f"Running SHAP analysis on {len(analysis_latents)} samples...")
    print(f"Using {len(background_latents)} background samples...")
    print(f"Latent dimension size: {latent_dims}")
    
    # Create SHAP explainer
    explainer = shap.Explainer(model_wrapper, background_latents)
    
    # Calculate SHAP values
    shap_values = explainer(analysis_latents)
    
    # Get feature importance (mean absolute SHAP values)
    feature_importance = np.mean(np.abs(shap_values.values), axis=(0, 2))
    
    print(f"SHAP analysis complete. Shape of SHAP values: {shap_values.values.shape}")
    print(f"Feature importance shape: {feature_importance.shape}")
    
    # Create visualizations
    plot_shap_analysis(shap_values, feature_importance, model_params, 
                      save=save, show=show)
    
    return shap_values, feature_importance

def example_shap_usage():
    """
    Example usage of the SHAP analysis function.
    
    This function demonstrates how to:
    1. Load a trained model and dataset
    2. Extract latent vectors
    3. Run SHAP analysis
    4. Interpret the results
    
    Usage (in a Jupyter notebook):
    ```python
    import ipynb_utils
    
    # Assuming you have model_params and dataset already loaded
    latent_vectors, all_latent_vectors = ipynb_utils.get_latent_vectors(model_params, dataset)
    
    # Run SHAP analysis
    shap_values, feature_importance = ipynb_utils.analyze_latent_dimensions_with_shap(
        model_params, dataset, latent_vectors, 
        background_size=50,  # Use smaller values for faster computation
        num_samples=100,     # Adjust based on your dataset size
        save=True,          # Save plots to model folder
        show=True           # Display plots in notebook
    )
    
    # The analysis will:
    # 1. Create a comprehensive plot with 6 subplots showing different aspects of SHAP analysis
    # 2. Print summary statistics about which latent dimensions are most important
    # 3. Return shap_values and feature_importance for further analysis
    
    # Example of further analysis:
    print("Most important latent dimensions:")
    importance_ranking = np.argsort(feature_importance)[::-1]
    for i, dim_idx in enumerate(importance_ranking[:3]):
        print(f"  Latent Dim {dim_idx}: {feature_importance[dim_idx]:.4f}")
    ```
    
    The SHAP analysis will create visualizations showing:
    - Bar plot of feature importance for each latent dimension
    - Beeswarm plot showing distribution of SHAP values
    - Waterfall plot for a single sample showing individual contributions
    - Heatmap of SHAP values over time for top dimensions
    - Scatter plot of importance vs variability
    - Time evolution plot for the top 3 most important dimensions
    """
    print("This is an example function showing how to use SHAP analysis.")
    print("See the docstring for detailed usage instructions.")
    
    print("\n=== Quick Start Example ===")
    print("# In your Jupyter notebook:")
    print("latent_vectors, _ = ipynb_utils.get_latent_vectors(model_params, dataset)")
    print("shap_values, importance = ipynb_utils.analyze_latent_dimensions_with_shap(")
    print("    model_params, dataset, latent_vectors)")

def extract_linear_map(infer_step_decode, model_params, time_tensor):
    device = model_params['device']
    D = model_params['latent_dim']
    # 1) Get the bias by passing all-zeros through
    z0 = torch.zeros((1, D), device=device)
    y0, _ = infer_step_decode(z0, time_tensor)
    b = y0.squeeze()[0].item()         # scalar bias

    # 2) For each latent dim i, set that dim = 1 and compute output
    weights = torch.zeros(D, device=device)
    bias = torch.tensor(b, device=device)  # ensure bias is a tensor for subtraction
    for i in range(D):
        zi = torch.zeros((1, D), device=device)
        zi[0, i] = 1.0
        yi, _ = infer_step_decode(zi, time_tensor)
        weights[i] = yi.squeeze()[0] - b

    return weights.cpu().numpy(), b



if __name__ == "__main__":
    collapse_cells()