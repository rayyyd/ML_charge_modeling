"""
Visualization module for plotting and analyzing ML charge modeling results.

This module provides functions for visualizing training results, model predictions,
and latent space characteristics for variational autoencoder models.
"""

import os
import logging
import glob
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.animation as animation
from PIL import Image
from scipy.signal import savgol_filter

import sys
sys.path.append('../../libs/')
import parameters
import shjnn


def plot_training_loss(model_params, save=False, split=False, plot_total=False, plot_MSE=True, plot_KL=True, scale='log'):
    """
    Plot training loss curves from model training history.
    
    This function visualizes the training loss over epochs, optionally showing different
    loss components (MSE, KL, total) on separate axes. Can produce either a combined
    or split view of the loss components.
    
    Parameters
    ----------
    model_params : dict
        Dictionary containing model parameters and training history
        Required keys: 'MSE_loss', 'KL_loss', 'loss', 'epochs', 'folder'
    save : bool, default=False
        Whether to save the plot to disk
    split : bool, default=False
        If True, plot MSE and KL loss on separate axes
    plot_total : bool, default=False
        If True, plot the total loss (MSE + KL)
    plot_MSE : bool, default=True
        If True, plot the MSE loss component
    plot_KL : bool, default=True
        If True, plot the KL loss component
    scale : str, default='log'
        Scale for y-axis, either 'log' or 'linear'
    
    Returns
    -------
    None
        The function creates and optionally saves a plot but doesn't return any values
    """
    if split:
        mse_loss = model_params['MSE_loss']
        kl_loss = model_params['KL_loss']
        total_loss = model_params['loss']
        print(f"Debug: total_loss shape: {len(total_loss)} at epoch {model_params['epochs']}")
        
        # Set up figure dimensions
        fig_width = 5
        fig_height = 2
        fig = plt.figure(figsize=(fig_width, fig_height))

        # Create main subplot
        main_ax = fig.add_subplot(1, 1, 1)

        # Create epoch numbers for x-axis - scale by 10 to fix epoch display
        # The issue is that the loss is recorded every 0.1 epochs (10 times per epoch)
        epochs = np.arange(1, len(mse_loss) + 1) * model_params['epochs_per_train']
        
        # Plot MSE loss with red color on the primary y-axis
        if plot_MSE:
            main_ax.plot(epochs, mse_loss.detach().cpu().numpy() if isinstance(mse_loss, torch.Tensor) else mse_loss, '-', label='MSE loss', alpha=0.3, color='red')
            main_ax.set_yscale('log')
            main_ax.set_ylabel('MSE Loss')

        # Plot total loss with blue color on a secondary y-axis
        if plot_total:
            total_ax = main_ax.twinx()
            total_ax.plot(epochs, total_loss.detach().cpu().numpy() if isinstance(total_loss, torch.Tensor) else total_loss, '-', label='eval loss', alpha=0.3, color='blue')
            total_ax.set_yscale('log')
            total_ax.set_ylabel('Total Loss')

        # Plot KL loss with green color on another secondary y-axis
        if plot_KL:
            kl_ax = main_ax.twinx()
            kl_ax.plot(epochs, kl_loss.detach().cpu().numpy() if isinstance(kl_loss, torch.Tensor) else kl_loss, '-', label='KL loss', alpha=0.3, color='green')
            kl_ax.set_ylabel('KL Loss')

        # Optional: Plot smoothed losses using Savitzky-Golay filter
        # window_size = 13  # must be odd and less than data length
        # poly_order = 3    # polynomial order for the filter
        # smoothed_mse = np.abs(savgol_filter(mse_loss, window_size, poly_order))
        # smoothed_kl = np.abs(savgol_filter(kl_loss, window_size, poly_order))
        # main_ax.plot(epochs, smoothed_mse, '-', alpha=0.8, color='red')
        # kl_ax.plot(epochs, smoothed_kl, '-', alpha=0.8, color='green')

        # Labels and legend
        main_ax.set_xlabel('Epochs')
        main_ax.set_ylabel('Losses')

        # Add legends to appropriate axes
        main_ax.legend(loc='upper left')
        if plot_KL:
            kl_ax.legend(loc='upper right')
        if plot_total:
            total_ax.legend(loc='upper center')
        
        # Add the final loss value to the title
        plt.title(f"Final Loss: {round(float(total_loss[-1].detach().cpu().numpy()) if isinstance(total_loss[-1], torch.Tensor) else round(total_loss[-1]))}")
        plt.tight_layout()

        if save:
            save_dir = f"{model_params['folder']}/loss_graph/"
            print("Saving to:", save_dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(f"{save_dir}/loss_graph_split.png")
    else:
        # Get the main loss
        loss_history = model_params['loss']

        # Set up figure dimensions
        fig_width = 5
        fig_height = 2
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # Create subplot
        ax = fig.add_subplot(1, 1, 1)

        epochs = np.arange(1, len(loss_history) + 1) * model_params['epochs_per_train']
        
        # Plot raw loss values
        ax.plot(epochs, loss_history.detach().cpu().numpy() if isinstance(loss_history, torch.Tensor) else loss_history, '-', label='Raw loss', alpha=0.3, color='blue')

        # Apply Savitzky-Golay filter to smooth the loss curve
        # Window size 13 (must be odd), polynomial order 3
        # smoothed_loss = np.abs(savgol_filter(loss_history, 3, 3))
        # ax.plot(epochs, smoothed_loss, '-', label='Smoothed loss', alpha=0.8, color='blue')

        # Set y-axis scale (log or linear)
        if scale == 'linear':
            plt.yscale('linear')
        else:
            plt.yscale('log')
        
        # Labels and formatting
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.yticks([0.1, 1, 10, 100, 1000], ['0.1', '1', '10', '100', '1000'])
        plt.title('Training Loss')
        plt.legend()
        plt.tight_layout()

        if save:
            save_dir = f"{model_params['folder']}/loss_graph/"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(f"{save_dir}/loss_graph.png")


def display_random_fit(model_params=parameters.model_params, dataset=parameters.dataset, show=True, save=False, random_samples=True):
    """
    Display model fit on random samples from the dataset.
    
    This function selects random trajectory samples, runs model inference on them,
    and plots both the original data and model predictions for visual comparison.
    
    Parameters
    ----------
    model_params : dict
        Dictionary containing model parameters and components
        Required keys: 'func', 'rec', 'dec', 'optim', 'device', 'epochs', 'lr', 'beta', 'folder'
    dataset : dict
        Dictionary containing dataset components
        Required keys: 'trajs', 'times', 'y'
    show : bool, default=True
        Whether to display the plot interactively
    save : bool, default=False
        Whether to save the plot to disk
    random_samples : bool, default=True
        If True, randomly sample trajectories; otherwise use sequential samples
    
    Returns
    -------
    None
        The function creates and optionally saves a plot but doesn't return any values
    """
    # Extract data from dataset - use original data without missing points
    trajectories = dataset['trajs']  # Original data (with no missing points)
    time_points = dataset['times']   # Original time points
    metadata = dataset['y']          # Contains parameters like intensity, bias, delay
    
    # print(f"Debug: B-VAE: visualisation: display_random: trajectories shape: {trajectories.shape}")
    # print(f"Debug: B-VAE: visualisation: display_random: time_points shape: {time_points.shape}")

    # Extract model components
    model_func = model_params['func']
    encoder = model_params['rec']
    decoder = model_params['dec']
    optimizer = model_params['optim']
    device = model_params['device']
    epoch_num = model_params['epochs']
    
    # Set up figure with subplots (one per trajectory dimension)
    num_dims = trajectories[0].shape[-1]
    fig_width = 7
    fig_height = 4 * num_dims
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Create a subplot for each dimension
    axes = [fig.add_subplot(num_dims, 1, i+1) for i in range(num_dims)]

    # Create inference function from the trained model
    infer_step = shjnn.make_infer_step(
        model_func, encoder, decoder, optimizer, device, 
        input_mode='traj', sample=False
    )

    # Select indices of trajectories to plot
    sample_indices = list(range(len(trajectories)))
    if random_samples:
        random.shuffle(sample_indices)
        # Downsample to avoid cluttering the plot
        sample_indices = sample_indices[::30]
    else:
        sample_indices = model_params['plot']

    # Create colormap for different trajectories
    color_norm = colors.Normalize(vmin=0, vmax=len(sample_indices))
    color_map = cmx.ScalarMappable(norm=color_norm, cmap='brg')
    
    # Iterate over selected trajectories
    loss_list = []
    for idx, traj_idx in enumerate(sample_indices):
        # Get color for this trajectory
        color = color_map.to_rgba(idx)
        
        # Prepare input trajectory tensor
        traj_tensor = trajectories[traj_idx].view(1, *trajectories[traj_idx].size()).to(device)
        # print(f"Debug: B-VAE: visualisation: display_random: traj_tensor shape: {traj_tensor.shape}")
        # Create time points for prediction (denser than original data)
        pred_times = np.linspace(0, 2.5, 1000) + 1  # +1 accounts for time bias
        time_tensor = torch.Tensor(pred_times).to(device)

        # Run model inference
        pred_x, pred_z = infer_step(traj_tensor, time_tensor)
        
        #Run model inference for loss reasons only
        loss_x, _ = infer_step(traj_tensor, time_points[traj_idx].view(1, *time_points[traj_idx].size()).to(device))
        
        # TODO: how do i calculate the loss of the MSE here??
        
        loss = torch.nn.MSELoss()(loss_x.to(device), traj_tensor.to(device))
        loss_list.append(loss.item())
        # print(f"Debug: B-VAE: visualisation: display_random: pred_x shape: {pred_x.shape}")

        # Convert prediction to numpy for plotting
        pred_x_np = pred_x.detach().cpu().numpy()[0]
        
        # Get original trajectory and time data
        orig_traj = trajectories[traj_idx].detach().cpu()
        orig_time = time_points[traj_idx].detach().cpu()
        # print(f"Debug: B-VAE: visualisation: display_random: orig_traj shape: {orig_traj.shape}")
        # Scaling factor for better visualization
        scale_factor = 50 * 1e2 / 1e3
        
        # Plot each dimension
        for dim in range(num_dims):
            # Plot original trajectory points
            axes[dim].plot(orig_time, orig_traj[:, dim] / scale_factor, 
                      '.', alpha=0.6, color=color)
            
            # Plot model prediction
            axes[dim].plot(pred_times - 1.0, pred_x_np[:, dim] / scale_factor, 
                      '-', linewidth=2, alpha=0.4, color=color,
                      label='{:.1f} J$, {:.1f} V, {:.0e} s'.format(
                          metadata[traj_idx][0], metadata[traj_idx][1], metadata[traj_idx][2]))
            
    # Add labels and title
    plt.xlabel('Time [10$^{-7}$ + -log$_{10}(t)$ s]')
    plt.ylabel('Charge [mA]')
    plt.title('Epoch: {}, lr: {:.1e}, beta: {:.1e}, loss: {:.1e}'.format(
        epoch_num, model_params['lr'], model_params['beta'], np.mean([l for l in loss_list])))

    # Add legend
    plt.legend(loc='upper right', title='Intensity, Bias, Delay')
    plt.tight_layout()

    # Show or save the figure
    if show:
        plt.show()
    if save:
        save_dir = model_params['folder']
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig(f"{save_dir}/training_epoch_{epoch_num:04d}.png", dpi=300)


def compile_learning_gif(model_params=parameters.model_params, display=True):
    """
    Compile a GIF animation from training epoch visualizations.
    
    This function reads PNG files saved during training and compiles them into an
    animated GIF to visualize the model's learning progress over time.
    
    Parameters
    ----------
    model_params : dict
        Dictionary containing model parameters
        Required key: 'folder' - directory where training visualizations are stored
    display : bool, default=True
        Whether to display the animation (currently not implemented)
    
    Returns
    -------
    None
        The function creates and saves a GIF animation but doesn't return any values
    """
    # Get directory containing the visualization images
    image_dir = model_params['folder']
    glob_pattern = image_dir + '/*.png'
    print("Looking for images in:", glob_pattern)
    
    # Get sorted list of image files
    image_files = sorted(glob.glob(glob_pattern))
    image_list = []

    # Read all images
    for image_file in image_files:
        image = Image.open(image_file)
        image_list.append(image)

    print('Animation will contain', len(image_list), 'frames')

    # Create figure for animation
    fig, ax = plt.subplots()
    
    # Set initial image
    animation_image = ax.imshow(image_list[0], animated=True)

    # Animation update function
    def update_frame(frame_idx):
        animation_image.set_array(image_list[frame_idx])
        return animation_image, 

    # Create the animation
    anim = animation.FuncAnimation(
        fig, update_frame, frames=len(image_list),
        interval=200,  # milliseconds between frames
        blit=True, 
        repeat_delay=5000,  # delay before looping
    )

    # Create output directory if it doesn't exist
    gif_dir = image_dir + "/gifs/"
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)
        
    # Save the animation as a GIF
    anim.save(gif_dir + "learning_pattern.gif", writer="pillow")

    # TODO: Implement HTML5 video display if needed
    # if display:
    #     from IPython import display
    #     video = anim.to_html5_video()
    #     html = display.HTML(video)
    #     display.display(html)


def sweep_latent_adaptives(model_params=parameters.model_params, dataset=parameters.dataset, do_save=False, do_show=False):
    """
    Generate visualizations for all latent dimensions.
    
    This function iterates through all latent dimensions of the model and
    creates a visualization for each one showing how it affects the output.
    
    Parameters
    ----------
    model_params : dict
        Dictionary containing model parameters
        Required key: 'latent_dim' - number of latent dimensions in the model
    dataset : dict
        Dictionary containing dataset components
    
    Returns
    -------
    None
        The function delegates to sweep_latent_adaptive for each dimension
    """
    for dim_idx in range(model_params['latent_dim']):
        sweep_latent_adaptive(model_params, dataset, dim_idx, do_save, do_show)


def sweep_latent_adaptive(model_params, dataset, latent_dim_number, specific_traj_list=None, save=False, show=False):
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
    infer_step_encode = shjnn.make_infer_step(
        model_func, encoder, decoder, optimizer, device, 
        input_mode='traj', sample=False
    )

    # Get indices of all trajectories
    if specific_traj_list is not None:
        sample_indices = list(range(len(trajectories)))
    else:
        sample_indices = specific_traj_list

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


def time_evolved_sweep(model_params, dataset, latent_dim_number, timesteps_interest, traj_idx=None, save=False, show=False):
    """
    Visualize the effect of varying a specific latent dimension over time.
    
    This function generates predictions by varying the value of a single latent
    dimension while keeping others fixed at their mean values. It shows how the
    output evolves over time for different values of the latent dimension.
    
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
    timesteps_interest : list
        List of timestep indices to visualize
    traj_idx : list, optional
        List of trajectory indices to use for encoding. If None, uses all trajectories
    save : bool, default=False
        Whether to save the plots to disk
    show : bool, default=False
        Whether to display the plots
    
    Returns
    -------
    None
        The function creates and optionally saves plots but doesn't return any values
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

    # Create inference function for latent space decoding
    infer_step_decode = shjnn.make_infer_step(
        model_func, encoder, decoder, optimizer, device, 
        input_mode='latent'
    )

    # Define range of values to test for the selected latent dimension
    range_size = 3  # +/- 3 standard deviations
    test_values = np.linspace(-range_size, range_size, 10)

    # Create colormap for the different test values
    import matplotlib.colors as colors
    import matplotlib.cm as cmx
    color_norm = colors.Normalize(vmin=0, vmax=len(test_values))
    color_map = cmx.ScalarMappable(norm=color_norm, cmap='cividis')

    # Set up figure with subplots for each timestep of interest
    num_timesteps = len(timesteps_interest)
    fig_width = 7 * min(3, num_timesteps)  # Limit width for readability
    fig_height = 4 * ((num_timesteps + 2) // 3)  # Arrange in rows of 3
    
    fig, axes = plt.subplots(
        nrows=((num_timesteps + 2) // 3), 
        ncols=min(3, num_timesteps), 
        figsize=(fig_width, fig_height)
    )
    
    # Handle case where there's only one subplot
    if num_timesteps == 1:
        axes = [axes]
    elif num_timesteps <= 3:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    else:
        axes = axes.flatten()

    # Create time points for prediction
    pred_times = np.linspace(0, 2.5, 1000)
    time_tensor = torch.Tensor(pred_times).to(device)

    # Process each timestep of interest
    for timestep_idx, timestep in enumerate(timesteps_interest):
        ax = axes[timestep_idx] if len(axes) > timestep_idx else axes[0]
        
        # Test each value in the range
        for i, test_value in enumerate(test_values):
            # Get color for this test value
            color = color_map.to_rgba(i)

            # Start with the mean latent vector from the dataset at the specific timestep
            if timestep < all_latent_vectors.shape[1]:
                base_latent = np.expand_dims(np.mean(all_latent_vectors[:, timestep, :], 0), 0)
            else:
                # Fallback to first timestep if timestep is out of bounds
                base_latent = np.expand_dims(np.mean(latent_vectors, 0), 0)
                print(f"Warning: Timestep {timestep} out of bounds, using first timestep")
            
            # Modify the target dimension with the test value
            base_latent[..., latent_dim_number] += test_value
            
            # Convert to tensor and move to device
            latent_tensor = torch.Tensor(base_latent).to(device)

            # Get model prediction from the modified latent vector
            pred_x, pred_z = infer_step_decode(latent_tensor, time_tensor)

            # Convert prediction to numpy for plotting
            pred_x_np = pred_x.detach().cpu().numpy()[0]

            # Plot the prediction for the first dimension
            label = 'z{}, {:.1f} + {:.1f}'.format(
                latent_dim_number, 
                base_latent[0, latent_dim_number] - test_value,  # Original value before modification
                test_value
            )
            ax.plot(pred_times, pred_x_np[:, 0], '-', 
                   label=label, alpha=0.6, color=color, linewidth=2)
        
        # Add labels and formatting for each subplot
        ax.set_xlabel('Time [10$^{-7}$ + -log$_{10}(t)$ s]')
        ax.set_ylabel('Charge [mA]')
        ax.set_title(f'Timestep {timestep}')

        # Add horizontal line at y=0
        ax.hlines(0., -.1, 2.6, colors='k', linestyle='--', alpha=0.5)
        ax.set_xlim(-.1, 2.6)
        
        # Add legend to first subplot only to avoid clutter
        if timestep_idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Hide empty subplots
    for i in range(num_timesteps, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    # Show the figure
    if show:
        plt.show()
    
    # Save the figure with subfolder structure
    if save:
        # Create main save directory
        save_dir = os.path.join(model_params['folder'], 'time_evolved_latent_dims')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Create subfolder for this latent dimension
        dim_save_dir = os.path.join(save_dir, f'latent_dim_{latent_dim_number}')
        if not os.path.exists(dim_save_dir):
            os.makedirs(dim_save_dir)
        
        # Save the combined plot
        fig.savefig(
            os.path.join(dim_save_dir, f'epoch_{epoch_num}_dim_{latent_dim_number}_timesteps_{"_".join(map(str, timesteps_interest))}.png'), 
            dpi=300, bbox_inches='tight'
        )
        
        # Optionally save individual timestep plots
        for timestep_idx, timestep in enumerate(timesteps_interest):
            # Create individual plot for this timestep
            individual_fig, individual_ax = plt.subplots(figsize=(7, 4))
            
            # Recreate the plot for this specific timestep
            for i, test_value in enumerate(test_values):
                color = color_map.to_rgba(i)
                
                if timestep < all_latent_vectors.shape[1]:
                    base_latent = np.expand_dims(np.mean(all_latent_vectors[:, timestep, :], 0), 0)
                else:
                    base_latent = np.expand_dims(np.mean(latent_vectors, 0), 0)
                
                base_latent[..., latent_dim_number] += test_value
                latent_tensor = torch.Tensor(base_latent).to(device)
                pred_x, pred_z = infer_step_decode(latent_tensor, time_tensor)
                pred_x_np = pred_x.detach().cpu().numpy()[0]
                
                label = 'z{}, {:.1f} + {:.1f}'.format(
                    latent_dim_number, 
                    base_latent[0, latent_dim_number] - test_value,
                    test_value
                )
                individual_ax.plot(pred_times, pred_x_np[:, 0], '-', 
                                 label=label, alpha=0.6, color=color, linewidth=2)
            
            individual_ax.set_xlabel('Time [10$^{-7}$ + -log$_{10}(t)$ s]')
            individual_ax.set_ylabel('Charge [mA]')
            individual_ax.set_title(f'Latent Dim {latent_dim_number}, Timestep {timestep}')
            individual_ax.hlines(0., -.1, 2.6, colors='k', linestyle='--', alpha=0.5)
            individual_ax.set_xlim(-.1, 2.6)
            individual_ax.legend()
            
            plt.tight_layout()
            
            # Save individual plot
            individual_fig.savefig(
                os.path.join(dim_save_dir, f'epoch_{epoch_num}_dim_{latent_dim_number}_timestep_{timestep}.png'), 
                dpi=300, bbox_inches='tight'
            )
            plt.close(individual_fig)
        
        print(f"Plots saved to: {dim_save_dir}")