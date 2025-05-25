"""
Orchestration Module for Stochastic Hidden Jump Neural Network (SHJNN)

This module provides high-level functions for orchestrating the SHJNN model:
1. Training loop
2. Model state saving and loading
3. Data preparation and handling
"""

# Import custom components
from .data import CustomDataset, prep_data
from .model import init_model
from .train import make_train_step
from .infer import make_infer_step

# Standard libraries
import pickle
import numpy as np
from sklearn import preprocessing

# PyTorch
import torch
from torch.utils.data import DataLoader


def train(dynamics_func, recognition_network, decoder, optimizer, 
          trajectories, time_points, num_epochs, batch_size, 
          device, beta=None, save_checkpoint=None, stochastic_noise=0.0):
    """
    Main training loop for the SHJNN model.
    
    Args:
        dynamics_func (nn.Module): ODE function for latent dynamics 
        recognition_network (nn.Module): Recognition network for encoding trajectories
        decoder (nn.Module): Decoder network for decoding latent states
        optimizer (torch.optim.Optimizer): Optimizer for parameter updates
        trajectories (list): List of trajectory tensors
        time_points (list): List of time point tensors
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        device (torch.device): Device to run computations on
        beta (float, optional): KL divergence weight in the loss function
        save_checkpoint (str, optional): Name prefix for saving model checkpoints
        
    Returns:
        tuple: (epochs_trained, loss_history, mse_loss_history, kl_loss_history)
    """
    # Initialize dataset and dataloader
    dataset = CustomDataset(trajectories, time_points)
    
    # In this implementation, we use the full dataset for training
    # (No validation split is performed here)
    train_dataset = dataset
    
    # Initialize training data loader for random mini-batches
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True
    )
    
    # Create the train_step function
    if beta is not None:
        train_step = make_train_step(
            dynamics_func, recognition_network, decoder, 
            optimizer, device, beta=beta, stochastic_noise=stochastic_noise
        )
    else:
        train_step = make_train_step(
            dynamics_func, recognition_network, decoder, 
            optimizer, device, stochastic_noise=stochastic_noise
        )

    # Store loss values during training
    loss_history = []
    mse_loss_history = []
    kl_loss_history = []
    
    # Iterate through epochs
    for epoch in range(1, num_epochs + 1):
        try:
            epoch_loss = 0
            num_batches = 0
            
            # Process mini-batches from dataloader
            for batch_trajectory, batch_time in train_loader:
                # Send mini-batch to device
                batch_trajectory = batch_trajectory.to(device)
                batch_time = batch_time.to(device)
                
                # Perform training step and compute loss
                batch_loss, batch_mse, batch_kl = train_step(batch_trajectory, batch_time)
                
                # Convert loss components to float
                batch_mse = float(batch_mse)
                batch_kl = float(batch_kl)
                
                # Store loss values
                loss_history.append(batch_loss)
                mse_loss_history.append(-batch_mse)  # Negative MSE (log likelihood)
                kl_loss_history.append(batch_kl)
                
                # Display batch progress
                print(f'Epoch {epoch}: Total Loss {batch_loss:.3f}, '
                      f'KL Loss {batch_kl:.2f}, Feature Loss {-batch_mse:.2f}')
                
                # Update epoch statistics
                num_batches += 1
                epoch_loss += batch_loss
            
            # Compute average loss for the epoch
            epoch_loss /= num_batches
            
            # Display epoch summary
            print(f'Epoch {epoch}: Average: {epoch_loss:.2f}, '
                  f'Median: {np.median(loss_history):.2f}')
            
            # Save model checkpoint if requested
            if save_checkpoint is not None:
                path = f'../models/ckpt_{save_checkpoint}_{epoch}.pth'
                save_state(path, dynamics_func, recognition_network, decoder, 
                          optimizer, loss_history, epoch)
                
        # Handle early termination
        except KeyboardInterrupt:
            return epoch, loss_history, mse_loss_history, kl_loss_history
    
    # Return training results
    return epoch, loss_history, mse_loss_history, kl_loss_history


def save_state(path, dynamics_func, recognition_network, decoder, optimizer, loss, epochs):
    """
    Save model state to disk.
    
    Args:
        path (str): Path to save the model checkpoint
        dynamics_func (nn.Module): ODE function model
        recognition_network (nn.Module): Recognition network model
        decoder (nn.Module): Decoder network model
        optimizer (torch.optim.Optimizer): Optimizer
        loss (list): Loss history
        epochs (int): Number of epochs trained
    """
    torch.save({
        'func_state_dict': dynamics_func.state_dict(),
        'rec_state_dict': recognition_network.state_dict(),
        'dec_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'epochs': epochs,
    }, path)


def load_state(path, dynamics_func, recognition_network, decoder, optimizer, loss, epochs, dev='gpu'):
    """
    Load model state from disk.
    
    Args:
        path (str): Path to the model checkpoint
        dynamics_func (nn.Module): ODE function model to load state into
        recognition_network (nn.Module): Recognition network model to load state into
        decoder (nn.Module): Decoder network model to load state into
        optimizer (torch.optim.Optimizer): Optimizer to load state into
        loss (list): Loss history reference (will be replaced)
        epochs (int): Epochs trained reference (will be replaced)
        dev (str): Device to load onto ('cpu' or 'gpu')
        
    Returns:
        int: Number of epochs trained
    """
    # Load checkpoint, handling device appropriately
    if dev == 'cpu':
        checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
    else:
        checkpoint = torch.load(path, weights_only=False)
    
    # Load model state dictionaries
    dynamics_func.load_state_dict(checkpoint['func_state_dict'])
    recognition_network.load_state_dict(checkpoint['rec_state_dict'])
    decoder.load_state_dict(checkpoint['dec_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Update loss history and epochs trained
    loss = checkpoint['loss']
    epochs = checkpoint['epochs']
    
    return epochs