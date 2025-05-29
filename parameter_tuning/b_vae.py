import parameters
import torch
from torch import nn
import visualisation
import numpy as np
import loader
import os

import sys
sys.path.append('../../libs/')
import shjnn

class B_VAE:
    def __init__(self, m=parameters.model_params):
        self.visualiser = self.Visualiser(self)

    def eval(self, model_params=parameters.model_params, dataset=parameters.dataset):
        #in visualisation.py, traj_tensor is of size 70 and if you feed the infer method with a time tensor that's unextrapolated (70),
        #it will return a tensor of size 70 which is the prediction and now you can MSE it with the original time_tensory.
        # Extract data from dataset - use original data for evaluation
        trajectories = dataset['trajs']  # Original trajectories without missing data
        time_points = dataset['times']   # Original time points without missing data
        metadata = dataset['y']  # Contains parameters like intensity, bias, delay

        # Extract model components
        model_func = model_params['func']
        encoder = model_params['rec']
        decoder = model_params['dec']
        optimizer = model_params['optim']
        device = model_params['device']
        epoch_num = model_params['epochs']
        
        num_dims = trajectories[0].shape[-1]
        
        infer_step = shjnn.make_infer_step(
            model_func, encoder, decoder, optimizer, device, 
            input_mode='traj', sample=False
        )
        total_samples = len(trajectories)
        print(f"logs: b_vae: eval: evaluating over {total_samples} samples.")
        loss_list = []
        
        try:
            # Process in batches for better GPU utilization
            batch_size = model_params['n_batch']
            n_batches = (total_samples + batch_size - 1) // batch_size  # Ceiling division
            
            for batch_idx in range(n_batches):
                # Clear GPU cache before each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                # Determine batch range
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_samples)
                batch_indices = list(range(start_idx, end_idx))
                batch_size_actual = len(batch_indices)
                
                # Skip empty batches
                if batch_size_actual == 0:
                    continue
                    
                # Prepare batch tensors
                batch_trajs = []
                batch_times = []
                
                for traj_idx in batch_indices:
                    batch_trajs.append(trajectories[traj_idx])
                    batch_times.append(time_points[traj_idx])
                
                # Stack into batch tensors and move to device
                traj_tensor = torch.stack(batch_trajs).to(device)
                time_tensor = torch.stack(batch_times).to(device)
                
                # We need to process each sample in the batch separately
                # because infer_step's ODE solver expects a specific format for time_points
                batch_pred_x = []
                for i in range(batch_size_actual):
                    # Process individual samples from the batch
                    sample_traj = traj_tensor[i:i+1]
                    sample_time = time_tensor[i:i+1]
                    
                    # Run inference step
                    sample_pred_x, _ = infer_step(sample_traj, sample_time)
                    batch_pred_x.append(sample_pred_x)
                
                # Combine results back into a batch
                pred_x = torch.cat(batch_pred_x, dim=0)
                
                # Compute individual losses for each sample
                for i in range(batch_size_actual):
                    # Extract individual predictions and targets
                    pred_x_i = pred_x[i:i+1]
                    traj_tensor_i = traj_tensor[i:i+1]
                    
                    # Compute individual sample loss
                    individual_loss = torch.nn.MSELoss()(pred_x_i.to(device), traj_tensor_i.to(device))
                    loss_list.append(individual_loss.item())
                
                # Calculate and report batch average loss
                batch_avg_loss = sum(loss_list[-batch_size_actual:]) / batch_size_actual
                print(f"logs: b_vae: eval: batch {batch_idx+1}/{n_batches}, avg loss: {batch_avg_loss:.6f}")
                
        except Exception as e:
            print(f"Error during batch evaluation: {e}")
            
            # Fallback to single sample processing if batch processing fails
            print("logs: b_vae: eval: falling back to single sample processing")
            loss_list = []
            
            for traj_idx in range(total_samples):
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Process individual sample
                traj_tensor = trajectories[traj_idx].view(1, *trajectories[traj_idx].size()).to(device)
                time_tensor = time_points[traj_idx].view(1, *time_points[traj_idx].size()).to(device)
                
                pred_x, pred_z = infer_step(traj_tensor, time_tensor)
                
                # Compute loss
                loss = torch.nn.MSELoss()(pred_x.to(device), traj_tensor.to(device))
                loss_list.append(loss.item())
                
                # Debug output every 10 samples
                if (traj_idx + 1) % 10 == 0:
                    print(f"logs: b_vae: eval: processed {traj_idx+1}/{total_samples} samples")
        print(f"logs: b_vae: eval: mean loss over {total_samples} samples: {np.mean(loss_list)} at epoch {epoch_num}.") 
        return np.mean(loss_list)
            
    def save_model(self, model_params):
        #save model params as json file
        folder = model_params['folder']
        epoch = model_params['epochs']
        if not os.path.exists(folder + '/model'):
            os.makedirs(folder + '/model')

        # save model
        #load model params
        func = model_params['func']
        rec = model_params['rec']
        dec = model_params['dec']
        optim = model_params['optim']
        loss = model_params['loss']
        epochs = model_params['epochs']
        folder = model_params['folder']
        path = folder + f"/model/save_model_ckpt_{epoch}.pth"
        shjnn.save_state(path, func, rec, dec, optim, loss, epochs) 
        
        
    def load_model(self, model_params, path):
        shjnn.load_state(path, model_params['func'], model_params['rec'], dec = model_params['dec'], optim = model_params['optim'], loss = model_params['loss'], epochs = model_params['epochs'], dev = model_params['device'])


    class Visualiser:
        def __init__(self, b_vae_instance):
            pass


        def plot_training_loss(self, model_params=parameters.model_params, save=False, split=False, plot_total=True, plot_MSE=True, plot_KL=True):
            # NOTE: uncomment or comment these to toggle between.
            visualisation.plot_training_loss(model_params, save=save, split=split, plot_total=plot_total, plot_MSE=False, plot_KL=False)
            # visualisation.plot_training_loss(model_params, save=save, split=True, plot_total=False, plot_MSE=plot_MSE, plot_KL=plot_KL)
            

        def display_random_fit(self, model_params=parameters.model_params, dataset=parameters.dataset, show=False, save=True, random_samples=True):
            visualisation.display_random_fit(model_params, dataset, show=show, save=save, random_samples=random_samples)
        
        def compile_learning_gif(model_params=parameters.model_params, display=True):
            visualisation.compile_learning_gif(model_params, display=display)

        def sweep_latent_adaptives(self, model_params=parameters.model_params, dataset=parameters.dataset):
            visualisation.sweep_latent_space(model_params, dataset)

        def sweep_latent_adaptive(self, model_params=parameters.model_params, dataset=parameters.dataset, latent_dim_number=0):
            visualisation.sweep_latent_space(model_params, dataset, latent_dim_number)
