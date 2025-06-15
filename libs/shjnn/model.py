"""
Neural ODE Model Components for Stochastic Hidden Jump Neural Network (SHJNN)

This module contains the model architecture components for the SHJNN model:
- Recognition RNN: Encodes trajectories into latent space
- Latent ODE: Models dynamics in latent space
- Decoder: Decodes from latent space back to observation space
"""

# PyTorch components for model
import torch
import torch.nn as nn
import torch.nn.functional as F


"""Model Initialization"""

def init_model(latent_dim, nhidden, rnn_nhidden, obs_dim, nbatch, lr, device=None):
    """
    Legacy name compatibility wrapper around the actual init_model function
    """
    """
    Initialize the complete SHJNN model.
    
    This function creates and initializes all the components of the SHJNN model:
    dynamics function, recognition network, and decoder.
    
    Args:
        latent_dim (int): Dimension of the latent space
        nhidden (int): Dimension of hidden layers in dynamics and decoder
        rnn_nhidden (int): Dimension of hidden layers in recognition RNN
        obs_dim (int): Dimension of the observation space
        nbatch (int): Batch size for training
        lr (float): Learning rate for optimizer
        device (torch.device, optional): Device to place model on. Defaults to GPU if available.
    
    Returns:
        tuple: (dynamics_func, recognition_network, decoder, optimizer, device)
    """
    # Initialize model components
    dynamics_func = LatentODEfunc(latent_dim, nhidden)
    recognition_network = RecognitionRNN(latent_dim, obs_dim, rnn_nhidden, nbatch)
    decoder = Decoder(latent_dim, obs_dim, nhidden)

    # Determine computing device (GPU or CPU)
    print(f"cuda is available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print(f'Initializing model on device: {device}')

    # Move all components to the specified device
    dynamics_func.to(device)
    recognition_network.to(device)
    decoder.to(device)

    # Aggregate all model parameters for the optimizer
    all_parameters = (
        list(dynamics_func.parameters()) + 
        list(decoder.parameters()) + 
        list(recognition_network.parameters())
    )

    # Initialize optimizer
    optimizer = torch.optim.Adam(all_parameters, lr=lr)

    # Return all model components
    return dynamics_func, recognition_network, decoder, optimizer, device


"""Model Components"""

class RecognitionRNN(nn.Module):
    """
    Recognition RNN for trajectory encoding.
    
    Processes trajectories in reverse order to encode them into latent space
    distributions (mean and variance). The model acts as a variational encoder.
    """

    def __init__(self, latent_dim, obs_dim, hidden_size, batch_size=1):
        """
        Initialize the recognition RNN.
        
        Args:
            latent_dim (int): Dimension of the latent space
            obs_dim (int): Dimension of the observation space
            hidden_size (int): Size of hidden state
            batch_size (int): Batch size for training
        """
        super(RecognitionRNN, self).__init__()

        # Store sizes for later use
        self.nhidden = hidden_size
        self.nbatch = batch_size

        # Input and hidden state to new hidden state
        self.i2h = nn.Linear(obs_dim + hidden_size, hidden_size)

        # Hidden state to output (mean and log-variance of latent vars)
        self.h2o = nn.Linear(hidden_size, latent_dim * 2)

    def forward(self, observation, hidden_state):
        """
        Forward pass through the RNN.
        
        Args:
            observation (Tensor): Current observation [batch_size, obs_dim]
            hidden_state (Tensor): Current hidden state [batch_size, hidden_size]
            
        Returns:
            tuple: (output, new_hidden_state)
                - output: latent distribution parameters [batch_size, latent_dim*2]
                - new_hidden_state: updated hidden state [batch_size, hidden_size]
        """
        # Concatenate input and hidden state
        combined = torch.cat((observation, hidden_state), dim=1)

        # Compute new hidden state with tanh activation
        new_hidden_state = torch.tanh(self.i2h(combined))

        # Compute output from new hidden state
        output = self.h2o(new_hidden_state)

        # Return output and new hidden state
        return output, new_hidden_state

    def initHidden(self):
        """
        Initialize hidden state with zeros.
        
        Returns:
            Tensor: Zero-initialized hidden state [batch_size, hidden_size]
        """
        return torch.zeros(self.nbatch, self.nhidden)


class LatentODEfunc(nn.Module):
    """
    Neural ODE function for modeling dynamics in latent space.
    
    This network parameterizes the dynamics function that governs
    how the latent states evolve over time.
    """

    def __init__(self, latent_dim=4, hidden_size=20):
        """
        Initialize the ODE function network.
        
        Args:
            latent_dim (int): Dimension of the latent space
            hidden_size (int): Dimension of hidden layers
        """
        super(LatentODEfunc, self).__init__()

        # Activation function
        self.elu = nn.ELU(inplace=True)

        # Network architecture
        self.input_layer = nn.Linear(latent_dim, hidden_size)  # Input layer
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)  # Hidden layer
        self.output_layer = nn.Linear(hidden_size, latent_dim)  # Output layer

        # Counter for function evaluations (useful for profiling)
        self.nfe = 0

    def forward(self, t, latent_state):
        """
        Forward pass computing the derivative of the latent state.
        
        Args:
            t (Tensor): Current time point (not used in this implementation)
            latent_state (Tensor): Current latent state [batch_size, latent_dim]
            
        Returns:
            Tensor: Time derivative of latent state [batch_size, latent_dim]
        """
        # Increment function evaluation counter
        self.nfe += 1

        # Forward pass through network with activations
        out = self.input_layer(latent_state)
        out = self.elu(out)

        out = self.hidden_layer(out)
        out = self.elu(out)

        out = self.output_layer(out)

        return out


class Decoder(nn.Module):
    """
    Decoder network for transforming latent states to observation space.
    
    Maps points from the latent space back to the observation space,
    acting as the decoder in the variational autoencoder framework.
    """

    def __init__(self, latent_dim, obs_dim=2, hidden_size=20):
        """
        Initialize the decoder network.
        
        Args:
            latent_dim (int): Dimension of the latent space
            obs_dim (int): Dimension of the observation space
            hidden_size (int): Dimension of hidden layer
        """
        super(Decoder, self).__init__()

        # Activation function
        self.relu = nn.ReLU(inplace=True)

        # Network architecture
        self.fc1 = nn.Linear(latent_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, obs_dim)

    def forward(self, latent_state):
        """
        Forward pass decoding latent state to observation space.
        
        Args:
            latent_state (Tensor): Latent state [batch_size, latent_dim]
            
        Returns:
            Tensor: Decoded observation [batch_size, obs_dim]
        """
        # Apply network layers with activation
        out = self.fc1(latent_state)
        out = self.relu(out)

        out = self.fc2(out)

        return out


"""
Neural Controlled Differential Equation (CDE) Model
Note: This is a work in progress - not currently used in the main model
"""

class CDEFunc(torch.nn.Module):
    """
    Neural CDE function for modeling continuous dynamics.
    
    This is an alternative dynamics model that uses controlled differential equations.
    Currently not used in the main model.
    """

    def __init__(self, input_channels, hidden_channels):
        """
        Initialize the CDE function network.
        
        Args:
            input_channels (int): Number of input channels in the data
            hidden_channels (int): Number of channels for the hidden state
        """
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 128)
        self.linear2 = torch.nn.Linear(128, input_channels * hidden_channels)

    def forward(self, z):
        """
        Forward pass computing the CDE vector field.
        
        Args:
            z (Tensor): Current state
            
        Returns:
            Tensor: Vector field for the CDE
        """
        z = self.linear1(z)
        z = torch.tanh(z)
        z = self.linear2(z)

        # Reshape output to represent a linear map
        z = z.view(*z.shape[:-1], self.hidden_channels, self.input_channels)

        return z


class NeuralCDE(torch.nn.Module):
    """
    Neural CDE model for time series modeling.
    
    This model uses controlled differential equations to model dynamics.
    Currently not used in the main model.
    """

    def __init__(self, input_channels, hidden_channels):
        """
        Initialize the Neural CDE model.
        
        Args:
            input_channels (int): Number of input channels in the data
            hidden_channels (int): Number of channels for the hidden state
        """
        super(NeuralCDE, self).__init__()

        self.hidden_channels = hidden_channels
        self.func = CDEFunc(input_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, input_channels)

    def forward(self, times, coeffs):
        """
        Forward pass integrating the CDE.
        
        Args:
            times (Tensor): Time points to evaluate at
            coeffs (tuple): Spline coefficients
            
        Returns:
            Tensor: Predicted outputs
        """
        # Extract batch dimensions from coefficients
        coeff, _, _, _ = coeffs
        batch_dims = coeff.shape[:-2]
        z0 = torch.zeros(*batch_dims, self.hidden_channels, dtype=times.dtype, device=times.device)

        # Note: This requires the controldiffeq package which may not be available
        # Integrate the CDE from start to end time
        z_T = controldiffeq.cdeint(
            dX_dt=controldiffeq.NaturalCubicSpline(times, coeffs).derivative,
            z0=z0,
            func=self.func,
            t=times[[0, -1]],
            atol=1e-2,
            rtol=1e-2
        )

        # Extract final state and transform to output
        z_T = z_T[1]
        pred_y = self.linear(z_T)

        return pred_y