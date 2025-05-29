import sys
sys.path.append('..')
import parameters
from torch import nn
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

import loader
import sys
sys.path.append('../../libs/')
import shjnn
from .rnn import RNN


class CNN(RNN):
    def __init__(self, m=parameters.model_params):
        super().__init__(m)
        self.windows=[5]
        self.filters=32
        # For large kernels, we'll dynamically select which ones to use based on input size
        self.temporal = nn.ModuleList([
            nn.Conv1d(in_channels=2,  # Take both charge and time features
                      device=m['device'],
                      out_channels=self.filters,
                      kernel_size=w,
                      padding=w-1)  # Use full padding (kernel_size-1) to handle short sequences
            for w in self.windows
        ])
        
        # Use LeakyReLU instead of ReLU to avoid dead neurons
        self.act = nn.LeakyReLU(0.1)
        
        # Add batch normalization to help with training stability
        self.batchnorm1 = nn.BatchNorm1d(self.filters * len(self.windows)).to(m['device'])
        self.batchnorm2 = nn.BatchNorm1d(256).to(m['device'])
        self.batchnorm3 = nn.BatchNorm1d(32).to(m['device'])
        
        total_features = self.filters * len(self.windows)
        self.h2h0 = nn.Linear(total_features, 256).to(m['device'])
        self.h2h1 = nn.Linear(256, 32).to(m['device'])
        # Output 2 features to match RNN's expected output format (charge and time)
        self.h2o = nn.Linear(32, 2).to(m['device'])
        
        # Initialize weights properly
        self._init_weights()
        self.data = torch.empty((self.model_params['n_batch'], 2, 0), device=self.model_params['device'], dtype=torch.float32)
        
        # Use Adam with weight decay and lower learning rate for stability
        self.optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=m['lr'] * 0.1,  # Lower learning rate
            weight_decay=1e-5   # L2 regularization
        )
        self.visualiser = self.Visualiser(self)
        
    def _init_weights(self):
        """Initialize network weights using Kaiming initialization"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
    def init_hidden(self, batch_size):
        d = 1
        self.data = torch.empty((self.model_params['n_batch'], 2, 0), device=self.model_params['device'], dtype=torch.float32)
        return torch.zeros(d, batch_size, self.model_params['rnn_nhidden']).to(self.model_params['device'])
        
    def forward(self, data, hidden):
        # First reshape data to [batch, channels, seq_len] for 1D convolution
        # Input shape is [batch, seq_len, features]
        batch_size = data.shape[0]
        seq_len = data.shape[1]
        
        if batch_size != self.data.shape[0]:
            self.data = torch.empty((batch_size, 2, 0), device=self.model_params['device'], dtype=torch.float32)
        
        # Correct reshape for Conv1d: [batch, channels, seq_len]
        # Conv1d expects channels first format
        x = data.transpose(1, 2)  # From [batch, seq_len, features] to [batch, features, seq_len]
        self.data = torch.cat((self.data, x), dim=2)
        x = self.data
        
        # Process through each convolutional filter with different window sizes
        conv_outputs = []
        
        
        for i, conv in enumerate(self.temporal):
            kernel_size = self.windows[i]
            
            # Skip convolutions where kernel size is still too large
            if kernel_size > seq_len:
                # Create a dummy output with zeros when kernel is too large
                dummy_output = torch.zeros(batch_size, self.filters, device=data.device)
                conv_outputs.append(dummy_output)
                continue
                
                # Apply convolution
            conv_out = conv(x)
            # Apply ReLU
            conv_out = self.act(conv_out)
            # Global max pooling for each filter
            pooled = torch.max(conv_out, dim=2)[0]
            conv_outputs.append(pooled)
        
        
        
        # Concatenate all filter outputs
        x = torch.cat(conv_outputs, dim=1)
        
        # Process through linear layers
        x = self.h2h0(x)
        # x = self.act(x)
        x = self.h2h1(x)
        # x = self.act(x)
        out = self.h2o(x)
        
        out = out.view(1, batch_size, 2)
        
        return out, hidden