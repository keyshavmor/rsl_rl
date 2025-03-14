import torch.nn as nn
import numpy as np
import torch


class StateHistoryEncoder(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size):
        super(StateHistoryEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps
        self.input_shape = input_size*tsteps
        self.output_shape = output_size
        # self.encoder = nn.Sequential(
        #         nn.Linear(input_size, 128), self.activation_fn(),
        #         nn.Linear(128, 32), self.activation_fn()
        #         )

        if tsteps == 50:
            self.encoder = nn.Sequential(
            nn.Linear(input_size, 32), self.activation_fn
            )
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 8, stride = 4), nn.LeakyReLU(),
                    nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 5, stride = 1), nn.LeakyReLU(),
                    nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 5, stride = 1), nn.LeakyReLU(), nn.Flatten())
            self.linear_output = nn.Sequential(
            nn.Linear(32 * 3, output_size), self.activation_fn
            )

        elif tsteps == 10:
            self.encoder = nn.Sequential(
            nn.Linear(input_size, 32), self.activation_fn
            )
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 4, stride = 2), nn.LeakyReLU(), 
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 2, stride = 1), nn.LeakyReLU(), 
                nn.Flatten())
            self.linear_output = nn.Sequential(
            nn.Linear(32 * 3, output_size), self.activation_fn
            )
        
        elif tsteps == 20:
            self.encoder = nn.Sequential(
            nn.Linear(input_size, 32), self.activation_fn
            )
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 6, stride = 2), nn.LeakyReLU(), 
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 4, stride = 2), nn.LeakyReLU(), 
                nn.Flatten())
            self.linear_output = nn.Sequential(
                nn.Linear(32 * 3, output_size), self.activation_fn
            )
        else:
            raise NotImplementedError()

    def forward(self, obs):
        bs = obs.shape[0]
        T = self.tsteps
        projection = self.encoder(obs.reshape([bs * T, -1]))
        output = self.conv_layers(projection.reshape([bs, -1, T]))
        output = self.linear_output(output)
        return output

def min_max_scaler(tensor, feature_range=(-1, 1)):
    min_val = torch.min(tensor, dim=0, keepdim=True)[0]
    max_val = torch.max(tensor, dim=0, keepdim=True)[0]
    
    scale = (feature_range[1] - feature_range[0]) / (max_val - min_val + 1e-8)  # Adding small value to avoid division by zero
    min_max_scaled = feature_range[0] + (tensor - min_val) * scale
    
    return min_max_scaled