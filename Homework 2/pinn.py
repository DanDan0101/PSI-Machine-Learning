import torch
device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device("cpu")

from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
from matplotlib import pyplot as plt

import os
if os.name == 'posix':
    # Cluster
    os.chdir("/home/dsun/PSI-Machine-Learning/Homework 2")

class FCNN(nn.Module):
    def __init__(self, w_norm):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 8),
            nn.SiLU(),
            nn.Linear(8, 4),
            nn.SiLU(),
            nn.Linear(4, 2),
            nn.Sigmoid()
        )
    def forward(self, t):
        return self.net(t)

class PINN(nn.Module):
    def __init__(self, w_norm):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(w_norm[0], device = device, dtype = torch.float32))
        self.beta = nn.Parameter(torch.tensor(w_norm[1], device = device, dtype = torch.float32))
        self.gamma = nn.Parameter(torch.tensor(w_norm[2], device = device, dtype = torch.float32))
        self.delta = nn.Parameter(torch.tensor(w_norm[3], device = device, dtype = torch.float32))
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 8),
            nn.SiLU(),
            nn.Linear(8, 4),
            nn.SiLU(),
            nn.Linear(4, 2),
            nn.Sigmoid()
        )
    def forward(self, t):
        return self.net(t)
class data:
    def __init__(self, p, name, longname, period, t = None):
        """
        Conventions:
        * NumPy arrays are unnormalized
        * PyTorch tensors are normalized to [0, 1]
        """

        if t is None:
            t = np.arange(p.shape[0])
        self.t = t
        self.dt = self.t[1] - self.t[0]
        self.period = period # Eyeballed period, as an integer index
        self.t_scale = self.period * self.dt
        # p has shape (T, 2), with columns [hare, lynx]
        self.p = p
        self.p_scale = np.max(self.p, axis = 0)

        # Parameter scaling
        self.w_scale = 1 / (self.t_scale * np.array([1, self.p_scale[1], 1, self.p_scale[0]]))

        # Convert to tensors for training
        self.t_norm = torch.from_numpy(self.t / self.t_scale).to(device, dtype = torch.float32).unsqueeze(1)
        self.p_norm = torch.from_numpy(self.p / self.p_scale).to(device, dtype = torch.float32)

        # Metadata
        self.name = name
        self.longname = longname

        # Use first 60% for training
        self.t_train = self.t_norm[:int(np.ceil(0.6 * self.t.shape[0]))]
        self.p_train = self.p_norm[:int(np.ceil(0.6 * self.t.shape[0]))]

        # Create DataLoaders (with shuffling) for training and evaluation
        self.dataloader = DataLoader(TensorDataset(self.t_norm, self.p_norm), shuffle = True)
        self.train_dataloader = DataLoader(TensorDataset(self.t_train, self.p_train), shuffle = True)

        # Load saved models
        try:
            self.fcnn = torch.load(f"models/{self.name}_fcnn.pt", weights_only = False)
        except FileNotFoundError:
            pass
        try:
            self.pinn = torch.load(f"models/{self.name}_pinn.pt", weights_only = False)
        except FileNotFoundError:
            pass

    def unnormalize_pred(self, pred):
        return pred.detach().cpu().numpy() * self.p_scale
    def unnormalize_w(self, w):
        return w.detach().cpu().numpy() * self.w_scale

    def plot(self):
        plt.figure(dpi = 300, layout = "constrained")
        plt.plot(self.t, self.p[:, 0], label = "Hare")
        plt.plot(self.t, self.p[:, 1], label = "Lynx")
        plt.axvline(self.t[int(np.ceil(0.6 * self.t.shape[0])) - 1], color = "black", linestyle = "--", label = "Train/Eval Split")
        plt.xlabel("Time (years)")
        plt.ylabel("Population (thousands)")
        plt.title(self.longname)
        plt.legend()
        # plt.savefig(f"plots/{self.name}.pdf")
        plt.show()
    
    def plot_pred(self, pred, title, filename = None, train_line = True, figsize = None):
        plt.figure(dpi = 300, layout = "constrained", figsize = figsize)
        plt.plot(self.t, self.p[:, 0], label = "Hare (True)")
        plt.plot(self.t, self.p[:, 1], label = "Lynx (True)")

        if torch.is_tensor(pred):
            pred = self.unnormalize_pred(pred)
        plt.plot(self.t, pred[:, 0], label = "Hare (Pred)", color = "C0", linestyle = "--")
        plt.plot(self.t, pred[:, 1], label = "Lynx (Pred)", color = "C1", linestyle = "--")

        if train_line:
            plt.axvline(self.t[int(np.ceil(0.6 * self.t.shape[0])) - 1], color = "black", linestyle = "--", label = "Train/Eval Split")
        
        plt.xlabel("Time (years)")
        plt.ylabel("Population (thousands)")
        plt.title(title)
        plt.legend()
        if filename is not None:
            plt.savefig(f"plots/{filename}.pdf")
        plt.show()