import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FrequencyEnergyIndicator(nn.Module):
    def __init__(self, num_bands=3):
        super().__init__()
        self.num_bands = num_bands
        self.scales = [2**(k) for k in range(num_bands)] 

    def get_gaussian_kernel(self, kernel_size, sigma, channels):
        x = torch.arange(kernel_size) - (kernel_size - 1) / 2.
        x_grid = x.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
        
        variance = sigma**2.
        kernel = (1./(2.*math.pi*variance)) * torch.exp(
            -torch.sum(xy_grid**2., dim=-1) / (2*variance)
        )
        kernel = kernel / torch.sum(kernel)
        
        # Reshape for depthwise conv
        return kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)

    def apply_gaussian_blur(self, x, sigma):
        if sigma <= 0: return x
        k_size = int(2 * 4 * sigma + 1) | 1 
        kernel = self.get_gaussian_kernel(k_size, sigma, x.shape[1]).to(x.device).type(x.dtype)
        padding = k_size // 2
        return F.conv2d(x, kernel, padding=padding, groups=x.shape[1])

    def forward(self, z_t):
        B, C, H, W = z_t.shape
        kappa = min(H, W) / 128.0
        sigmas = [kappa * s for s in self.scales]
        
        gaussian_pyramid = [z_t]
        for s in sigmas:
            gaussian_pyramid.append(self.apply_gaussian_blur(z_t, s))
            
        band_components = []
        band_components.append(gaussian_pyramid[-1])
        
        for i in range(len(sigmas)-1, 0, -1):
            band_components.append(gaussian_pyramid[i] - gaussian_pyramid[i+1])
            
        band_components.append(gaussian_pyramid[0] - gaussian_pyramid[1])
        
        if len(band_components) > self.num_bands:
            band_components = band_components[:self.num_bands]

        energies = [torch.sum(b**2, dim=[1, 2, 3]) for b in band_components]
        energy_vec = torch.stack(energies, dim=1) # (B, num_bands)
        
        e_t = energy_vec / (torch.sum(energy_vec, dim=1, keepdim=True) + 1e-8)
        return e_t, band_components