import itertools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.distributions.normal import Normal

class football_NN(nn.Module):

	def __init__(self,
                input_channels:int,
                output_channels:int,
				p_mc_dropout = 0.5) :

		super().__init__()

		self.loss = nn.MSELoss()
		self.val_loss = nn.MSELoss()

		self.p_mc_dropout = p_mc_dropout

		self.linear1 = nn.Linear(input_channels,128)
		self.linear2 = nn.Linear(128,256)
		self.linear3 = nn.Linear(256,128)
		self.linear4 = nn.Linear(128,64)
		self.linear5 = nn.Linear(64,output_channels)



	def forward(self, x):

		x = nn.functional.relu(self.linear1(x))
		x = nn.functional.relu(self.linear2(x))
		x = nn.functional.relu(self.linear3(x))
		x = nn.functional.relu(self.linear4(x))
		x = self.linear5(x)

		return x

class fire_NN(nn.Module):
	'''
	two-headed belief model for the fire env.

	input is the flattened array produced by fire_obs_packaging:
	[ fire_maps (num_frames * num_classes * window_size * window_size),
	  pos_change ((min_obs-1) * 2),
	  last_team_obs ((n_agents-1) * 2) ]

	each fire map is encoded over num_classes fire states (channels first,
	RED/BLACK only - GREEN/WHITE are all-zero, see _one_hot_fire_grid) rather
	than a single scalar cell value. the fire_maps prefix is reshaped back
	into a (num_frames * num_classes, window_size, window_size) image stack
	and fed to a cnn head; everything after it (pos over time + last team
	estimate) is fed to a linear head. the two heads are fused before the
	output layer.
	'''

	def __init__(self,
                input_channels:int,
                output_channels:int,
				window_size:int = 61,
				num_frames:int = 10,
				num_classes:int = 2,
				p_mc_dropout = 0.5) :

		super().__init__()

		self.loss = FirePermutationInvariantLoss()
		self.val_loss = FirePermutationInvariantLoss()

		self.p_mc_dropout = p_mc_dropout

		self.window_size = window_size
		self.num_frames = num_frames
		self.num_classes = num_classes
		self.fire_input_size = num_frames * num_classes * window_size * window_size
		self.linear_input_size = input_channels - self.fire_input_size

		# ---------- cnn head: reconstructed fire maps over time ----------
		self.cnn_head = nn.Sequential(
			nn.Conv2d(num_frames * num_classes, 32, kernel_size=5, stride=2, padding=2),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
			nn.ReLU(inplace=True),
			# AdaptiveAvgPool2d has no deterministic CUDA backward; conv stack
			# always yields an 8x8 map here, so a plain AvgPool2d(2) is
			# equivalent and keeps training deterministic
			nn.AvgPool2d(kernel_size=2, stride=2),
		)
		self.cnn_proj = nn.Linear(128 * 4 * 4, 64)

		# ---------- linear head: pos over time + last team estimate ----------
		self.linear_head = nn.Sequential(
			nn.Linear(self.linear_input_size, 128),
			nn.ReLU(inplace=True),
			nn.Linear(128, 256),
			nn.ReLU(inplace=True),
		)

		# ---------- fusion ----------
		self.fusion = nn.Sequential(
			nn.Linear(64 + 256, 256),
			nn.ReLU(inplace=True),
			nn.Linear(256, 128),
			nn.ReLU(inplace=True),
			nn.Linear(128, 64),
			nn.ReLU(inplace=True),
		)
		self.out = nn.Linear(64, output_channels)

	def forward(self, x):
		unbatched = x.dim() == 1
		if unbatched:
			x = x.unsqueeze(0)

		fire_x = x[:, :self.fire_input_size].contiguous().view(
			-1, self.num_frames * self.num_classes, self.window_size, self.window_size
		)
		linear_x = x[:, self.fire_input_size:]

		fire_feat = self.cnn_head(fire_x)
		fire_feat = torch.flatten(fire_feat, 1)
		fire_feat = F.relu(self.cnn_proj(fire_feat))

		linear_feat = self.linear_head(linear_x)

		combined = torch.cat((fire_feat, linear_feat), dim=1)
		combined = self.fusion(combined)

		out = self.out(combined)

		return out.squeeze(0) if unbatched else out

class fire_CNN(nn.Module):
	'''
	cnn-only belief model for the fire env.

	takes the same flattened observation as fire_NN (see fire_NN docstring
	for the layout), but only runs the cnn head on the fire_maps prefix;
	the trailing linear input (pos over time + last team estimate) is
	sliced off and ignored.
	'''

	def __init__(self,
                input_channels:int,
                output_channels:int,
				window_size:int = 61,
				num_frames:int = 10,
				num_classes:int = 2,
				p_mc_dropout = 0.5) :

		super().__init__()

		self.loss = FirePermutationInvariantLoss()
		self.val_loss = FirePermutationInvariantLoss()

		self.p_mc_dropout = p_mc_dropout

		self.window_size = window_size
		self.num_frames = num_frames
		self.num_classes = num_classes
		self.fire_input_size = num_frames * num_classes * window_size * window_size

		# ---------- cnn head: reconstructed fire maps over time ----------
		self.cnn_head = nn.Sequential(
			nn.Conv2d(num_frames * num_classes, 32, kernel_size=5, stride=2, padding=2),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
			nn.ReLU(inplace=True),
			# AdaptiveAvgPool2d has no deterministic CUDA backward; conv stack
			# always yields an 8x8 map here, so a plain AvgPool2d(2) is
			# equivalent and keeps training deterministic
			nn.AvgPool2d(kernel_size=2, stride=2),
		)
		self.cnn_proj = nn.Linear(128 * 4 * 4, 64)

		# ---------- head: cnn features straight to output ----------
		self.head = nn.Sequential(
			nn.Linear(64, 128),
			nn.ReLU(inplace=True),
			nn.Linear(128, 64),
			nn.ReLU(inplace=True),
		)
		self.out = nn.Linear(64, output_channels)

	def forward(self, x):
		unbatched = x.dim() == 1
		if unbatched:
			x = x.unsqueeze(0)

		fire_x = x[:, :self.fire_input_size].contiguous().view(
			-1, self.num_frames * self.num_classes, self.window_size, self.window_size
		)

		fire_feat = self.cnn_head(fire_x)
		fire_feat = torch.flatten(fire_feat, 1)
		fire_feat = F.relu(self.cnn_proj(fire_feat))

		out = self.head(fire_feat)
		out = self.out(out)

		return out.squeeze(0) if unbatched else out

class football_VAE_NN(nn.Module):
	def __init__(self, input_channels, output_channels, latent_dim=16):
		super().__init__()

		self.loss = FootballVAEMSE()
		self.val_loss = nn.MSELoss()

		self.linear1 = nn.Linear(input_channels, 128)
		self.linear2 = nn.Linear(128, 256)
		self.linear3 = nn.Linear(256, 128)
		self.linear4 = nn.Linear(128, 64)

		self.vae = VAELayer(64, latent_dim)
		self.decoder = nn.Linear(latent_dim, output_channels)

	def forward(self, x, stochastic=False):
		x = nn.functional.relu(self.linear1(x))
		x = nn.functional.relu(self.linear2(x))
		x = nn.functional.relu(self.linear3(x))
		x = nn.functional.relu(self.linear4(x))

		z, mu, logvar = self.vae(x, stochastic=stochastic)
		pred = self.decoder(z)

		return pred, mu, logvar

class predator_prey_VAE_NN(nn.Module):
	def __init__(self, input_channels, output_channels, latent_dim=16):
		super().__init__()

		self.loss = PredPreyPermutationInvariantVAEMSE()
		self.val_loss = PredPreyPermutationInvariantMSE()

		self.linear1 = nn.Linear(input_channels, 512)
		self.linear2 = nn.Linear(512, 1024)
		self.linear3 = nn.Linear(1024, 4096)
		self.linear6 = nn.Linear(4096, 1024)
		self.linear7 = nn.Linear(1024, 64)

		self.vae = VAELayer(64, latent_dim)
		self.decoder = nn.Linear(latent_dim, output_channels)

	def forward(self, x, stochastic=False):
		x = nn.functional.relu(self.linear1(x))
		x = nn.functional.relu(self.linear2(x))
		x = nn.functional.relu(self.linear3(x))
		x = nn.functional.relu(self.linear6(x))
		x = nn.functional.relu(self.linear7(x))

		z, mu, logvar = self.vae(x, stochastic=stochastic)
		pred = self.decoder(z)

		return pred, mu, logvar

class drones_VAE_NN(nn.Module):
    def __init__(self, input_channels, output_channels, latent_dim=16):
        super().__init__()

        self.linear1 = nn.Linear(input_channels, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, 4096)
        self.linear6 = nn.Linear(4096, 1024)
        self.linear7 = nn.Linear(1024, 64)

        self.vae = VAELayer(64, latent_dim)
        self.decoder = nn.Linear(latent_dim, output_channels)

    def forward(self, x, stochastic=False):
        x = nn.functional.relu(self.linear1(x))
        x = nn.functional.relu(self.linear2(x))
        x = nn.functional.relu(self.linear3(x))
        x = nn.functional.relu(self.linear6(x))
        x = nn.functional.relu(self.linear7(x))

        z, mu, logvar = self.vae(x, stochastic=stochastic)
        pred = self.decoder(z)

        return pred, mu, logvar

class predator_prey_NN(nn.Module):

	def __init__(self, 
                input_channels:int,
                output_channels:int,
				p_mc_dropout = 0.5) :
		
		super().__init__()

		self.loss = PredPreyPermutationInvariantMSE()
		self.val_loss = PredPreyPermutationInvariantMSE()

		self.p_mc_dropout = p_mc_dropout

		self.linear1 = nn.Linear(input_channels,128)
		self.linear2 = nn.Linear(128,256)
		self.linear3 = nn.Linear(256,128)
		self.linear4 = nn.Linear(128,64)
		self.linear5 = nn.Linear(64,output_channels)



	def forward(self, x):

		x = nn.functional.relu(self.linear1(x))
		x = nn.functional.relu(self.linear2(x))
		x = nn.functional.relu(self.linear3(x))
		x = nn.functional.relu(self.linear4(x))
		x = self.linear5(x)

		return x

class drones_NN(nn.Module):

	def __init__(self,
                input_channels:int,
                output_channels:int,
				p_mc_dropout = 0.5) :

		super().__init__()

		self.loss = DronesPermutationInvariantMSE()
		self.val_loss = DronesPermutationInvariantMSE()

		self.p_mc_dropout = p_mc_dropout

		self.linear1 = nn.Linear(input_channels,128)
		self.linear2 = nn.Linear(128,256)
		self.linear3 = nn.Linear(256,128)
		self.linear4 = nn.Linear(128,64)
		self.linear5 = nn.Linear(64,output_channels)



	def forward(self, x):

		x = nn.functional.relu(self.linear1(x))
		x = nn.functional.relu(self.linear2(x))
		x = nn.functional.relu(self.linear3(x))
		x = nn.functional.relu(self.linear4(x))
		x = self.linear5(x)

		return x

class market_game_NN(nn.Module):

	def __init__(self,
                input_channels:int,
                output_channels:int,
				p_mc_dropout = 0.5) :

		super().__init__()

		self.loss = MarketGamePermutationInvariantMSE()
		self.val_loss = MarketGamePermutationInvariantMSE()

		self.p_mc_dropout = p_mc_dropout

		self.linear1 = nn.Linear(input_channels,128)
		self.linear2 = nn.Linear(128,256)
		self.linear3 = nn.Linear(256,128)
		self.linear4 = nn.Linear(128,64)
		self.linear5 = nn.Linear(64,output_channels)

	def forward(self, x):

		x = nn.functional.relu(self.linear1(x))
		x = nn.functional.relu(self.linear2(x))
		x = nn.functional.relu(self.linear3(x))
		x = nn.functional.relu(self.linear4(x))
		x = self.linear5(x)

		return x

class PredPreyPermutationInvariantMSE(nn.Module):

	def __init__(
			self,
		):
		super().__init__()

	def forward(self, pred, target):
		return self.permutation_invariant_loss(pred, target)

	def permutation_invariant_loss(self, pred, target):
		"""
		pred:   (batch, num_frames * 2 * dim)
		target: (batch, num_frames * 2 * dim)
		"""
		# each time frame picks its own direct/swapped teammate
		# assignment independently, so the previous and current
		# state estimates don't have to agree on slot identity
		num_frames = pred.shape[1] // 4

		pred = pred.view(-1, num_frames, 2, 2)
		target = target.view(-1, num_frames, 2, 2)

		# direct assignment
		loss1 = ((pred - target) ** 2).mean(dim=3).sum(dim=2)

		# swapped assignment
		loss2 = ((pred - target.flip(2)) ** 2).mean(dim=3).sum(dim=2)

		# take minimum per frame, then sum across frames
		loss = torch.min(loss1, loss2).sum(dim=1)

		return loss.mean()


class MarketGamePermutationInvariantMSE(nn.Module):

	def __init__(
			self,
			num_frames: int = 2,
		):
		super().__init__()
		self.num_frames = num_frames

	def forward(self, pred, target):
		return self.permutation_invariant_loss(pred, target)

	def permutation_invariant_loss(self, pred, target):
		"""
		pred:   (batch, num_frames * n_team)
		target: (batch, num_frames * n_team)
		"""
		# each frame picks its own teammate assignment, as in PredPreyPermutationInvariantMSE. with one
		# scalar (stockpile) per teammate the optimal assignment is the sorted order, so no search is needed
		pred = pred.view(pred.shape[0], self.num_frames, -1).sort(dim=2).values
		target = target.view(target.shape[0], self.num_frames, -1).sort(dim=2).values

		return ((pred - target) ** 2).sum(dim=2).sum(dim=1).mean()


class PredPreyPermutationInvariantVAEMSE(nn.Module):

	def __init__(
			self,
		):
		super().__init__()

	def forward(self, pred, target, mu, logvar, beta_kl=1):

		kl_loss = -0.5 * torch.sum(
			1 + logvar - mu.pow(2) - logvar.exp(),
			dim=1
		)
		kl_loss = kl_loss.mean() * beta_kl

		total_loss = kl_loss + self.permutation_invariant_loss(pred, target)

		return total_loss, kl_loss #kl1+kl2

	def permutation_invariant_loss(
		self,
		recon,
		target,
		return_parts=False,
	):
		# each time frame picks its own direct/swapped teammate
		# assignment independently, so the previous and current
		# state estimates don't have to agree on slot identity
		num_frames = recon.shape[1] // 4

		recon = recon.view(-1, num_frames, 2, 2)
		target = target.view(-1, num_frames, 2, 2)

		mse_direct = ((recon - target) ** 2).mean(dim=3).sum(dim=2)
		mse_swapped = ((recon - target.flip(2)) ** 2).mean(dim=3).sum(dim=2)

		recon_loss = torch.min(mse_direct, mse_swapped).sum(dim=1)

		return recon_loss.mean()

class DronesPermutationInvariantMSE(nn.Module):

	def __init__(
			self,
		):
		super().__init__()

	def forward(self, pred, target):
		return self.permutation_invariant_loss(pred, target)

	def permutation_invariant_loss(self, pred, target):
		"""
		pred:   (batch, num_frames * 2 * dim)
		target: (batch, num_frames * 2 * dim)
		"""
		# each time frame picks its own direct/swapped teammate
		# assignment independently, so the previous and current
		# state estimates don't have to agree on slot identity
		num_frames = pred.shape[1] // 6

		pred = pred.view(-1, num_frames, 2, 3)
		target = target.view(-1, num_frames, 2, 3)

		# direct assignment
		loss1 = ((pred - target) ** 2).mean(dim=3).sum(dim=2)

		# swapped assignment
		loss2 = ((pred - target.flip(2)) ** 2).mean(dim=3).sum(dim=2)

		# take minimum per frame, then sum across frames
		loss = torch.min(loss1, loss2).sum(dim=1)

		return loss.mean()


class DronesPermutationInvariantVAEMSE(nn.Module):

	def __init__(
			self,
		):
		super().__init__()

	def forward(self, pred, target, mu, logvar, beta_kl=1):

		kl_loss = -0.5 * torch.sum(
			1 + logvar - mu.pow(2) - logvar.exp(),
			dim=1
		)
		kl_loss = kl_loss.mean() * beta_kl

		total_loss = kl_loss + self.permutation_invariant_loss(pred, target)

		return total_loss, kl_loss #kl1+kl2

	def permutation_invariant_loss(
		self,
		recon,
		target,
		return_parts=False,
	):
		# each time frame picks its own direct/swapped teammate
		# assignment independently, so the previous and current
		# state estimates don't have to agree on slot identity
		num_frames = recon.shape[1] // 6

		recon = recon.view(-1, num_frames, 2, 3)
		target = target.view(-1, num_frames, 2, 3)

		mse_direct = ((recon - target) ** 2).mean(dim=3).sum(dim=2)
		mse_swapped = ((recon - target.flip(2)) ** 2).mean(dim=3).sum(dim=2)

		recon_loss = torch.min(mse_direct, mse_swapped).sum(dim=1)

		return recon_loss.mean()

class FirePermutationInvariantLoss(nn.Module):

	def __init__(
			self,
		):
		super().__init__()

		# the 3 teammates within a team estimate are interchangeable,
		# so try every ordering of them rather than just a flip
		self.perms = list(itertools.permutations(range(3)))

	def forward(self, pred, target):
		return self.permutation_invariant_loss(pred, target)

	def permutation_invariant_loss(self, pred, target):
		"""
		pred:   (batch, num_team_estimates * 3 * 2)
		target: (batch, num_team_estimates * 3 * 2)
		"""
		# each team estimate picks its own teammate-order assignment
		# independently, so the two estimates don't have to agree on
		# slot identity
		num_estimates = pred.shape[1] // 6

		pred = pred.view(-1, num_estimates, 3, 2)
		target = target.view(-1, num_estimates, 3, 2)

		losses = [
			((pred - target[:, :, perm, :]) ** 2).mean(dim=3).sum(dim=2)
			for perm in self.perms
		]

		# take minimum per team estimate, then sum across estimates
		loss = torch.stack(losses, dim=0).min(dim=0).values.sum(dim=1)

		return loss.mean()


class FootballVAEMSE(nn.Module):

	def __init__(
			self,
		):
		super().__init__()

		self.reconstruction_loss = nn.MSELoss()

	def forward(self, pred, target, mu, logvar, beta_kl=0.0044):

		kl_loss = -0.5 * torch.sum(
			1 + logvar - mu.pow(2) - logvar.exp(),
			dim=1
		)
		kl_loss = kl_loss.mean() * beta_kl

		total_loss = kl_loss + self.reconstruction_loss(pred, target)

		return total_loss, kl_loss

class VAELayer(nn.Module):
    """
    A custom Variational Autoencoder bottleneck layer.
    Maps an incoming hidden representation to a latent distribution, 
    samples from it using the reparameterization trick, and prepares it for decoding.
    """
    def __init__(self, hidden_dim: int, latent_dim: int):
        super(VAELayer, self).__init__()
        
        # Parallel layers to parameterize the Gaussian latent distribution
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, stochastic: bool = False) -> torch.Tensor:
        """
        Applies the reparameterization trick to allow backpropagation.
        """
        if self.training or stochastic:
            # Calculate standard deviation: std = exp(0.5 * logvar)
            std = torch.exp(0.5 * logvar)
            # Sample random noise epsilon from standard normal distribution
            eps = torch.randn_like(std)
            # Return the sampled latent vector
            return mu + eps * std
        else:
            # During evaluation/inference, bypass noise and use the mean directly
            return mu
			
    def forward(self, h: torch.Tensor, stochastic: bool = False):
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        z = self.reparameterize(mu, logvar, stochastic=stochastic)

        return z, mu, logvar