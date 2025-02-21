import torch
from torch import nn
from torch.nn import functional as F
from misc import reset_seed

# This implementation is based on open source resources, provided by Leon Pielage

class DenoisingDiffusion:
    def __init__(self, eps_model: nn.Module, n_steps: int, device: str = None, **kwargs):
        super().__init__(**kwargs)
        self.eps_model = eps_model  # Model to predict epsilon (the noise)
        self._T = n_steps  # Maximum diffusion time steps
        self._s = 0.0001  # Small offset to prevent beta_t from being too small

        # Choose device to run the model on
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def _beta(self, t: torch.Tensor):
        return torch.clip(1 - (self._alpha_bar(t) / self._alpha_bar(t - 1)), 0, 0.999).view(-1, 1, 1, 1)

    def _alpha(self, t: torch.Tensor):
        return (1. - self._beta(t)).view(-1, 1, 1, 1)

    def _alpha_bar(self, t: torch.Tensor):
        t = t.to(self.device)  # Ensure t is on the correct device
        f_t = torch.cos((((t / self._T) + self._s) / (1 + self._s)) * (torch.pi / 2))
        f_0 = torch.cos((((torch.tensor(0, device=self.device) / self._T) + self._s) / (1 + self._s)) * (torch.pi / 2))
        return (f_t / f_0).view(-1, 1, 1, 1)

    def _q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor):
        x0 = x0.to(self.device)  # Ensure x0 is on the correct device
        t = t.to(self.device)  # Ensure t is on the correct device
        mean = self._sqrt(self._alpha_bar(t)) * x0
        std = self._sqrt(1 - self._alpha_bar(t))
        return mean, std

    def normal(self, shape: tuple, mean: float = 0.0, std: float = 1.0, seed: int = None):
        reset_seed(seed)
        return torch.empty(shape, dtype=torch.float32, device=self.device, requires_grad=False).normal_(mean, std)

    def _normal_like(self, x0: torch.Tensor, mean: torch.Tensor = 0.0, std: torch.Tensor = 1.0, seed: int = None):
        reset_seed(seed)
        return torch.empty_like(x0, dtype=torch.float32, device=self.device, requires_grad=False).normal_(mean, std)

    def _sqrt(self, x: torch.Tensor):
        return torch.sqrt(torch.clamp(x, min=0.0)) + 1e-8

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor = None, seed: int = None):
        x0 = x0.to(self.device)  # Ensure x0 is on the correct device
        t = t.to(self.device)  # Ensure t is on the correct device
        if eps is None:
            eps = self._normal_like(x0, seed=seed)
        mean, std = self._q_xt_x0(x0, t)
        return mean + std * eps

    def q_sample_rand_t(self, x0: torch.Tensor, noise: torch.Tensor = None):
        x0 = x0.to(self.device)  # Ensure x0 is on the correct device
        batch_size = x0.shape[0]
        t = torch.randint(low=1, high=self._T + 1, size=(batch_size,), dtype=torch.int64, device=self.device)
        if noise is None:
            noise = self._normal_like(x0)
        xt = self.q_sample(x0, t, eps=noise)
        return xt, t

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        return self.p_sample_guided(xt, t, noise, cond_gradient=None)

    def p_sample_guided(self, xt: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None, cond_gradient: torch.Tensor = None):
        eps_theta = self.eps_model(xt, t)
        alpha_bar = self._alpha_bar(t)
        alpha = self._alpha(t)
        beta = self._beta(t)
        eps_coef = (1 - alpha) / self._sqrt(1 - alpha_bar)
        mean = 1 / self._sqrt(alpha) * (xt - eps_coef * eps_theta)
        std = self._sqrt(beta)
        if noise is None:
            noise = self._normal_like(xt)
        if cond_gradient is None:
            return mean + std * noise
        else:
            return mean + std * cond_gradient + std * noise

    def loss(self, x0: torch.Tensor, noise: torch.Tensor = None):
        x0 = x0.to(self.device)  # Ensure x0 is on the correct device
        if noise is None:
            noise = self._normal_like(x0)
        xt, t = self.q_sample_rand_t(x0, noise)
        eps_theta = self.eps_model(xt, t)
        return F.mse_loss(noise, eps_theta)

    def loss_guided(self, x0: torch.Tensor, noise: torch.Tensor = None, classifier: nn.Module = None, out_channels: int = 1):
        x0 = x0.to(self.device)  # Ensure x0 is on the correct device
        if noise is None:
            noise = self._normal_like(x0)
        xt, t = self.q_sample_rand_t(x0, noise)
        eps_theta = self.eps_model(xt, t)
        class_pred = classifier(xt, t).view(-1, out_channels, 1, 1)
        return F.mse_loss(noise, eps_theta * class_pred)
