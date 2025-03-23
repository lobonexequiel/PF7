# %%
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import trange

torch.backends.cudnn.benchmark = True

import pickle

import seaborn as sns

with open('data/path.pkl', 'rb') as f:
    path = pickle.load(f)
X = np.array(path)
X = torch.tensor(X, dtype=torch.float32)
X = (X[1000:, ]-X.mean())/X.std()
X_copy = X.clone()

# %%
fig, ax1 = plt.subplots()
sample = 0
# First vector (primary y-axis)
ax1.plot(np.sin(X[:1000, 1].detach().cpu()), label='tita2_new', color='b')
ax1.set_ylabel('tita2_new', color='b')
ax1.tick_params(axis='y', labelcolor='b')

fig.legend(loc='upper right')

plt.show()


# %%
class ForwardProcess(nn.Module):
    def __init__(self, betas: torch.Tensor):
        super().__init__()
        self.beta = betas

        self.alphas = 1. - betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=-1)

    def get_x_t(self, x_0: torch.Tensor, t: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion process given the unperturbed sample x_0.

        Args:
            x_0: Original, unperturbed samples.
            t: Target timestamp of the diffusion process of each sample.

        Returns:
            Noise added to original sample and perturbed sample.
        """
        eps_0 = torch.randn_like(x_0).to(x_0)
        # eps_0 /= eps_0.max()#.to('cuda')
        alpha_bar = self.alpha_bar[t, None]  # .unsqueeze(1)#.expand(-1, 2, 1000)
        mean = (alpha_bar ** 0.5) * x_0
        std = ((1. - alpha_bar) ** 0.5)
        return (eps_0, mean + std * eps_0)


class NoisePredictor(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T
        self.t_encoder = nn.Linear(T, 1)
        #
        # self.model = nn.Sequential(
        #     nn.Conv1d(2 + 1, 16, kernel_size=2),  # Input: Noisy data x_t and t
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv1d(16, 8, kernel_size=2),
        #     nn.LeakyReLU(inplace=True),
        #     # Output: Predicted noise that was added to the original data point
        #     nn.Conv1d(8, 2, kernel_size=2),
        # )
        # self.model = nn.Sequential(
        #     nn.Linear(2 + 1, 6),  # Input: Noisy data x_t and t
        #     nn.ReLU(inplace=True),
        #     nn.LayerNorm(6, eps=1e-6),
        #     nn.Linear(6, 2),
        #     # nn.ReLU(inplace=True),
        #     # Output: Predicted noise that was added to the original data point
        #     # nn.Linear(3, 2),
        #
        #     # nn.ReLU(inplace=True),
        #     # nn.Linear(5, 4),
        # )
        self.model = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x_t, t):
        # Encode the time index t as one-hot and then use one layer to encode
        # into a single value
        self.t_embedding = self.t_encoder(
            nn.functional.one_hot(t - 1, num_classes=self.T).to(torch.float)
        )  # .reshape(-1, 1, 1)
        # self.t_embedding = self.t_encoder(t)
        # self.t_embedding = self.t_embedding.unsqueeze(1).expand(-1, 1000, 1).permute(0, 2, 1)
        inp = torch.cat([x_t, self.t_embedding], dim=-1).to(x_t)
        return self.model(inp)


# %%
epochs = 300
T = 200  # Number of diffusion steps
# betas = torch.linspace(0.0001, 0.2, T + 1, device='cuda') ** 4
betas = torch.linspace(1e-4, 0.02, T+1, device='cuda')
fp = ForwardProcess(betas=betas)
N = X.shape[0]

# %%
model = NoisePredictor(T=T).to('cuda')
optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=2e-4)
loss_history = []
X = X.to('cuda')
for epoch in trange(epochs):
    with torch.no_grad():
        t = torch.randint(low=1, high=T + 1, size=(N,), device='cuda')
        # Get the noise added and the noisy version of the data using the forward
        # process given t
        eps_0, x_t = fp.get_x_t(X, t=t)
    # Predict the noise added to x_0 from x_t
    # x_t = x_t.to('cuda')--

    pred_eps = model(x_t, t)

    # Simplified objective without weighting with alpha terms (Ho et al, 2020)
    # alpha_t = fp.alpha_bar[t, None]
    # loss = torch.nn.functional.mse_loss(pred_eps * alpha_t, eps_0 * alpha_t)
    loss = torch.nn.functional.mse_loss(pred_eps, eps_0)
    loss_history.append(float(loss))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
plt.title(f'Epoch {epoch}')
plt.plot(loss_history)
plt.savefig('loss_history.png')
plt.show()
# %%
for step in range(20, 150, 20):
    X_cpu = X.cpu()
    t = torch.ones(N, device='cuda') * step
    eps, x_t = fp.get_x_t(X, t=t.long())
    pred_eps = model(x_t, t.long()).cpu()
    X_pred = x_t.cpu() - pred_eps

    plt.title(f'T={step}')
    plt.ylabel(r'θ_2')
    plt.xlabel(r'Initial state')
    plt.xticks(range(20))
    plt.plot(X_cpu[:1000, 0], '-', label=r'$X_0$')
    plt.plot(X_pred[:1000, 0].detach().cpu(), '.', label=rf'$X_{{{step}}} - \epsilon_{{{step}}}(X_0, {step})$')
    # plt.plot(pred_eps[:, 2], '.g', alpha=.3, label=r'$\theta_t$')
    plt.legend()
    plt.show()

    sns.kdeplot(pred_eps[:, 0].detach().cpu(), label="Predicted ε")
    sns.kdeplot(eps[:, 0].detach().cpu(), label="True ε")

    plt.legend()
    plt.show()
# %%
t = torch.ones(N, device='cuda') * 100
eps, _ = fp.get_x_t(X, t=t.long())
pred_eps = model(X, t.long()).cpu()

sns.kdeplot(pred_eps[:, 0].detach().cpu(), label="Predicted ε")
sns.kdeplot(eps[:, 0].detach().cpu(), label="True ε")

plt.legend()
plt.show()


# %%
class ReverseProcess(ForwardProcess):
    def __init__(self, betas: torch.Tensor, model: nn.Module):
        super().__init__(betas)
        self.model = model
        self.T = len(betas) - 1

        self.sigma = (
                             (1 - self.alphas)
                             * (1 - torch.roll(self.alpha_bar, 1)) / (1 - self.alpha_bar)
                     ) ** 0.5
        self.sigma[1] = 0.

    def get_x_t_minus_one(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        with torch.no_grad():
            t_vector = torch.full(size=(len(x_t),), fill_value=t, dtype=torch.long, device='cuda')
            eps = self.model(x_t, t=t_vector)

        eps *= (1 - self.alphas[t]) / ((1 - self.alpha_bar[t]) ** 0.5)
        mean = 1 / (self.alphas[t] ** 0.5) * (x_t - eps)
        return mean + self.sigma[t] * torch.randn_like(x_t)

    def sample(self, n_samples=1, full_trajectory=False):
        # Initialize with X_T ~ N(0, I)
        x_t = torch.randn(1000, 2, device='cuda')
        trajectory = [x_t.clone()]

        for t in range(n_samples, 0, -1):
            x_t = self.get_x_t_minus_one(x_t, t=t)

            if full_trajectory:
                trajectory.append(x_t.clone())
        return torch.stack(trajectory, dim=0) if full_trajectory else x_t


# %%
rp = ReverseProcess(betas=betas, model=model)
samples = rp.sample(n_samples=10, ).cpu().numpy()

plt.plot(samples[:, 0], '-b', label=r'$\theta_t$')
plt.legend()
plt.show()
