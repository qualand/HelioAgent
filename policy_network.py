# The agent's network, in PyTorch. One shared "brain" serves every heliostat (the same weights look at each
# mirror's own features), a GRU keeps a memory of the episode, and each mirror gets its own 12-way choice.
#
#   per mirror   : [aim dx, aim dv, position x, y, z, distance to tower]  -> shared MLP -> embedding e_i
#   whole plant  : [hour, sun az, sun el, DNI, power, peak flux, intercept, T out, walls, column T's] -> MLP -> g
#   memory       : GRU over [mean_i e_i, g]                               -> hidden h (carried between steps)
#   actor        : logits_i = head([e_i, h]) for every mirror i          -> 218 categorical distributions of 12
#   critic       : its own mirror and plant encoders (no memory), value = v([mean_i e'_i, g'])
#
# Observations are the scaled vectors from gym_wrapper.HelioAimingGym:
#   [hour, az, el, dni] + aim offsets (2 n) + [power, peak, intercept, T_out, T_wall, T_snout] + column T's

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def mlp(sizes, activation=nn.Tanh):
    # Linear layers with an activation between them and none after the last
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), activation()]
    return nn.Sequential(*layers[:-1])


class HeliostatPolicy(nn.Module):
    def __init__(self, positions, n_points=12, n_plant=11, embed=64, hidden=128):
        # positions: (n, 3) heliostat positions in m, fixed for the plant
        super().__init__()
        positions = np.asarray(positions, dtype=np.float32)
        distance = np.linalg.norm(positions[:, :2], axis=1, keepdims=True)
        static = np.c_[positions / 300.0, distance / 300.0]                     # scaled to about 0..1
        self.register_buffer('static', torch.as_tensor(static))               # (n, 4), moves with the model
        self.n = len(positions)
        self.n_points = n_points
        self.n_plant = n_plant
        self.hidden = hidden

        self.mirror_net = nn.Sequential(mlp([2 + 4, embed, embed]), nn.Tanh())  # shared across mirrors
        self.plant_net = nn.Sequential(mlp([n_plant, embed]), nn.Tanh())
        self.gru = nn.GRUCell(2 * embed, hidden)
        self.actor = mlp([embed + hidden, embed, n_points])
        # The critic has its own encoders: value targets are large and noisy early on, and sharing the actor's
        # mirror network with them wipes out the per-mirror features the actor needs (tested: no learning when shared).
        self.critic_mirror_net = nn.Sequential(mlp([2 + 4, embed, embed]), nn.Tanh())
        self.critic_plant_net = nn.Sequential(mlp([n_plant, embed]), nn.Tanh())
        self.critic = mlp([2 * embed, embed, 1])
        for layer in (self.actor[-1], self.critic[-1]):
            layer.weight.data.mul_(0.01)                                       # small initial logits: near-uniform start

    def split(self, obs):
        # obs: (B, 4 + 2n + n_plant - 4) -> per-mirror (B, n, 2) and plant (B, n_plant)
        offsets = obs[:, 4:4 + 2 * self.n].reshape(-1, self.n, 2)
        plant = torch.cat([obs[:, :4], obs[:, 4 + 2 * self.n:]], dim=1)
        return offsets, plant

    def initial_hidden(self, batch, device=None):
        return torch.zeros(batch, self.hidden, device=device)

    def forward(self, obs, hidden, mask):
        # One time step for a batch of environments.
        #   obs (B, obs_dim), hidden (B, hidden), mask (B,) = 0 where the episode just ended (memory wiped)
        offsets, plant = self.split(obs)
        mirror_in = torch.cat([offsets, self.static.unsqueeze(0).expand(offsets.shape[0], -1, -1)], dim=2)
        embeddings = self.mirror_net(mirror_in)                                 # (B, n, embed)
        pooled = embeddings.mean(dim=1)                                        # order-free summary of the field
        hidden = self.gru(torch.cat([pooled, self.plant_net(plant)], dim=1), hidden * mask.unsqueeze(1))
        context = hidden.unsqueeze(1).expand(-1, self.n, -1)
        logits = self.actor(torch.cat([embeddings, context], dim=2))           # (B, n, n_points)
        critic_pooled = self.critic_mirror_net(mirror_in).mean(dim=1)
        value = self.critic(torch.cat([critic_pooled, self.critic_plant_net(plant)], dim=1)).squeeze(1)   # (B,)
        return logits, value, hidden

    def distribution(self, logits):
        return Categorical(logits=logits)                                      # batch of (B, n) independent choices

    @torch.no_grad()
    def act(self, obs, hidden, mask, deterministic=False):
        logits, value, hidden = self(obs, hidden, mask)
        dist = self.distribution(logits)
        action = logits.argmax(dim=2) if deterministic else dist.sample()      # (B, n)
        log_prob = dist.log_prob(action)                                       # (B, n): one log-probability per mirror
        return action, log_prob, value, hidden

    def evaluate(self, obs_seq, hidden0, mask_seq, action_seq):
        # Re-run a stored sequence for the PPO update.
        #   obs_seq (T, B, obs_dim), hidden0 (B, hidden), mask_seq (T, B), action_seq (T, B, n)
        # Returns per-mirror log-probabilities and entropies (T, B, n) and values (T, B). The PPO ratio is
        # clipped per mirror with the shared advantage (as in MAPPO, Yu et al. 2022): the joint ratio over
        # 218 mirrors would swing by orders of magnitude and make the clipping useless.
        log_probs, entropies, values = [], [], []
        hidden = hidden0
        for t in range(obs_seq.shape[0]):
            logits, value, hidden = self(obs_seq[t], hidden, mask_seq[t])
            dist = self.distribution(logits)
            log_probs.append(dist.log_prob(action_seq[t]))
            entropies.append(dist.entropy())
            values.append(value)
        return torch.stack(log_probs), torch.stack(entropies), torch.stack(values)
