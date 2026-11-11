import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


#--- Linear layers with an activation between them and none after the last
def mlp(sizes, activation = nn.Tanh):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), activation()]
    return nn.Sequential(*layers[:-1])


#--- The agent's network: one shared encoder serves every heliostat, a GRU keeps a memory of the episode, each mirror gets its own choice
class HeliostatPolicy(nn.Module):
    def __init__(self, positions, n_points = 12, n_plant = 11, embed = 64, hidden = 128):
        super().__init__()

        #--- Fixed per-heliostat features: position (m) and distance to the tower, scaled to about 0..1
        positions = np.asarray(positions, dtype = np.float32)
        distance = np.linalg.norm(positions[:, :2], axis = 1, keepdims = True)
        static = np.c_[positions / 300.0, distance / 300.0]
        self.register_buffer('static', torch.as_tensor(static))               # (n, 4), moves with the model
        self.n = len(positions)                                               # Heliostats
        self.n_points = n_points                                              # Menu points per heliostat
        self.n_plant = n_plant                                                # Plant-level observation length
        self.hidden = hidden                                                  # GRU memory size

        #--- Actor: mirror encoder (shared weights) + plant encoder -> GRU memory -> one head applied to every mirror
        self.mirror_net = nn.Sequential(mlp([2 + 4, embed, embed]), nn.Tanh())   # [aim dx, aim dv] + the 4 static features -> embedding e_i
        self.plant_net = nn.Sequential(mlp([n_plant, embed]), nn.Tanh())         # [hour, sun, DNI, power, peak flux, intercept, T out, walls, column T's] -> g
        self.gru = nn.GRUCell(2 * embed, hidden)                                 # Memory over [mean_i e_i, g], carried between steps
        self.actor = mlp([embed + hidden, embed, n_points])                      # logits_i = head([e_i, h]): n categorical distributions of n_points

        #--- Critic: its own encoders, no memory. Sharing the actor's mirror network wipes out the per-mirror features the actor needs (tested: no learning)
        self.critic_mirror_net = nn.Sequential(mlp([2 + 4, embed, embed]), nn.Tanh())
        self.critic_plant_net = nn.Sequential(mlp([n_plant, embed]), nn.Tanh())
        self.critic = mlp([2 * embed, embed, 1])                                 # value = v([mean_i e'_i, g'])
        for layer in (self.actor[-1], self.critic[-1]):
            layer.weight.data.mul_(0.01)                                         # Small initial logits: near-uniform start
        return

    #--- Split the scaled observation of gym_wrapper into per-mirror (B, n, 2) and plant (B, n_plant) parts
    def split(self, obs):
        offsets = obs[:, 4:4 + 2 * self.n].reshape(-1, self.n, 2)
        plant = torch.cat([obs[:, :4], obs[:, 4 + 2 * self.n:]], dim = 1)
        return offsets, plant

    def initial_hidden(self, batch, device = None):
        return torch.zeros(batch, self.hidden, device = device)

    #--- One time step for a batch of environments: obs (B, obs_dim), hidden (B, hidden), mask (B,) = 0 where the episode just ended (memory wiped)
    def forward(self, obs, hidden, mask):
        offsets, plant = self.split(obs)
        mirror_in = torch.cat([offsets, self.static.unsqueeze(0).expand(offsets.shape[0], -1, -1)], dim = 2)
        embeddings = self.mirror_net(mirror_in)                                 # (B, n, embed)
        pooled = embeddings.mean(dim = 1)                                      # Order-free summary of the field
        hidden = self.gru(torch.cat([pooled, self.plant_net(plant)], dim = 1), hidden * mask.unsqueeze(1))
        context = hidden.unsqueeze(1).expand(-1, self.n, -1)
        logits = self.actor(torch.cat([embeddings, context], dim = 2))         # (B, n, n_points)
        critic_pooled = self.critic_mirror_net(mirror_in).mean(dim = 1)
        value = self.critic(torch.cat([critic_pooled, self.critic_plant_net(plant)], dim = 1)).squeeze(1)   # (B,)
        return logits, value, hidden

    def distribution(self, logits):
        return Categorical(logits = logits)                                    # Batch of (B, n) independent choices

    #--- Act: sample one menu point per mirror (or take the most probable one); returns the per-mirror log-probabilities too
    @torch.no_grad()
    def act(self, obs, hidden, mask, deterministic = False):
        logits, value, hidden = self(obs, hidden, mask)
        dist = self.distribution(logits)
        action = logits.argmax(dim = 2) if deterministic else dist.sample()    # (B, n)
        log_prob = dist.log_prob(action)                                       # (B, n): one log-probability per mirror
        return action, log_prob, value, hidden

    #--- Re-run a stored sequence for the PPO update: obs_seq (T, B, obs_dim), hidden0 (B, hidden), mask_seq (T, B), action_seq (T, B, n)
    #    Returns per-mirror log-probabilities and entropies (T, B, n) and values (T, B). The PPO ratio is clipped per mirror (as in MAPPO,
    #    Yu et al. 2022): the joint ratio over 218 mirrors would swing by orders of magnitude and make the clipping useless.
    def evaluate(self, obs_seq, hidden0, mask_seq, action_seq):
        log_probs, entropies, values = [], [], []
        hidden = hidden0
        for t in range(obs_seq.shape[0]):
            logits, value, hidden = self(obs_seq[t], hidden, mask_seq[t])
            dist = self.distribution(logits)
            log_probs.append(dist.log_prob(action_seq[t]))
            entropies.append(dist.entropy())
            values.append(value)
        return torch.stack(log_probs), torch.stack(entropies), torch.stack(values)