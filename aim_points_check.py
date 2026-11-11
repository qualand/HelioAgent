import argparse
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')     # pip torch + Anaconda MKL in one process (see train_ppo.py)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from gym_wrapper import HelioAimingGym
from policy_network import HeliostatPolicy


#--- Where does the agent send the heliostats? One episode with a checkpoint, the menu point of every heliostat recorded at every step.
#    Prints how many heliostats sit on each point per step, the field colored by chosen point, and the menu with the counts.
#      python aim_points_check.py runs/night_08_nolimit/checkpoint.pt --wall-limit 1500 --stochastic
#    writes aim_points_per_step.csv, aim_points_field.png, aim_points_menu.png
p = argparse.ArgumentParser()
p.add_argument('checkpoint')
p.add_argument('--day', type = int, default = 355)
p.add_argument('--hour', type = float, default = 10.0)
p.add_argument('--minutes', type = int, default = 60)
p.add_argument('--wall-limit', type = float, default = 1200.0, help = 'C; use the value the agent was trained with (night_08_nolimit: 1500)')
p.add_argument('--nray', type = float, default = 2e5, help = 'the choices depend only weakly on the ray count; 2e5 keeps this quick')
p.add_argument('--stochastic', action = 'store_true', help = 'sample the actions as in training instead of taking the most probable one')
args = p.parse_args()

#--- Policy from the checkpoint
state = torch.load(args.checkpoint, map_location = 'cpu', weights_only = False)
policy = HeliostatPolicy(state['positions'], state['n_points'], n_plant = state['n_plant'], embed = state['embed'], hidden = state['hidden'])
policy.load_state_dict(state['policy'])
policy.eval()
print(f"checkpoint from update {state['update']}, {state['steps']} steps")

#--- One episode from the center start, recording the choice of every heliostat and the policy's probabilities before each action
env = HelioAimingGym(env_kwargs = dict(dt_min = 5.0, nray = args.nray, episode_minutes = args.minutes, wall_limit_c = args.wall_limit), reset_kwargs = dict(day = args.day, hour_range = (args.hour, args.hour)))
grid = env.env.grid
positions = np.asarray(env.env.kin.pos)
obs, _ = env.reset(seed = 0, options = dict(start = 'center'))
hidden, mask = policy.initial_hidden(1), torch.ones(1)
choices, hours = [env.env.choice.copy()], [env.env.hour]
probabilities = []
while True:
    with torch.no_grad():
        logits, _, _ = policy(torch.as_tensor(obs[None], dtype = torch.float32), hidden, mask)
        probabilities.append(torch.softmax(logits[0], dim = -1).numpy())                   # (218, 15) before acting
    action, _, _, hidden = policy.act(torch.as_tensor(obs[None], dtype = torch.float32), hidden, mask, deterministic = not args.stochastic)
    obs, reward, terminated, truncated, info = env.step(action[0].numpy())
    choices.append(env.env.choice.copy()); hours.append(env.env.hour)
    if terminated or truncated:
        break
choices = np.array(choices)                                                                    # (steps + 1, 218), row 0 = the center start

#--- Table: heliostats on each point, per step
counts = pd.DataFrame([np.bincount(c, minlength = len(grid)) for c in choices], index = [f'{h:5.2f} h' for h in hours])
counts.columns = [f'p{k}' for k in range(len(grid))]
counts['changed'] = [0] + [int((choices[k] != choices[k - 1]).sum()) for k in range(1, len(choices))]
print('\nmenu points (dx sideways, dv up, m):')
for k, (dx, dv) in enumerate(grid):
    print(f'  p{k:<2d} ({dx:+.2f}, {dv:+.2f})' + ('  <- aperture center' if k == env.env.center_point else ''))
print('\nheliostats on each point, per step (row 0 is the start):')
print(counts.to_string())
prob = np.array(probabilities)                                                                 # (steps, 218, 15)
print(f"\npolicy at the first step: mean probability per point {np.round(prob[0].mean(axis = 0), 3).tolist()}")
print(f"most probable point per heliostat: {np.bincount(prob[0].argmax(axis = 1), minlength = len(grid)).tolist()}  (uniform would be {1 / len(grid):.3f} each)")
counts.to_csv('aim_points_per_step.csv')

#--- Field maps: chosen point per heliostat at the first step after the start and at the last step
fig, axes = plt.subplots(1, 2, figsize = (14, 6))
for ax, step in zip(axes, (1, len(choices) - 1)):
    sc = ax.scatter(positions[:, 0], positions[:, 1], c = choices[step], cmap = 'tab20', vmin = 0, vmax = len(grid) - 1, s = 28)
    ax.plot(0, 0, 'k^', ms = 10)                                                             # The tower
    ax.set_aspect('equal'); ax.grid(alpha = 0.3); ax.set_xlabel('east (m)'); ax.set_ylabel('north (m)')
    ax.set_title(f'{hours[step]:.2f} h: menu point per heliostat ({len(np.unique(choices[step]))} points in use)')
cbar = fig.colorbar(sc, ax = axes, ticks = range(len(grid)), fraction = 0.03)
cbar.set_label('menu point (p7 = center)')
fig.suptitle(f"Day {args.day}, start {args.hour:.1f} h, {'sampled' if args.stochastic else 'most probable'} actions")
fig.savefig('aim_points_field.png', dpi = 150, bbox_inches = 'tight')

#--- The menu with the number of heliostats on each point, first step after the start
fig, ax = plt.subplots(figsize = (5, 6))
ax.add_patch(plt.Rectangle((-0.66, -0.66), 1.32, 1.32, fill = False, ls = '--', color = 'k'))   # The aperture
n = counts.iloc[1, :len(grid)].to_numpy()
ax.scatter(grid[:, 0], grid[:, 1], s = 20 + 12 * n, c = '#2E6F9E', alpha = 0.6)
for k, (dx, dv) in enumerate(grid):
    ax.text(dx, dv, f'p{k}\n{n[k]}', ha = 'center', va = 'center', fontsize = 9)
ax.set_aspect('equal'); ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.8, 0.8)
ax.set_xlabel('sideways on the aperture (m)'); ax.set_ylabel('up (m)')
ax.set_title(f'Heliostats per menu point at {hours[1]:.2f} h')
fig.savefig('aim_points_menu.png', dpi = 150, bbox_inches = 'tight')
print('\nwrote aim_points_per_step.csv, aim_points_field.png, aim_points_menu.png')