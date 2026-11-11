import argparse
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')     # pip torch + Anaconda MKL in one process (see train_ppo.py)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from gym_wrapper import HelioAimingGym
from policy_network import HeliostatPolicy


#--- One episode with a policy (sampled or most probable actions) or with the starting aim points held fixed (policy = None); returns a log per step
def run_episode(env, policy = None, start = 'spread', seed = 0, deterministic = True):
    obs, _ = env.reset(seed = seed, options = dict(start = start))
    hidden = policy.initial_hidden(1) if policy else None
    mask = torch.ones(1)
    rows = []
    while True:
        if policy:
            action, _, _, hidden = policy.act(torch.as_tensor(obs[None], dtype = torch.float32), hidden, mask, deterministic = deterministic)
            action = action[0].numpy()
        else:
            action = env.env.choice                                            # Keep the starting aim points all episode
        n_changed = int(np.sum(action != env.env.choice))
        obs, reward, terminated, truncated, info = env.step(action)
        rows.append(dict(hour = env.env.hour, reward = reward, why = info['why'], n_changed = n_changed,
                         **{k: v for k, v in info.items() if np.isscalar(v) and not isinstance(v, str) and k != 'reward'}))
        if terminated or truncated:
            break
    return pd.DataFrame(rows)


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#--- Compare a trained policy with the simple aiming it starts from (every heliostat on the aperture center, left there), same day, hour and rays
#      python evaluate_episode.py runs/night_08_nolimit/checkpoint.pt --wall-limit 1500 --stochastic
#      add --rule for the image-size rule as a third line; writes eval_<name>.csv and eval_agent_vs_center.png
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint')
    p.add_argument('--day', type = int, default = 355)
    p.add_argument('--hour', type = float, default = 10.0)
    p.add_argument('--minutes', type = int, default = 180)
    p.add_argument('--dt-min', type = float, default = 5.0)
    p.add_argument('--wall-limit', type = float, default = 1200.0, help = 'C; use the value the agent was trained with (night_08_nolimit: 1500)')
    p.add_argument('--nray', type = float, default = 1e6, help = 'rays per trace; 1e6 for reporting')
    p.add_argument('--rule', action = 'store_true', help = 'also run the image-size rule (spread by distance) as a third line')
    p.add_argument('--stochastic', action = 'store_true', help = 'sample the actions as in training instead of taking the most probable point; the fair view of an undecided policy')
    args = p.parse_args()

    #--- Policy from the checkpoint
    state = torch.load(args.checkpoint, map_location = 'cpu', weights_only = False)
    policy = HeliostatPolicy(state['positions'], state['n_points'], n_plant = state['n_plant'], embed = state['embed'], hidden = state['hidden'])
    policy.load_state_dict(state['policy'])
    policy.eval()

    #--- Environment on the chosen day and hour
    env = HelioAimingGym(env_kwargs = dict(dt_min = args.dt_min, nray = args.nray, episode_minutes = args.minutes, wall_limit_c = args.wall_limit),
                         reset_kwargs = dict(day = args.day, hour_range = (args.hour, args.hour)))
    assert policy.n == env.env.n, f'checkpoint was trained with {policy.n} heliostats, this field has {env.env.n}'
    assert state['n_points'] == len(env.env.grid), f"checkpoint was trained with {state['n_points']} aim points, the menu now has {len(env.env.grid)}"

    #--- The episodes: all on center (fixed), the agent from the center, optionally the image-size rule
    runs = [('all on the center point, fixed', 'center', run_episode(env, None, start = 'center')),
            ('trained agent, from the center point' + (' (sampled actions)' if args.stochastic else ''), 'agent', run_episode(env, policy, start = 'center', deterministic = not args.stochastic))]
    if args.rule:
        runs.insert(0, ('image-size rule, fixed', 'rule', run_episode(env, None, start = 'spread')))
    for label, name, log in runs:
        log.to_csv(f'eval_{name}.csv', index = False)
    agent = runs[-1][2]

    #--- Figure: reward, the terms, motor travel, aim points changed, SNOUT maximum, step by step
    fig, axes = plt.subplots(6, 1, figsize = (12, 16), sharex = True)
    for ax, column, ylabel in zip(axes, ['reward', 'optical_efficiency', 'temperature_unevenness', 'motor_travel', 'n_changed', 'snout_wall_max_c'],
                                  ['reward', 'optical efficiency', 'temperature unevenness', 'motor travel\n(0 still, 1 full speed)',
                                   'aim points changed\nin the step (of 218)', 'SNOUT wall max (C)']):
        for label, name, log in runs:
            if column == 'n_changed':
                ax.bar(log.hour, log[column], width = 0.025, alpha = 0.6, label = label)
            else:
                ax.plot(log.hour, log[column], 'o-', ms = 3, label = label)
        ax.set_ylabel(ylabel); ax.grid(alpha = 0.3)
    axes[3].text(0.01, 0.95, 'tracking the sun alone costs about 0.02; aim-point changes show as spikes above it', transform = axes[3].transAxes, va = 'top', fontsize = 9, color = 'gray')
    axes[-1].axhline(1200, color = 'k', ls = '--', label = '1200 C stop')
    axes[0].legend(loc = 'upper left', bbox_to_anchor = (1.01, 1.0), frameon = False)
    axes[-1].set_xlabel('solar hour')
    axes[0].set_title(f'Day {args.day}, start {args.hour:.1f} h: agent vs all on the center point')
    fig.savefig('eval_agent_vs_center.png', dpi = 150, bbox_inches = 'tight')

    #--- Summary per episode
    for label, name, log in runs:
        print(f"{name:6s}: mean reward {log.reward.mean():.3f}, efficiency {log.optical_efficiency.mean():.3f}, unevenness {log.temperature_unevenness.mean():.3f}, "
              f"motor travel {log.motor_travel.mean():.4f} (reward cost {0.05 * log.motor_travel.mean():.4f}), aim points changed per step {log.n_changed.mean():.1f}, "
              f"max wall {max(log.wall_max_c.max(), log.snout_wall_max_c.max()):.0f} C, ended: {log.why.iloc[-1]}")
    print(f"agent changed on average {agent.n_changed.mean():.1f} aim points per step")