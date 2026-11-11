import argparse
import csv
import os
import time
import numpy as np
from parallel_envs import ParallelEnvs

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')   # pip torch (libomp) and Anaconda MKL (libiomp5md) in one process; torch is imported inside main()
                                                        # because on Windows the environment processes re-import this file when they start


#--- Command line. Recurrent PPO (Schulman et al. 2017) in plain PyTorch; one update = n_steps from every environment, then a few epochs of
#    clipped policy-gradient steps on those steps. Training ends after --total-steps; continue a run with --resume runs/<name>/checkpoint.pt.
#      python train_ppo.py --n-envs 8 --n-steps 24 --total-steps 100000 --lr 1e-3 --out runs/night_09
#      python train_ppo.py --n-envs 96 --n-steps 24 --total-steps 1000000 --lr 1e-3 --out runs/kestrel_01      (one Kestrel node)
#      tensorboard --logdir runs
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n-envs', type = int, default = 4)
    p.add_argument('--n-steps', type = int, default = 24, help = 'steps per environment per update (24 x 5 min = 2 h)')
    p.add_argument('--total-steps', type = int, default = 5000, help = 'environment steps in total, all environments together')
    p.add_argument('--epochs', type = int, default = 4)
    p.add_argument('--minibatches', type = int, default = 2, help = 'environments are split into this many minibatches per epoch')
    p.add_argument('--lr', type = float, default = 3e-4)
    p.add_argument('--gamma', type = float, default = 0.9, help = 'credit horizon ~10 steps = 50 min, the SNOUT thermal lag; the optics responds within one step')
    p.add_argument('--lam', type = float, default = 0.9)
    p.add_argument('--clip', type = float, default = 0.2)
    p.add_argument('--ent-coef', type = float, default = 0.03, help = 'entropy bonus; 0.01 let the policy freeze in 30 updates with the local credit (night_04)')
    p.add_argument('--vf-coef', type = float, default = 0.5)
    p.add_argument('--max-grad-norm', type = float, default = 0.5)
    p.add_argument('--local-coef', type = float, default = 0.5, help = 'weight of the per-heliostat credit in the advantage; 1.0 let it override the shared reward (night_04), 0 = shared advantage only')
    p.add_argument('--hidden', type = int, default = 128)
    p.add_argument('--embed', type = int, default = 64)
    p.add_argument('--dt-min', type = float, default = 5.0)
    p.add_argument('--wall-limit', type = float, default = 1200.0, help = 'C; the episode ends with the shutdown penalty above it (receiver walls and SNOUT). Raise it to train without the SNOUT constraint')
    p.add_argument('--nray', type = float, default = 1e6, help = 'rays per trace; on the 5 cm mesh the SNOUT maximum reads +30 C of noise at 5e5 and +20 C at 1e6 (snout_mesh_convergence.py)')
    p.add_argument('--episode-minutes', type = int, default = 120, help = '2 h of regular operation; day, start time and start aiming come from AimingEnv.reset')
    p.add_argument('--seed', type = int, default = 0)
    p.add_argument('--out', default = 'runs/ppo')
    p.add_argument('--checkpoint-every', type = int, default = 10, help = 'updates between checkpoints')
    p.add_argument('--resume', default = '', help = 'checkpoint to continue from')
    return p.parse_args()


#--- Running standard deviation of the discounted return, used to scale the rewards the critic sees (as in SB3's VecNormalize).
#    Keeps the value targets near 1 whatever gamma is; the logged rewards stay raw.
class ReturnScale:
    def __init__(self, n_envs, gamma):
        self.gamma, self.running = gamma, np.zeros(n_envs, np.float32)
        self.count, self.mean, self.var = 1e-4, 0.0, 1.0

    def update(self, rewards, dones):
        self.running = self.gamma * self.running * (1.0 - dones) + rewards
        batch_mean, batch_var, n = self.running.mean(), self.running.var(), len(rewards)
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean += delta * n / total
        self.var = (self.var * self.count + batch_var * n + delta ** 2 * self.count * n / total) / total
        self.count = total

    @property
    def std(self):
        return float(np.sqrt(self.var) + 1e-8)


#--- Generalized advantage estimation: rewards, values, dones (T, B), last_value (B,); dones[t] = the episode ended at step t
def compute_gae(rewards, values, dones, last_value, gamma, lam):
    T = rewards.shape[0]
    advantages = np.zeros_like(rewards)
    running = np.zeros(rewards.shape[1], dtype = np.float32)
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        continues = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * continues - values[t]
        running = delta + gamma * lam * continues * running
        advantages[t] = running
    return advantages, advantages + values


def main():
    global torch
    import torch
    from policy_network import HeliostatPolicy
    args = parse_args()
    os.makedirs(args.out, exist_ok = True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #--- Environments: one process each, every one builds its own field
    env_kwargs = dict(dt_min = args.dt_min, nray = args.nray, episode_minutes = args.episode_minutes, wall_limit_c = args.wall_limit, trace_seed = None)
    print(f'starting {args.n_envs} environments (each builds its own field)...', flush = True)
    envs = ParallelEnvs(args.n_envs, env_kwargs, reset_kwargs = {}, seed = args.seed)
    obs_dim = envs.obs.shape[1]
    n_plant = obs_dim - 2 * envs.n_heliostats
    print(f'ready: {envs.n_heliostats} heliostats, {envs.n_points} aim points, observation {obs_dim}', flush = True)

    #--- Policy, optimizer, resume
    policy = HeliostatPolicy(envs.positions, envs.n_points, n_plant = n_plant, embed = args.embed, hidden = args.hidden).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr = args.lr, eps = 1e-5)
    actor_params = [p for name, p in policy.named_parameters() if not name.startswith('critic')]
    critic_params = [p for name, p in policy.named_parameters() if name.startswith('critic')]
    return_scale = ReturnScale(args.n_envs, args.gamma)
    update, steps_done = 0, 0
    if args.resume:
        state = torch.load(args.resume, map_location = device, weights_only = False)
        policy.load_state_dict(state['policy']); optimizer.load_state_dict(state['optimizer'])
        update, steps_done = state['update'], state['steps']
        print(f'resumed from {args.resume} at update {update}, {steps_done} steps')

    #--- Logging: log.csv (one row per update) and tensorboard if installed
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(args.out)
    except ImportError:
        writer = None
    log_file = open(os.path.join(args.out, 'log.csv'), 'a', newline = '')
    log = csv.writer(log_file)
    if log_file.tell() == 0:
        log.writerow(['update', 'steps', 'seconds', 'reward_mean', 'episode_return', 'optical_efficiency', 'on_setpoint',
                      'temperature_unevenness', 'motor_travel', 'wall_max_c', 'snout_wall_max_c', 'intercept', 'policy_loss', 'value_loss', 'entropy', 'approx_kl'])

    obs = torch.as_tensor(envs.obs, dtype = torch.float32, device = device)
    hidden = policy.initial_hidden(args.n_envs, device)
    mask = torch.ones(args.n_envs, device = device)
    episode_returns, running_return = [], np.zeros(args.n_envs, dtype = np.float32)
    T, B = args.n_steps, args.n_envs
    started = time.perf_counter()

    while steps_done < args.total_steps:

        #--- Collect T steps from every environment, the GRU memory carried along
        buf_obs = torch.zeros(T, B, obs_dim, device = device)
        buf_act = torch.zeros(T, B, envs.n_heliostats, dtype = torch.long, device = device)
        buf_logp = torch.zeros(T, B, envs.n_heliostats, device = device)
        buf_mask = torch.zeros(T, B, device = device)
        buf_val, buf_rew, buf_done = np.zeros((T, B), np.float32), np.zeros((T, B), np.float32), np.zeros((T, B), np.float32)
        buf_local = np.zeros((T, B, envs.n_heliostats), np.float32)                 # Per-heliostat credit of every step
        hidden0 = hidden.clone()                                                     # Memory at the start of the batch, re-run during the update
        term_sums = {}
        for t in range(T):
            action, log_prob, value, hidden = policy.act(obs, hidden, mask)
            buf_obs[t], buf_act[t], buf_logp[t], buf_mask[t], buf_val[t] = obs, action, log_prob, mask, value.cpu().numpy()
            next_obs, rewards, dones, infos = envs.step(action.cpu().numpy())
            buf_rew[t], buf_done[t] = rewards, dones
            return_scale.update(rewards, dones.astype(np.float32))
            running_return += rewards
            for k, done in enumerate(dones):
                if done:
                    episode_returns.append(running_return[k]); running_return[k] = 0.0
            buf_local[t] = np.stack([info['heliostat_credit'] for info in infos])
            for info in infos:
                for key in ('optical_efficiency', 'on_setpoint', 'temperature_unevenness', 'motor_travel', 'wall_max_c', 'snout_wall_max_c', 'intercept'):
                    term_sums[key] = term_sums.get(key, 0.0) + info[key]
            obs = torch.as_tensor(next_obs, dtype = torch.float32, device = device)
            mask = torch.as_tensor(1.0 - dones, dtype = torch.float32, device = device)   # Wipe the memory where an episode ended

        #--- Advantage per mirror: the shared advantage of the step plus the mirror's own credit, both standardized
        with torch.no_grad():
            _, last_value, _ = policy(obs, hidden, mask)
        advantages, returns = compute_gae(buf_rew / return_scale.std, buf_val, buf_done, last_value.cpu().numpy(), args.gamma, args.lam)
        advantages = torch.as_tensor(advantages, device = device)
        returns = torch.as_tensor(returns, device = device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        local = buf_local - buf_local.mean(axis = (0, 1), keepdims = True) - buf_local.mean(axis = 2, keepdims = True) + buf_local.mean()   # Each mirror's usual level and the step's field level removed
        local = torch.as_tensor(local / (local.std() + 1e-8), device = device)
        advantages_mirror = advantages.unsqueeze(2) + args.local_coef * local        # (T, B, n)
        steps_done += T * B

        #--- Update: a few epochs of clipped policy gradient, minibatches of whole sequences, one ratio per mirror
        stats = dict(policy_loss = 0.0, value_loss = 0.0, entropy = 0.0, approx_kl = 0.0)
        n_batches = 0
        for epoch in range(args.epochs):
            for env_index in np.array_split(np.random.permutation(B), args.minibatches):
                idx = torch.as_tensor(env_index, device = device)
                new_logp, entropy, new_values = policy.evaluate(buf_obs[:, idx], hidden0[idx], buf_mask[:, idx], buf_act[:, idx])
                ratio = torch.exp(new_logp - buf_logp[:, idx])                     # (T, b, n)
                adv = advantages_mirror[:, idx]                                     # (T, b, n)
                policy_loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv).mean()
                value_loss = 0.5 * (new_values - returns[:, idx]).pow(2).mean()
                loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy.mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_params, args.max_grad_norm)      # Clipped separately: a large critic gradient
                torch.nn.utils.clip_grad_norm_(critic_params, args.max_grad_norm)     # must not shrink the actor's step
                optimizer.step()
                with torch.no_grad():
                    stats['approx_kl'] += (buf_logp[:, idx] - new_logp).mean().item()
                stats['policy_loss'] += policy_loss.item(); stats['value_loss'] += value_loss.item(); stats['entropy'] += entropy.mean().item()
                n_batches += 1
        update += 1

        #--- Log: means over the T x B steps of this update; episode return = mean of the last 20 finished episodes
        seconds = time.perf_counter() - started
        terms = {key: value / (T * B) for key, value in term_sums.items()}
        recent_return = float(np.mean(episode_returns[-20:])) if episode_returns else float('nan')
        row = [update, steps_done, round(seconds), buf_rew.mean(), recent_return] + [terms[k] for k in
              ('optical_efficiency', 'on_setpoint', 'temperature_unevenness', 'motor_travel', 'wall_max_c', 'snout_wall_max_c', 'intercept')] + \
              [stats[k] / n_batches for k in ('policy_loss', 'value_loss', 'entropy', 'approx_kl')]
        log.writerow([f'{v:.4f}' if isinstance(v, float) else v for v in row]); log_file.flush()
        if writer:
            writer.add_scalar('reward/step_mean', buf_rew.mean(), steps_done)
            writer.add_scalar('reward/episode_return', recent_return, steps_done)
            for key, value in terms.items():
                writer.add_scalar(f'plant/{key}', value, steps_done)
            for key in stats:
                writer.add_scalar(f'train/{key}', stats[key] / n_batches, steps_done)
        print(f"update {update:4d}  steps {steps_done:7d}  {seconds / 60:6.1f} min  reward/step {buf_rew.mean():6.3f}  "
              f"episode return {recent_return:7.2f}  efficiency {terms['optical_efficiency']:.3f}  unevenness {terms['temperature_unevenness']:.3f}  intercept {terms['intercept']:.3f}  "
              f"entropy {stats['entropy'] / n_batches:.3f}  kl {stats['approx_kl'] / n_batches:.4f}", flush = True)
        if update % args.checkpoint_every == 0:
            torch.save(dict(policy = policy.state_dict(), optimizer = optimizer.state_dict(), update = update, steps = steps_done,
                            positions = np.asarray(envs.positions).tolist(), n_points = envs.n_points, n_plant = n_plant, embed = args.embed, hidden = args.hidden),
                       os.path.join(args.out, 'checkpoint.pt'))

    torch.save(dict(policy = policy.state_dict(), optimizer = optimizer.state_dict(), update = update, steps = steps_done,
                    positions = np.asarray(envs.positions).tolist(), n_points = envs.n_points, n_plant = n_plant, embed = args.embed, hidden = args.hidden),
               os.path.join(args.out, 'final.pt'))
    envs.close()
    print(f'done: {steps_done} steps in {(time.perf_counter() - started) / 3600:.2f} h')


if __name__ == '__main__':
    main()