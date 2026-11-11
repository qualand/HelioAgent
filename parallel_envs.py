import numpy as np
import multiprocessing as mp


#--- One environment in its own process: builds its field once, then answers step / reset / close over a pipe.
#    Episodes restart by themselves when they end, so the training loop never waits.
def worker(connection, env_kwargs, reset_kwargs, seed):
    from gym_wrapper import HelioAimingGym
    env = HelioAimingGym(env_kwargs = env_kwargs, reset_kwargs = reset_kwargs)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed = int(rng.integers(1 << 30)))
    connection.send(dict(obs = obs, positions = env.env.kin.pos, n = env.env.n, n_points = len(env.env.grid)))
    while True:
        command, data = connection.recv()
        if command == 'step':
            obs, reward, terminated, truncated, info = env.step(data)
            done = terminated or truncated
            if done:
                obs, _ = env.reset(seed = int(rng.integers(1 << 30)))        # The obs returned is the first of the next episode
            connection.send((obs, reward, done, info))
        elif command == 'reset':
            obs, _ = env.reset(seed = int(rng.integers(1 << 30)), options = data)
            connection.send(obs)
        elif command == 'close':
            connection.close()
            break


#--- N copies of the environment stepped together. Each traces with one thread, so N environments use N cores: a laptop runs 4 to 8, a Kestrel node about 100
class ParallelEnvs:
    def __init__(self, n_envs, env_kwargs = None, reset_kwargs = None, seed = 0):
        context = mp.get_context('spawn')                                     # Works the same on Windows, Linux and the HPC
        self.connections, self.processes = [], []
        for k in range(n_envs):
            parent, child = context.Pipe()
            process = context.Process(target = worker, args = (child, env_kwargs or {}, reset_kwargs or {}, seed + k), daemon = True)
            process.start()
            self.connections.append(parent)
            self.processes.append(process)
        first = [connection.recv() for connection in self.connections]       # Waits for every field to build
        self.obs = np.stack([f['obs'] for f in first])
        self.positions = first[0]['positions']
        self.n_heliostats = first[0]['n']
        self.n_points = first[0]['n_points']
        self.n_envs = n_envs
        return

    #--- Step all environments: actions (n_envs, n_heliostats) ints -> obs (n_envs, obs_dim), rewards, dones, infos
    def step(self, actions):
        for connection, action in zip(self.connections, actions):
            connection.send(('step', np.asarray(action, dtype = int)))
        results = [connection.recv() for connection in self.connections]
        obs, rewards, dones, infos = zip(*results)
        self.obs = np.stack(obs)
        return self.obs, np.array(rewards, dtype = np.float32), np.array(dones, dtype = bool), list(infos)

    def reset(self, options = None):
        for connection in self.connections:
            connection.send(('reset', options))
        self.obs = np.stack([connection.recv() for connection in self.connections])
        return self.obs

    def close(self):
        for connection in self.connections:
            connection.send(('close', None))
        for process in self.processes:
            process.join(timeout = 5)