
# Drive the environment by hand, before any learning algorithm touches it.
# Part A: at one fixed sun, try a few spread patterns. Which one is safe?
# Part B: start a morning at center, move toward the safe pattern one
#         minute at a time, and plot power and peak over the episode.

import numpy as np
import matplotlib.pyplot as plt
from environment_class import AimingEnv
from functions_4_environment import set_aims, trace, sun_at

LIMIT = 1000.0   # kW/m2, the hard limit from step 4


def spread_pattern(env, s):
    """
    A hand-made aiming pattern with one knob, s, in meters.
    Far mirrors have wide beams, so they aim near the center.
    Near mirrors have tight beams, so they aim toward the edges.
    Every other mirror goes up, the rest go down, so the spread is balanced.
    s = 0 means everyone at center. s = 1.5 means near mirrors aim 1.5 m off center.
    """
    r = np.array([np.hypot(h.position.x, h.position.y) for h in env.field.heliostats])
    rank = np.argsort(np.argsort(-r))                 # 0 = farthest mirror
    u = rank / (env.n - 1)                            # 0 far ... 1 near
    sign = np.where(np.arange(env.n) % 2 == 0, 1.0, -1.0)
    off = np.zeros((env.n, 2))
    off[:, 1] = s * u * sign                          # vertical only, for now
    return off


if __name__ == "__main__":
    # limit set very high so the episode runs even while it is unsafe;
    # we draw the real limit on the plot instead of stopping
    env = AimingEnv(peak_limit_kw_m2=1e9)

    # ---------------- Part A: which spread is safe? ----------------------
    az, el = sun_at(172, 9.0)
    print(f"{'spread s (m)':>12s} {'power kW':>10s} {'peak kW/m2':>11s} {'intercept':>10s}")
    results = {}
    for s in [0.0, 0.5, 1.0, 1.5, 2.0]:
        set_aims(env.field, spread_pattern(env, s), env.axes)
        res, flux = trace(env.PT, env.field, az, el)
        results[s] = (res['Absorbed power (kW)'], flux.max(), res['Intercept efficiency (%)'])
        print(f"{s:12.1f} {results[s][0]:10.1f} {results[s][1]:11.1f} {results[s][2]:9.1f}%")

    safe = [s for s, (_, pk, _) in results.items() if pk < LIMIT]
    s_target = min(safe) if safe else 2.0
    print(f"\nsmallest safe spread: {s_target} m" if safe else "\nno spread tested is safe; using 2.0 m")

    # ---------------- Part B: drive toward it, one minute at a time --------
    target = spread_pattern(env, s_target)
    env.reset(day=172, hour=9.0, seed=0)
    minutes, power, peak, inter = [], [], [], []
    for k in range(30):
        d = target - env.offsets                      # ask for the full move; env clips to slew
        obs, m, done, why = env.step(d)
        minutes.append(k + 1); power.append(m['power_kw'])
        peak.append(m['peak_kw_m2']); inter.append(m['intercept'])
        print(f"min {k+1:2d}  power {m['power_kw']:8.1f}  peak {m['peak_kw_m2']:7.1f}  intercept {m['intercept']:.3f}")
        if done:
            print("episode ended:", why); break

    first_safe = next((mn for mn, pk in zip(minutes, peak) if pk < LIMIT), None)
    print(f"\npeak first below {LIMIT:.0f} at minute {first_safe}" if first_safe else "\nnever got below the limit")

    # ---------------- Plot ------------------------------------------------
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    a1.plot(minutes, power, marker='o'); a1.set_ylabel('absorbed power (kW)'); a1.grid(alpha=0.3)
    a2.plot(minutes, peak, marker='o', color='C1')
    a2.axhline(LIMIT, color='k', linestyle='--', label=f'limit {LIMIT:.0f}')
    a2.set_ylabel('peak flux (kW/m2)'); a2.set_xlabel('minute'); a2.grid(alpha=0.3); a2.legend()
    fig.suptitle(f'Hand-driven episode: center to spread {s_target} m at 0.2 m/min')
    fig.tight_layout()
    plt.savefig('step5_hand_drive.png', dpi=150)
    print("saved step5_hand_drive.png")