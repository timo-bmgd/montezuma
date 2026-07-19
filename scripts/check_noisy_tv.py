"""Deterministic validation checks for the NoisyTVWrapper (no training involved).

Usage, from the project root with the venv active:

    python scripts/check_noisy_tv.py             # run all wrapper assertions
    python scripts/check_noisy_tv.py --hash-off  # print SHA256 of a 500-step tv-off trajectory

--hash-off builds the env stack WITHOUT any tv_* arguments, so it runs on both
the pre-TV code and the TV branch. The two hashes must be identical: with
tv_mode="off" the wrapper is never constructed, so the trajectory bytes must
match the pre-TV code exactly. (Full training runs can't verify this — the
obs-norm init loop's envs.action_space.sample() uses the space's own
entropy-seeded RNG and is nondeterministic across runs.)
"""

import argparse
import hashlib
import os
import sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hash-off", action="store_true",
                   help="Print a SHA256 over a fixed-action tv-off trajectory instead of "
                        "running the wrapper assertions")
    p.add_argument("--src", default=os.path.join(os.path.dirname(__file__), "..", "src"),
                   help="src directory to import agents.base from (point at another checkout's "
                        "src to hash its pre-TV baseline)")
    return p.parse_args()


ENV_ID = "ALE/MontezumaRevenge-v5"


def _make(overlay=False, **tv_kwargs):
    from agents.base import make_env
    return make_env(ENV_ID, 0, False, "tvcheck", overlay_video=overlay, **tv_kwargs)()


def _find_tv(env):
    from agents.base import NoisyTVWrapper
    e = env
    while not isinstance(e, NoisyTVWrapper):
        e = e.env
    return e


def _region(obs, tv):
    row, col = tv.position
    return obs[..., row : row + tv.size[0], col : col + tv.size[1]]


def check_static_reset_fill():
    env = _make(tv_mode="static")
    tv = _find_tv(env)
    obs, _ = env.reset(seed=7)
    region = _region(obs, tv)  # (4, size, size)
    assert obs.shape == (4, 84, 84), obs.shape
    assert (region == region[3]).all(), "reset patch must fill all 4 stack slots identically"
    assert region.std() > 30, f"patch should be high-variance noise, got std={region.std():.1f}"
    env.close()
    print("ok: static reset fills all stack slots with the patch")


def check_remote_press_semantics():
    env = _make(tv_mode="remote")
    tv = _find_tv(env)
    assert env.action_space.n == 19
    obs, _ = env.reset(seed=7)
    p0 = _region(obs, tv)[3].copy()
    for a in range(18):
        obs, _, _, _, _ = env.step(a)
        assert (_region(obs, tv)[3] == p0).all(), f"game action {a} must not resample the patch"
    obs, _, _, _, _ = env.step(18)
    assert not (_region(obs, tv)[3] == p0).all(), "pressing the remote must resample the patch"
    env.close()

    env = _make(tv_mode="static")
    assert env.action_space.n == 18
    env.close()
    env = _make(tv_mode="sham-remote")
    assert env.action_space.n == 19
    env.close()
    print("ok: remote press semantics + action-space sizes")


def check_sham_equals_off():
    env_sham = _make(tv_mode="sham-remote")
    env_off = _make()
    obs_s, _ = env_sham.reset(seed=11)
    obs_o, _ = env_off.reset(seed=11)
    assert (obs_s == obs_o).all()
    for i in range(100):
        a = (i * 5) % 18  # never the TV action: off has no action 18
        obs_s, r_s, te_s, tr_s, _ = env_sham.step(a)
        obs_o, r_o, te_o, tr_o, _ = env_off.step(a)
        assert (obs_s == obs_o).all() and r_s == r_o and te_s == te_o and tr_s == tr_o, \
            f"sham-remote diverged from off at step {i}"
        if te_s or tr_s:
            obs_s, _ = env_sham.reset()
            obs_o, _ = env_off.reset()
    env_sham.close()
    env_off.close()
    print("ok: sham-remote observations byte-identical to off")


def check_noise_reproducibility():
    def run(seed):
        env = _make(tv_mode="remote")
        tv = _find_tv(env)
        obs, _ = env.reset(seed=seed)
        stream = [_region(obs, tv)[3].tobytes()]
        for i in range(60):
            a = 18 if i % 5 == 0 else (i * 7) % 18
            obs, _, te, tr, _ = env.step(a)
            stream.append(_region(obs, tv)[3].tobytes())
            if te or tr:
                obs, _ = env.reset()
        env.close()
        return stream

    assert run(3) == run(3), "same seed must reproduce the exact noise stream"
    assert run(3)[0] != run(4)[0], "different seeds must give different patches"
    print("ok: noise stream reproducible per seed")


def check_render_composite():
    import numpy as np
    import cv2
    env = _make(overlay=True, tv_mode="static")
    tv = _find_tv(env)
    env.reset(seed=3)
    frame = env.render()
    row, col = tv.position
    r0, r1 = round(row * 210 / 84), round((row + tv.size[0]) * 210 / 84)
    c0, c1 = round(col * 160 / 84), round((col + tv.size[1]) * 160 / 84)
    expected = cv2.resize(tv._patch, (c1 - c0, r1 - r0), interpolation=cv2.INTER_NEAREST)
    assert (frame[r0:r1, c0:c1] == expected[..., None]).all(), \
        "rendered frame must show the upscaled current patch"
    frame2 = env.render()
    assert (frame == frame2).all(), "render() must not resample the patch"
    env.close()
    print("ok: render() composites the patch, without side effects")


def hash_off_trajectory():
    import numpy as np
    env = _make()  # no tv_* kwargs: builds on pre-TV code and on the branch alike
    obs, _ = env.reset(seed=123)
    h = hashlib.sha256(obs.tobytes())
    for i in range(500):
        obs, reward, terminated, truncated, _ = env.step((i * 7) % 18)
        h.update(obs.tobytes())
        h.update(np.float64(reward).tobytes())
        h.update(bytes([terminated, truncated]))
        if terminated or truncated:
            obs, _ = env.reset()
            h.update(obs.tobytes())
    env.close()
    print(f"tv-off trajectory sha256: {h.hexdigest()}")


def main():
    args = parse_args()
    sys.path.insert(0, os.path.abspath(args.src))
    if args.hash_off:
        hash_off_trajectory()
        return
    check_static_reset_fill()
    check_remote_press_semantics()
    check_sham_equals_off()
    check_noise_reproducibility()
    check_render_composite()
    print("all noisy-TV wrapper checks passed")


if __name__ == "__main__":
    main()
