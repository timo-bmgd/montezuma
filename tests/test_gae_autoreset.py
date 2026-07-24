"""Regression test for the gymnasium NEXT_STEP autoreset GAE-masking bug.

Under gymnasium 1.x vector envs (`AutoresetMode.NEXT_STEP`, the repo default), the step
after a terminal step silently discards the chosen action and returns the reset obs with
reward 0. With the agents' bookkeeping (`obs_buf[t]=next_obs`, `done_buf[t]=next_done`),
that fake step is exactly the step where `done_buf[t]==1`. CleanRL's original masking
bootstrapped V(final frame of episode N) onto V(start of episode N+1) at that step.

`agents.base.compute_gae` fixes this by walling off the fake step. These tests encode the
boundary numerically so the bug FAILS before the fix and PASSES after, and pin the
documented NEXT_STEP env behaviour so it can't silently drift.

Run:  .venv/bin/python -m pytest tests/test_gae_autoreset.py -q
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agents.base import compute_gae, masked_mean  # noqa: E402


GAMMA, LAM = 0.99, 0.95


def _old_gae(rewards, values, done_buf, next_value, next_done, gamma, gae_lambda, episodic):
    """The pre-fix inline recursion (no fake-step wall) -- what all three agents used.

    Kept here as the FAIL-before reference so the regression is self-documenting.
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(T)):
        if t == T - 1:
            nextnonterminal = (1.0 - next_done) if episodic else 1.0
            nextvalues = next_value
        else:
            nextnonterminal = (1.0 - done_buf[t + 1]) if episodic else 1.0
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    return advantages


def _boundary_buffers(value_at_reset=4.0):
    """Synthetic (T=6, N=1) rollout with a NEXT_STEP episode boundary.

    Genuine terminal transition at t=2 (terminal reward 5.0); fake discarded-action reset
    step at t=3 (done_buf[3]==1, reward 0, stored value = V(final frame) = a deliberately
    large 100.0). New episode at t>=4; `value_at_reset` is V(reset frame) at t=4, varied
    by tests to detect cross-episode leakage.
    """
    rewards = torch.tensor([[0.1], [0.2], [5.0], [0.0], [0.4], [0.5]], dtype=torch.float64)
    values = torch.tensor([[1.0], [2.0], [3.0], [100.0], [value_at_reset], [5.0]], dtype=torch.float64)
    done_buf = torch.tensor([[0.0], [0.0], [0.0], [1.0], [0.0], [0.0]], dtype=torch.float64)
    next_value = torch.tensor([[6.0]], dtype=torch.float64)
    next_done = torch.tensor([0.0], dtype=torch.float64)
    return rewards, values, done_buf, next_value, next_done


def test_fake_step_advantage_is_walled_to_zero():
    r, v, d, nv, nd = _boundary_buffers()
    adv = compute_gae(r, v, d, nv, nd, GAMMA, LAM, episodic=True)
    assert adv[3].item() == 0.0, "fake (discarded-action) step must carry no advantage"


def test_terminal_step_does_not_bootstrap_across_the_boundary():
    r, v, d, nv, nd = _boundary_buffers()
    adv = compute_gae(r, v, d, nv, nd, GAMMA, LAM, episodic=True)
    # genuine terminal transition (t=2): advantage == reward - value, no bootstrap term,
    # and crucially NOT influenced by the large V(final frame)=100 at t=3.
    expected = r[2].item() - v[2].item()  # 5.0 - 3.0
    assert adv[2].item() == expected


def test_extrinsic_bug_present_in_old_recursion():
    """FAIL-before guard: the pre-fix recursion corrupts the fake step; the fix zeros it."""
    r, v, d, nv, nd = _boundary_buffers()
    old = _old_gae(r, v, d, nv, nd, GAMMA, LAM, episodic=True)
    new = compute_gae(r, v, d, nv, nd, GAMMA, LAM, episodic=True)
    # the old recursion bootstraps V(final frame)->V(reset) at the fake step (large value);
    # the fix walls it to 0.
    assert abs(old[3].item()) > 1.0
    assert new[3].item() == 0.0
    assert old[3].item() != new[3].item()


def test_intrinsic_stream_no_cross_episode_leak():
    """Non-episodic (RND) stream: advantages BEFORE the boundary must not depend on the
    NEW episode's values. This is the leak the extrinsic firewall never guarded against."""
    r_a, v_a, d, nv, nd = _boundary_buffers(value_at_reset=4.0)
    r_b, v_b, _, _, _ = _boundary_buffers(value_at_reset=999.0)  # differs only at reset (t=4)

    new_a = compute_gae(r_a, v_a, d, nv, nd, GAMMA, LAM, episodic=False)
    new_b = compute_gae(r_b, v_b, d, nv, nd, GAMMA, LAM, episodic=False)
    # PASS-after: pre-boundary advantages (t=0,1,2) unchanged by the new episode's value.
    assert torch.allclose(new_a[:3], new_b[:3]), "walled intrinsic stream leaked across episodes"

    old_a = _old_gae(r_a, v_a, d, nv, nd, GAMMA, LAM, episodic=False)
    old_b = _old_gae(r_b, v_b, d, nv, nd, GAMMA, LAM, episodic=False)
    # FAIL-before: the old recursion DID leak the new episode's value backwards.
    assert not torch.allclose(old_a[:3], old_b[:3]), "expected the pre-fix leak to be present"


def test_masked_mean_matches_plain_mean_when_all_kept():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    keep = torch.ones_like(x)
    assert torch.isclose(masked_mean(x, keep), x.mean())


def test_masked_mean_excludes_and_is_nan_safe():
    x = torch.tensor([1.0, 2.0, 100.0, 4.0])
    keep = torch.tensor([1.0, 1.0, 0.0, 1.0])  # drop the outlier at index 2
    assert torch.isclose(masked_mean(x, keep), torch.tensor((1.0 + 2.0 + 4.0) / 3.0))
    # all-zero mask -> 0, not NaN
    assert masked_mean(x, torch.zeros_like(x)).item() == 0.0


def test_env_next_step_fake_step_signature():
    """Pin the documented NEXT_STEP env behaviour: the step AFTER a terminal step has
    reward 0 and terminated 0 (the discarded-action reset step the fix handles)."""
    import gymnasium as gym
    from gymnasium.vector import SyncVectorEnv, AutoresetMode

    envs = SyncVectorEnv([lambda: gym.make("CartPole-v1")], autoreset_mode=AutoresetMode.NEXT_STEP)
    envs.reset(seed=0)
    boundary = None
    for step in range(200):
        _, reward, terminated, truncated, _ = envs.step(np.array([1]))
        if boundary is not None and step == boundary + 1:
            # the fake step: action discarded, reward 0, not terminal
            assert reward[0] == 0.0
            assert not terminated[0] and not truncated[0]
            break
        if boundary is None and (terminated[0] or truncated[0]):
            boundary = step
    envs.close()
    assert boundary is not None, "CartPole should have terminated within 200 steps"


if __name__ == "__main__":
    # Runnable without pytest (this repo pins no test framework):
    #   .venv/bin/python tests/test_gae_autoreset.py
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failures}/{len(fns)} passed")
    sys.exit(1 if failures else 0)
