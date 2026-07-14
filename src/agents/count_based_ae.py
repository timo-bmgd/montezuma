"""AE-SimHash count-based exploration + PPO for Montezuma's Revenge.

Sibling to count_based.py's fixed-random-projection SimHash, which is documented there
to fail on Montezuma's Revenge: a random Gaussian projection of raw downsampled pixels
gives nearly every state its own hash bucket (n≈1 everywhere), so the count-based bonus
degenerates to a flat constant with no exploration signal.

This variant replaces the fixed random projection with a trained autoencoder (Tang et al.,
2017 — "#Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning",
https://arxiv.org/abs/1611.04717, the AE-SimHash / "SmartHash" variant -- note this is NOT
arXiv:1703.01310, a different unrelated paper this file and CLAUDE.md previously mis-cited).
The encoder's sigmoid bottleneck, thresholded at 0.5, produces a D-bit code that is *further*
down-projected to a k-bit code via a second, fixed random Gaussian SimHash matrix (Algorithm 2:
phi(s) = sgn(A . g(s))) -- that k-bit code is what's actually counted. A saturation loss plus
noise injected before decoding pushes the D-bit code toward confident, perturbation-robust
bits, so perceptually similar frames are more likely to collide into the same bucket than
under raw-pixel hashing. See AEHashModel's docstring for the exact mechanism and the collapse
failure mode found during development, why it happened, and how it was actually fixed.

The AE is trained periodically (every --ae-update-every iterations, paper's j_update=3) on
batches sampled from a FIFO replay pool of visited states (AEReplayPool) -- not on the current
iteration's on-policy PPO batch -- via its own decoupled Adam optimizer, separate from the
policy's. This matches Algorithm 2 exactly and, per the paper's own stated rationale, is one of
two deliberate stability mechanisms (the other being the D->k down-projection above): "it is
important that the mapping from state to code needs to remain relatively consistent over
time... A solution is to downsample the binary code to a very low dimension, or by slowing down
the training process." Earlier versions of this file trained the AE every iteration on the
current on-policy batch, jointly with the policy via one shared optimizer (a pattern borrowed
from rnd.py) -- with neither of the paper's two stability mechanisms in place, and the AE code
collapsed to a constant within 2-4 iterations across three separate fix attempts. See
AEHashModel's docstring for the full debugging history and the 2026-07-14 resolution.

Resolved 2026-07-14, in two parts: implementing both of the paper's stability mechanisms
(D=256 -> k=64 down-projection, replay-pool-sourced updates every 3 iterations) turned out to
only *slow* the collapse by roughly an order of magnitude (from 2-4 iterations/256-512 steps to
several thousand steps), not prevent it -- verified both in the real training loop and in a
from-scratch synthetic isolation test with no RL involved at all (a batch of frames with a
genuinely random sprite position each sample, decoupled from any policy/replay/staleness
concern). That same isolation test traced the actual cause to sigmoid saturation itself: as the
bottleneck's pre-sigmoid logits drift away from 0 (for any reason -- initialization noise,
early gradient direction), sigmoid's own derivative vanishes, cutting off the gradient the
encoder needs to ever learn input-dependent codes, in a self-reinforcing spiral toward a single
constant code. This survived every mechanism the paper specifies (noise, saturation loss,
down-projection, slow training) because none of them add gradient back through an already-
saturated sigmoid -- confirmed by testing paper-faithful settings, zero saturation loss, small
bottleneck-layer initialization, and a batch-variance-reweighted reconstruction loss, none of
which prevented it (see AEHashModel's debugging history for the exact experiments and results).
What did fix it, verified in the same synthetic isolation test: `nn.BatchNorm1d` on the
bottleneck's pre-sigmoid logits. This is NOT a mechanism the paper specifies -- it's a standard,
widely-used technique for exactly this failure mode in other stochastic/saturating-bottleneck
architectures (e.g. combating posterior collapse in VAEs), added here because the paper's own
stated mechanisms, faithfully implemented and empirically verified, were insufficient on this
game's early, extremely low-diversity (static background, tiny moving sprite) visual
distribution. See AEHashModel's docstring for the full mechanism and verification numbers.

This pass also closed two previously deliberate, documented fidelity gaps: input is now
downsampled to 52x52 (was 42x42, matching the paper), and the reconstruction loss is now a
pixel-wise 64-bin categorical cross-entropy with label smoothing (was plain MSE, matching the
paper's stated softmax reconstruction). The saturation-loss formula was also corrected to the
paper's (lambda/D)*sum_i (mean-over-bits, not plain sum) with the paper's own lambda=10 as the
new default. A handful of details the paper does not specify numerically (replay pool capacity,
AE batch size, AE learning rate) and one detail derived rather than transcribed (the
encoder/decoder's exact intermediate spatial shape, since the paper's own stated shape reflects
an unspecified original padding convention) remain documented, intentional choices -- see
AEHashModel's and AEReplayPool's docstrings.

Run from project root with the venv active:
    source .venv/bin/activate
    python src/agents/count_based_ae.py
    python src/agents/count_based_ae.py --exploration-coef 0.1 --hash-dim 128
"""

import os
import sys

import argparse
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agents.base import NatureCNN, layer_init, make_env


def parse_args():
    p = argparse.ArgumentParser()
    # experiment
    p.add_argument("--exp-name", default=os.path.basename(__file__)[:-3])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--track", action="store_true")
    p.add_argument("--wandb-project", default="montezuma-thesis")
    p.add_argument("--capture-video", action="store_true")
    p.add_argument("--record-room-discovery", action="store_true",
                   help="In addition to periodic recording, also record video whenever "
                        "agent sets a new room high-water mark")
    p.add_argument("--video-episode-interval", type=int, default=100,
                   help="Record a video every N episodes (env 0 only, when --capture-video is set)")
    p.add_argument("--clip-reward", action=argparse.BooleanOptionalAction, default=True,
                   help="Clip extrinsic reward to [-1, 1] (standard Atari preprocessing)")
    # env
    p.add_argument("--env-id", default="ALE/MontezumaRevenge-v5")
    p.add_argument("--total-timesteps", type=int, default=10_000_000)
    # ppo (same defaults as count_based.py)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--clip-coef", type=float, default=0.1)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    # count-based / AE-SimHash specific
    p.add_argument("--exploration-coef", type=float, default=0.01,
                   help="Intrinsic reward scale: beta / sqrt(n(hash(s)))")
    p.add_argument("--hash-dim", type=int, default=64,
                   help="k: number of bits in the final, SimHash-projected counting code "
                        "(after down-projection from the AE's own --ae-code-dim bottleneck)")
    p.add_argument("--ae-code-dim", type=int, default=256,
                   help="D: the AE's own sigmoid bottleneck size, before SimHash "
                        "down-projection to --hash-dim bits (paper value: 256)")
    p.add_argument("--ae-recon-coef", type=float, default=1.0,
                   help="Weight on the AE reconstruction (pixel-wise categorical "
                        "cross-entropy) loss")
    p.add_argument("--ae-sat-coef", type=float, default=10.0,
                   help="Weight (paper's lambda=10) on the saturation loss pushing latent "
                        "bits toward 0/1, applied to the paper's mean-over-bits formula. The "
                        "paper's own balance assumed a summed-per-image log-likelihood "
                        "reconstruction loss; this implementation uses mean-per-pixel "
                        "cross-entropy instead, so still treat this as tunable, not exact")
    p.add_argument("--ae-noise-amplitude", type=float, default=0.3,
                   help="Amplitude a of uniform noise U(-a,a) added to the post-sigmoid "
                        "code before decoding (train-time only). Tang et al. require "
                        "a > 0.25 for this to actually force distinct states to distinct "
                        "codes -- see AEHashModel's docstring")
    p.add_argument("--ae-num-bins", type=int, default=64,
                   help="Number of pixel-intensity quantization bins for the categorical "
                        "reconstruction loss (paper value)")
    p.add_argument("--ae-label-smoothing", type=float, default=0.003,
                   help="Label smoothing epsilon for the reconstruction cross-entropy "
                        "(paper value; see AEHashModel's docstring for the paper's more "
                        "ambiguously-worded original renormalization scheme)")
    p.add_argument("--ae-update-every", type=int, default=3,
                   help="Train the AE only every N PPO iterations (paper's j_update), "
                        "sampling from the replay pool rather than the current batch")
    p.add_argument("--ae-batch-size", type=int, default=256,
                   help="Batch size sampled from the replay pool per AE training step "
                        "(not paper-specified)")
    p.add_argument("--ae-lr", type=float, default=1e-3,
                   help="Learning rate for the AE's own Adam optimizer, decoupled from the "
                        "policy's (paper uses Adam for both but doesn't give an explicit AE LR)")
    p.add_argument("--ae-replay-capacity", type=int, default=100_000,
                   help="FIFO replay pool capacity in frames (not paper-specified)")
    p.add_argument("--image-log-interval", type=int, default=100,
                   help="Log original-vs-reconstruction image pairs to TensorBoard every N iterations")
    # infrastructure
    p.add_argument("--sync-envs", action="store_true",
                   help="Use SyncVectorEnv instead of AsyncVectorEnv (easier debugging)")
    p.add_argument("--runs-dir", default="runs", help="Directory for TensorBoard logs")
    p.add_argument("--videos-dir", default="videos", help="Directory for recorded videos")
    p.add_argument("--checkpoint-dir", default="checkpoints", help="Directory to save checkpoints")
    p.add_argument("--checkpoint-interval", type=int, default=100,
                   help="Save a checkpoint every N iterations")
    p.add_argument("--resume", default=None, help="Path to checkpoint .pt file to resume from")
    return p.parse_args()


class AEHashModel(nn.Module):
    """Autoencoder producing a saturating, thresholdable binary code for two-stage SimHash
    counting (Tang et al. 2017, "#Exploration: A Study of Count-Based Exploration for Deep
    Reinforcement Learning", https://arxiv.org/abs/1611.04717 -- the AE-SimHash /
    "SmartHash" variant; note this is NOT arXiv:1703.01310, a different, unrelated paper
    on PixelCNN density models that an earlier version of this docstring mis-cited).

    Encoder: 3x Conv2d(96 channels, kernel=6, stride=2), operating on a SINGLE normalised
    grayscale frame (N, 1, 84, 84) in [0, 1] -- the last frame of the 4-stack, matching RND's
    convention (rnd.py uses next_obs_np[:, 3:4, :, :]) -- downsampled to 52x52 before the conv
    trunk (see `downsample`; the paper's own input size). Channel count/kernel/stride match the
    paper's stated Atari architecture exactly. Flatten -> Linear(96*3*3, 1024) -> ReLU ->
    Linear(1024, code_dim) -> BatchNorm1d(code_dim) -> sigmoid = the D-bit bottleneck code b(s)
    ("code_dim", default 256, the paper's D). The BatchNorm1d is NOT in the paper -- see
    "Debugging history" entry 5 below for why it's there: without it, this bottleneck collapses
    to a single constant code regardless of input, and none of the paper's own stated mechanisms
    (noise, saturation loss, down-projection, slow training) prevent that on this game. Because
    train() relies on a deterministic, sample-independent code for hashing (not one that depends
    on whichever other frames happen to share a batch), `ae_model` is kept in `.eval()` mode
    (BatchNorm normalises with its running statistics) everywhere except during its own gradient
    update in train() (`.train()`, batch statistics) -- see train()'s AE update block.

    Noise + saturation (Tang et al. Section 2.3 / Eq. 3): at train time uniform noise
    U(-a, a) is added to b_clean (NOT to the pre-sigmoid logits) before decoding:
    b_for_decode = b_clean + noise. This is the paper's actual anti-thrash mechanism for the
    D-bit code itself, and its placement matters -- the paper states that with a > 1/4, the
    decoder can only reconstruct two distinct inputs correctly if their clean codes are spread
    far enough apart to survive the noise, which directly incentivizes distinct inputs to get
    distinct codes. Adding noise to the *pre-sigmoid* logits instead (this file's first
    version) doesn't have this property: a network can trivially defeat Gaussian logit noise by
    learning large-magnitude, input-independent logits, which saturate through the sigmoid
    regardless of the noise. The saturation loss (computed by the caller as
    `mean_i(min((1-b_i)^2, b_i^2))`, i.e. the paper's `(1/D)*sum_i` term) is evaluated on
    b_clean, matching the paper's stated motivation of preventing unused bits from fluctuating
    near 0.5, not on the noised value.

    Two-stage hash (Algorithm 2, the piece earlier versions of this file were missing
    entirely): the D-bit code above is never counted directly. `binary_code` rounds it to
    {0,1}^D, and `project_to_k` down-projects it through a second, FIXED random Gaussian
    matrix A (shape k_bits x code_dim, seeded from --seed) via phi(s) = sgn(A . g(s)) --
    exactly the same fixed-random-projection SimHash idea count_based.py applies to raw
    pixels, just applied here to the AE's learned code instead. `hash_code` composes both
    steps and is what train() actually feeds to AEHashCounter. The paper states this
    down-projection is deliberately for stability, not just dimensionality reduction: "it is
    important that the mapping from state to code needs to remain relatively consistent over
    time... A solution is to downsample the binary code to a very low dimension, or by slowing
    down the training process" -- the *other* half of that sentence is why train() also trains
    this model only periodically from a replay pool (see AEReplayPool and train()), rather than
    every iteration on the current on-policy batch.

    Decoder mirrors the encoder exactly: Linear(code_dim,1024) -> ReLU -> Linear(1024,96*3*3)
    -> ReLU -> reshape (96,3,3) -> 3x ConvTranspose2d(96 channels, kernel=6, stride=2), the
    last one projecting to num_bins channels (raw logits, no activation -- softmax happens
    inside the reconstruction loss). With kernel=6/stride=2/no padding on either side, the
    52->24->10->3 encoder trace inverts EXACTLY via matching zero-output-padding transposed
    convolutions: (3-1)*2+6=10, (10-1)*2+6=24, (24-1)*2+6=52 -- no rounding slop. This
    intentionally does not reproduce the paper's own stated intermediate shape (FC 2400,
    reshape 96x5x5): that number appears to reflect an unspecified padding convention in the
    paper's original Theano/Lasagne code that can't be reverse-engineered from the text alone,
    so this implementation instead derives its own internally-consistent shape from the paper's
    actual specified hyperparameters (96 channels, 6x6 kernel, stride 2, 3 layers, 52x52 input) --
    matching the paper's stability-relevant choices exactly while not claiming bit-for-bit
    architectural parity on a detail the paper doesn't fully specify.

    Reconstruction target: per Tang et al., "since the pixel intensities are discrete values in
    the range [0, 255], we make use of a pixel-wise softmax output layer that shares weights
    between all pixels," with "label smoothing... in which the log-probability of each of the
    bins is increased by 0.003, before normalizing." `forward` returns per-pixel target bin
    indices in [0, num_bins) (`floor(pixel_intensity_in_[0,1] * num_bins)`), and train() computes
    `F.cross_entropy(logits, target_bins, label_smoothing=...)` -- PyTorch's standard label
    smoothing (redistributing epsilon probability mass toward non-target bins) rather than a
    literal implementation of the paper's own renormalization wording, which is ambiguous as
    written (adding a constant to every bin's log-probability and then renormalizing via
    log-sum-exp is a no-op, so the paper likely means something closer to additive smoothing on
    probabilities, or a still different scheme; at epsilon=0.003 the practical difference between
    interpretations is negligible, and PyTorch's mechanism is standard and well-tested).

    Debugging history (each entry is a re-run of this module's smoke test, see train()'s
    verification instructions):
    1. Raw 84x84 target, Gaussian pre-sigmoid noise: bit variance -> 0 within 2-4
       iterations, persisted with --ae-sat-coef 0 --ae-noise-std 0 (ruled out those knobs).
    2. 42x42 downsampled target, same noise scheme: collapse persisted at the same
       magnitude/timing (ruled out "sprite too small a pixel fraction").
    3. Post-sigmoid uniform noise, matching the paper's actual mechanism: collapse persisted
       at the same magnitude/timing as (1) and (2) (unique_states=98, bit variance -> 0 by
       iteration 3-4). This paper-faithful noise fix alone did not resolve it. Diagnosis at the
       time: the noise term only penalizes codes that are close-but-not-identical for different
       inputs, creating pressure to push them further apart; it supplies no force pulling the
       network back out of a state where the code is already bit-for-bit IDENTICAL for every
       input -- once collapsed that far, the decoder just needs to invert one fixed noisy value,
       trivially achievable regardless of noise amplitude. All three attempts collapsed within
       2-4 iterations (256-512 steps), while the policy was still close to random and likely
       hadn't left the first screen -- a short, visually homogeneous window for the paper's
       per-code anti-thrash mechanism to get a foothold in before full collapse.
    4. Attempts 1-3 all trained the D-bit code every iteration on the current small,
       homogeneous on-policy batch, with no down-projection -- i.e. neither of the paper's two
       stated stability mechanisms was present, and the paper's own text predicts exactly this
       failure mode without them. Adding both (D=256->k=64 down-projection via a fixed random
       matrix, plus training only every --ae-update-every=3 iterations from an AEReplayPool of
       visited states rather than the current batch) only *slowed* the collapse, by roughly an
       order of magnitude (from 2-4 iterations/256-512 steps to ~8,000-14,000 steps in both a
       real smoke-test run and, with --ae-batch-size 32 to force early AE updates, an even
       shorter one) -- confirmed by watching charts/ae_code_bit_variance decay smoothly and
       exponentially to exactly 0.0 in both runs, at which point charts/unique_states froze
       permanently and losses/ae_recon_loss kept improving (the network was learning to
       reconstruct a fixed "average frame" well, not failing to learn).
    5. To isolate whether this was RL-specific (small/stale/repetitive early-rollout batches)
       or intrinsic to the AE itself, a standalone synthetic test (no RL, no replay pool, no
       PPO) trained this exact model on batches of frames with a genuinely random small bright
       "sprite" square over a fixed background, i.e. deliberately non-degenerate per-sample
       diversity. It STILL collapsed to bit_variance=0 within 5-10 gradient steps with paper-
       faithful settings, and within 20-40 steps with saturation loss coefficient set to 0 --
       ruling out both the RL loop and the saturation loss as the primary cause. Also tested and
       ruled out: small bottleneck-layer initialization (collapsed just as fast, once even
       faster), and reweighting the reconstruction loss by per-pixel batch variance to
       counteract the sprite being a small fraction of pixels (no meaningful change). Diagnosis:
       ordinary sigmoid saturation. As the bottleneck's pre-sigmoid logits drift away from 0 for
       any reason, sigmoid's derivative vanishes, cutting off the gradient the encoder needs to
       differentiate inputs at all -- a self-reinforcing spiral with no restoring force, which
       is why every paper-specified mechanism above (none of which re-inject gradient through an
       already-saturated sigmoid) only delayed rather than prevented it.
    6. Resolved 2026-07-14: adding `nn.BatchNorm1d(code_dim)` on the pre-sigmoid logits (not a
       mechanism the paper specifies, but a standard technique for exactly this kind of
       stochastic/saturating-bottleneck collapse elsewhere in the literature) fixed it in the
       same synthetic isolation test: bit variance stayed healthy and gradually increasing
       (0.040 -> 0.069 over 150 steps) instead of collapsing, while reconstruction loss improved
       right alongside it. Confirmed in the real training loop too (80,000 steps, 4 envs,
       otherwise-default settings): charts/ae_code_bit_variance rises from ~5e-5 and stabilises
       in a healthy 0.04-0.06 band from roughly step 35,000 onward (no further collapse for the
       remaining 45,000 steps), losses/ae_recon_loss falls smoothly to ~0.16 and
       losses/ae_sat_loss to ~0.07 (not 0.0 -- bits are becoming more confident, not degenerate),
       and charts/unique_states grows continuously to 34,012 by the end (vs. permanently frozen
       at 889-1,342 pre-fix) without exploding to "every state is its own bucket" (the
       original, different failure mode that motivated this file's existence -- see the module
       docstring). Checkpoint save/resume (including both optimizers and BatchNorm's running
       statistics) verified to round-trip correctly across this change.
    """

    _DOWNSAMPLE_SIZE = 52  # paper's input size

    def __init__(self, code_dim: int = 256, k_bits: int = 64, noise_amplitude: float = 0.3,
                 num_bins: int = 64, seed: int = 1):
        super().__init__()
        self.code_dim = code_dim
        self.k_bits = k_bits
        self.noise_amplitude = noise_amplitude
        self.num_bins = num_bins

        self.encoder_conv = nn.Sequential(
            layer_init(nn.Conv2d(1, 96, 6, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(96, 96, 6, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(96, 96, 6, stride=2)),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.encoder_fc1 = layer_init(nn.Linear(96 * 3 * 3, 1024))
        self.encoder_fc2 = layer_init(nn.Linear(1024, code_dim))
        self.bottleneck_bn = nn.BatchNorm1d(code_dim)

        self.decoder_fc1 = layer_init(nn.Linear(code_dim, 1024))
        self.decoder_fc2 = layer_init(nn.Linear(1024, 96 * 3 * 3))
        self.decoder_conv = nn.Sequential(
            layer_init(nn.ConvTranspose2d(96, 96, 6, stride=2)),
            nn.ReLU(),
            layer_init(nn.ConvTranspose2d(96, 96, 6, stride=2)),
            nn.ReLU(),
            layer_init(nn.ConvTranspose2d(96, num_bins, 6, stride=2)),
        )

        # fixed (never trained) k x D random Gaussian projection for the second SimHash
        # stage: phi(s) = sgn(A . g(s)), Tang et al. Algorithm 2.
        rng = np.random.default_rng(seed)
        proj = rng.standard_normal((k_bits, code_dim)).astype(np.float32)
        self.register_buffer("proj_matrix", torch.from_numpy(proj))

    def downsample(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 1, 84, 84) float in [0, 1] -> (N, 1, 52, 52) via area interpolation."""
        return F.interpolate(x, size=(self._DOWNSAMPLE_SIZE, self._DOWNSAMPLE_SIZE), mode="area")

    def _encode_from_downsampled(self, x_ds: torch.Tensor, add_noise: bool):
        """Returns (b_clean, b_for_decode). b_for_decode == b_clean unless add_noise."""
        feat = self.encoder_conv(x_ds)
        feat = F.relu(self.encoder_fc1(feat))
        logits = self.bottleneck_bn(self.encoder_fc2(feat))
        b_clean = torch.sigmoid(logits)
        if add_noise and self.noise_amplitude > 0:
            b_for_decode = b_clean + (torch.rand_like(b_clean) * 2 - 1) * self.noise_amplitude
        else:
            b_for_decode = b_clean
        return b_clean, b_for_decode

    def encode(self, x: torch.Tensor, add_noise: bool):
        """x: (N, 1, 84, 84) float in [0, 1] (raw, not yet downsampled).
        Returns (b_clean, b_for_decode), both (N, code_dim)."""
        return self._encode_from_downsampled(self.downsample(x), add_noise)

    def decode(self, b: torch.Tensor) -> torch.Tensor:
        """Returns (N, num_bins, 52, 52) -- raw per-pixel, per-bin logits."""
        feat = F.relu(self.decoder_fc1(b))
        feat = F.relu(self.decoder_fc2(feat))
        feat = feat.view(-1, 96, 3, 3)
        return self.decoder_conv(feat)

    def forward(self, x: torch.Tensor):
        """Training forward pass: downsample -> noisy D-bit code -> reconstruction logits.
        Returns (logits, b_clean, target_bins): target_bins (N, 52, 52) int64 in [0, num_bins)
        is the discretized-intensity classification target for the reconstruction
        cross-entropy; b_clean (not the noised code) is what the caller should use for the
        saturation loss."""
        x_ds = self.downsample(x)
        b_clean, b_for_decode = self._encode_from_downsampled(x_ds, add_noise=True)
        logits = self.decode(b_for_decode)
        target_bins = torch.clamp((x_ds.squeeze(1) * self.num_bins).floor(), 0, self.num_bins - 1).long()
        return logits, b_clean, target_bins

    @torch.no_grad()
    def reconstruction_preview(self, logits: torch.Tensor) -> torch.Tensor:
        """(N, num_bins, H, W) logits -> (N, 1, H, W) in [0, 1] for TensorBoard image
        logging: argmax bin per pixel, mapped to that bin's midpoint intensity."""
        bins = logits.argmax(dim=1, keepdim=True).float()
        return (bins + 0.5) / self.num_bins

    @torch.no_grad()
    def binary_code(self, x: torch.Tensor) -> torch.Tensor:
        """Clean (noise-free), rounded D-bit AE code: (N, code_dim), values in {0., 1.}."""
        b_clean, _ = self.encode(x, add_noise=False)
        return (b_clean > 0.5).float()

    @torch.no_grad()
    def project_to_k(self, d_code: torch.Tensor) -> torch.Tensor:
        """Down-projects a rounded D-bit code to a k-bit SimHash code via the fixed random
        matrix: phi(s) = sgn(A . g(s)). Returns (N, k_bits), values in {0., 1.}."""
        projected = d_code @ self.proj_matrix.T
        return (projected > 0).float()

    @torch.no_grad()
    def hash_code(self, x: torch.Tensor) -> torch.Tensor:
        """Full counting pipeline: raw last-frame -> rounded D-bit AE code -> k-bit
        down-projected SimHash code -- the value AEHashCounter actually counts."""
        return self.project_to_k(self.binary_code(x))


class AEReplayPool:
    """FIFO replay pool of visited last-frames for training the AE, per Tang et al.
    Algorithm 2 ("Add the state samples to a FIFO replay pool R" / "Update the AE loss ...
    using samples drawn from the replay pool"). Stores raw uint8 84x84 single-channel frames
    (matching the "counting only looks at the latest frame" convention used throughout this
    file); downsampling to the AE's actual input resolution happens lazily at sample time via
    AEHashModel.downsample, not at insertion time.

    Capacity (--ae-replay-capacity, default 100,000 frames) is not a value given by the paper,
    which only specifies the update cadence (j_update=3 iterations) and not the pool's size or
    the AE's own per-update batch size or learning rate -- all three are this file's own,
    documented-as-tunable choices (see --ae-replay-capacity/--ae-batch-size/--ae-lr help text).

    Not persisted across --resume checkpoints: resuming restarts with an empty pool that
    refills over subsequent iterations. This avoids ballooning checkpoint files with tens of
    thousands of frames per save; the cost is a brief AE "cold start" immediately after
    resuming, an accepted tradeoff rather than a bug.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buf = np.zeros((capacity, 84, 84), dtype=np.uint8)
        self._size = 0
        self._next = 0

    def add(self, frames: np.ndarray) -> None:
        """frames: (N, 84, 84) uint8."""
        n = frames.shape[0]
        if n >= self.capacity:
            self._buf[:] = frames[-self.capacity:]
            self._next = 0
            self._size = self.capacity
            return
        end = self._next + n
        if end <= self.capacity:
            self._buf[self._next:end] = frames
        else:
            first = self.capacity - self._next
            self._buf[self._next:] = frames[:first]
            self._buf[:end - self.capacity] = frames[first:]
        self._next = end % self.capacity
        self._size = min(self._size + n, self.capacity)

    def sample(self, batch_size: int, device) -> torch.Tensor:
        """Returns (batch_size, 1, 84, 84) float32 in [0, 1]."""
        idx = np.random.randint(0, self._size, size=batch_size)
        batch = self._buf[idx].astype(np.float32) / 255.0
        return torch.from_numpy(batch).unsqueeze(1).to(device)

    def __len__(self) -> int:
        return self._size


class AEHashCounter:
    """State visit counter keyed on the k-bit SimHash-projected AE code (Tang et al. 2017
    Algorithm 2: phi(s) = sgn(A . g(s)), where g(s) is AEHashModel's rounded D-bit code and A
    is AEHashModel's fixed random projection matrix -- see AEHashModel.hash_code). Reuses the
    same dict[bytes, int] counting + beta/sqrt(n) bonus formula as SimHashCounter in
    count_based.py; this class does not compute the hash itself, the caller supplies an
    already-projected k-bit binary code.

    The hash is non-stationary (the AE is retrained periodically -- every --ae-update-every
    iterations, from AEReplayPool samples, see train()), so the same physical state can map to
    a different code -- and thus a different counter -- as training progresses. This code
    changes only once every --ae-update-every iterations rather than continuously, which is
    itself one of the paper's two stated anti-collapse/anti-thrash mechanisms (the other being
    the D->k down-projection); see AEHashModel's docstring.
    """

    def __init__(self):
        self._counts: dict[bytes, int] = {}

    @staticmethod
    def _key(code: np.ndarray) -> bytes:
        return np.packbits(code.astype(np.uint8)).tobytes()

    def increment(self, code: np.ndarray) -> int:
        key = self._key(code)
        self._counts[key] = self._counts.get(key, 0) + 1
        return self._counts[key]

    def bonus(self, code: np.ndarray, beta: float) -> float:
        key = self._key(code)
        return beta / np.sqrt(max(self._counts.get(key, 1), 1))

    @property
    def num_unique(self) -> int:
        return len(self._counts)


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.cnn = NatureCNN()
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.cnn(x))

    def get_action_and_value(self, x, action=None):
        features = self.cnn(x)
        logits = self.actor(features)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(features)


def _save_checkpoint(path, iteration, global_step, agent, ae_model, agent_optimizer, ae_optimizer, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "iteration": iteration,
        "global_step": global_step,
        "agent_state_dict": agent.state_dict(),
        "ae_model_state_dict": ae_model.state_dict(),
        "agent_optimizer_state_dict": agent_optimizer.state_dict(),
        "ae_optimizer_state_dict": ae_optimizer.state_dict(),
        "args": vars(args),
    }, path)


def _load_checkpoint(path, agent, ae_model, agent_optimizer, ae_optimizer):
    ckpt = torch.load(path, weights_only=False)
    agent.load_state_dict(ckpt["agent_state_dict"])
    ae_model.load_state_dict(ckpt["ae_model_state_dict"])
    agent_optimizer.load_state_dict(ckpt["agent_optimizer_state_dict"])
    ae_optimizer.load_state_dict(ckpt["ae_optimizer_state_dict"])
    return ckpt["iteration"], ckpt["global_step"]


def train():
    args = parse_args()

    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_iterations = args.total_timesteps // batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(project=args.wandb_project, sync_tensorboard=True,
                   config=vars(args), name=run_name, save_code=True)

    writer = SummaryWriter(f"{args.runs_dir}/{run_name}")
    writer.add_text("hyperparameters",
                    "|param|value|\n|-|-|\n" +
                    "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    VecCls = gym.vector.SyncVectorEnv if args.sync_envs else gym.vector.AsyncVectorEnv
    envs = VecCls(
        [make_env(args.env_id, i, args.capture_video, run_name, args.videos_dir, args.video_episode_interval,
                  args.record_room_discovery, clip_reward=args.clip_reward) for i in range(args.num_envs)]
    )

    agent = Agent(envs).to(device)
    ae_model = AEHashModel(code_dim=args.ae_code_dim, k_bits=args.hash_dim,
                           noise_amplitude=args.ae_noise_amplitude, num_bins=args.ae_num_bins,
                           seed=args.seed).to(device)
    agent_optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    ae_optimizer = optim.Adam(ae_model.parameters(), lr=args.ae_lr, eps=1e-5)
    # ae_model stays in eval() mode except during its own periodic gradient update below, so
    # BatchNorm1d in the bottleneck (see AEHashModel) always normalises with its running
    # statistics rather than the current call's batch statistics -- hashing needs a
    # deterministic, sample-independent code, not one that depends on whatever other frames
    # happen to share a batch at rollout time (only num_envs=8-ish samples) or at health-check
    # time.
    ae_model.eval()
    counter = AEHashCounter()
    replay_pool = AEReplayPool(capacity=args.ae_replay_capacity)

    obs_shape = envs.single_observation_space.shape  # (4, 84, 84)
    obs_buf   = torch.zeros((args.num_steps, args.num_envs) + obs_shape, device=device)
    act_buf   = torch.zeros((args.num_steps, args.num_envs), device=device)
    logp_buf  = torch.zeros((args.num_steps, args.num_envs), device=device)
    rew_buf   = torch.zeros((args.num_steps, args.num_envs), device=device)
    done_buf  = torch.zeros((args.num_steps, args.num_envs), device=device)
    val_buf   = torch.zeros((args.num_steps, args.num_envs), device=device)

    start_iteration = 1
    global_step = 0

    if args.resume:
        start_iteration, global_step = _load_checkpoint(args.resume, agent, ae_model, agent_optimizer, ae_optimizer)
        ae_model.eval()  # load_state_dict doesn't restore the training/eval flag itself
        start_iteration += 1
        print(f"Resumed from {args.resume} at iteration {start_iteration - 1}, global_step={global_step}")

    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs  = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    for iteration in range(start_iteration, num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            agent_optimizer.param_groups[0]["lr"] = frac * args.lr

        # ── rollout collection ──────────────────────────────────────────────
        intrinsic_log = []
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_buf[step]  = next_obs
            done_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                val_buf[step] = value.flatten()
            act_buf[step]  = action
            logp_buf[step] = logprob

            next_obs_np, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminated, truncated)

            # AE-SimHash intrinsic reward: encode last frame (clean, no_grad) -> D-bit code
            # -> k-bit down-projected SimHash code -> visit count -> beta/sqrt(n) bonus.
            # AE weights here are from the last periodic update (see AEHashCounter docstring).
            last_frame_u8 = next_obs_np[:, 3, :, :]  # (N, 84, 84) uint8, for the replay pool
            last_frame_np = last_frame_u8[:, None, :, :].astype(np.float32) / 255.0  # (N, 1, 84, 84)
            codes = ae_model.hash_code(torch.from_numpy(last_frame_np).to(device))
            codes_np = codes.cpu().numpy()
            replay_pool.add(last_frame_u8)

            intrinsic = np.zeros(args.num_envs, dtype=np.float32)
            for i in range(args.num_envs):
                counter.increment(codes_np[i])
                intrinsic[i] = counter.bonus(codes_np[i], args.exploration_coef)
            intrinsic_log.append(intrinsic.mean())

            combined_reward = reward + intrinsic
            rew_buf[step] = torch.tensor(combined_reward, dtype=torch.float32, device=device)
            next_obs  = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.tensor(next_done_np, dtype=torch.float32, device=device)

            if "_episode" in infos:
                for i, ended in enumerate(infos["_episode"]):
                    if not ended:
                        continue
                    r = float(infos["episode"]["r"][i])
                    l = int(infos["episode"]["l"][i])
                    print(f"  step={global_step}  return={r:.1f}  length={l}")
                    writer.add_scalar("charts/episodic_return", r, global_step)
                    writer.add_scalar("charts/episodic_length", l, global_step)
                    if "rooms_visited" in infos:
                        writer.add_scalar("charts/rooms_visited", int(infos["rooms_visited"][i]), global_step)

        # ── GAE ─────────────────────────────────────────────────────────────
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rew_buf, device=device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - done_buf[t + 1]
                    nextvalues = val_buf[t + 1]
                delta = rew_buf[t] + args.gamma * nextvalues * nextnonterminal - val_buf[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + val_buf

        # ── PPO update ──────────────────────────────────────────────────────
        b_obs  = obs_buf.reshape((-1,) + obs_shape)
        b_logp = logp_buf.reshape(-1)
        b_act  = act_buf.reshape(-1)
        b_adv  = advantages.reshape(-1)
        b_ret  = returns.reshape(-1)
        b_val  = val_buf.reshape(-1)

        clipfracs = []
        for _ in range(args.update_epochs):
            mb_inds = np.random.permutation(batch_size)
            for start in range(0, batch_size, minibatch_size):
                mb = mb_inds[start : start + minibatch_size]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb], b_act.long()[mb]
                )
                logratio = newlogprob - b_logp[mb]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_adv_norm = b_adv[mb]
                mb_adv_norm = (mb_adv_norm - mb_adv_norm.mean()) / (mb_adv_norm.std() + 1e-8)

                pg_loss = torch.max(
                    -mb_adv_norm * ratio,
                    -mb_adv_norm * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
                ).mean()

                newvalue = newvalue.view(-1)
                v_clipped = b_val[mb] + torch.clamp(newvalue - b_val[mb], -args.clip_coef, args.clip_coef)
                v_loss = 0.5 * torch.max(
                    (newvalue - b_ret[mb]) ** 2,
                    (v_clipped - b_ret[mb]) ** 2,
                ).mean()

                loss = pg_loss - args.ent_coef * entropy.mean() + v_loss * args.vf_coef

                agent_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                agent_optimizer.step()

        # ── AE update (periodic, replay-pool sourced, decoupled optimizer) ──
        ae_recon_loss_val = None
        ae_sat_loss_val = None
        have_ae_batch = len(replay_pool) >= args.ae_batch_size
        if have_ae_batch and iteration % args.ae_update_every == 0:
            ae_model.train()  # BatchNorm uses this batch's statistics only while training
            ae_batch = replay_pool.sample(args.ae_batch_size, device)
            logits, b_clean, target_bins = ae_model(ae_batch)
            ae_recon_loss = F.cross_entropy(logits, target_bins, label_smoothing=args.ae_label_smoothing)
            ae_sat_loss = torch.minimum((1 - b_clean) ** 2, b_clean ** 2).mean(dim=1).mean()
            ae_loss = args.ae_recon_coef * ae_recon_loss + args.ae_sat_coef * ae_sat_loss

            ae_optimizer.zero_grad()
            ae_loss.backward()
            nn.utils.clip_grad_norm_(ae_model.parameters(), args.max_grad_norm)
            ae_optimizer.step()
            ae_model.eval()  # back to running-stats normalisation for hashing/health-check/preview

            ae_recon_loss_val = ae_recon_loss.item()
            ae_sat_loss_val = ae_sat_loss.item()

        # health-check metric for AE code collapse, on a fresh replay-pool sample
        bit_variance = None
        if have_ae_batch:
            with torch.no_grad():
                health_batch = replay_pool.sample(args.ae_batch_size, device)
                b_clean_health, _ = ae_model.encode(health_batch, add_noise=False)
                bit_variance = b_clean_health.var(dim=0).mean().item()

        if iteration % args.image_log_interval == 0 and len(replay_pool) >= 8:
            with torch.no_grad():
                sample = replay_pool.sample(8, device)
                target_ds = ae_model.downsample(sample)
                b_clean_sample, _ = ae_model.encode(sample, add_noise=False)
                recon_logits = ae_model.decode(b_clean_sample)
                recon_preview = ae_model.reconstruction_preview(recon_logits)
            # log the downsampled target, not the raw 84x84 frame -- that's what the
            # network actually sees and is being trained to reconstruct
            writer.add_images("debug/ae_original", target_ds, global_step, dataformats="NCHW")
            writer.add_images("debug/ae_reconstruction", recon_preview, global_step, dataformats="NCHW")

        y_pred, y_true = b_val.cpu().numpy(), b_ret.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        sps = int(global_step / (time.time() - start_time))
        print(f"iteration={iteration}/{num_iterations}  SPS={sps}  unique_states={counter.num_unique}")
        writer.add_scalar("charts/learning_rate",       agent_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/SPS",                 sps,                             global_step)
        writer.add_scalar("charts/unique_states",       counter.num_unique,              global_step)
        writer.add_scalar("charts/mean_intrinsic_rew",  np.mean(intrinsic_log),          global_step)
        writer.add_scalar("charts/ae_replay_pool_size", len(replay_pool),                global_step)
        if bit_variance is not None:
            writer.add_scalar("charts/ae_code_bit_variance", bit_variance,              global_step)
        writer.add_scalar("losses/value_loss",          v_loss.item(),                   global_step)
        writer.add_scalar("losses/policy_loss",         pg_loss.item(),                  global_step)
        writer.add_scalar("losses/entropy",             entropy.mean().item(),           global_step)
        writer.add_scalar("losses/approx_kl",           approx_kl.item(),                global_step)
        writer.add_scalar("losses/clipfrac",            np.mean(clipfracs),              global_step)
        writer.add_scalar("losses/explained_variance",  explained_var,                   global_step)
        if ae_recon_loss_val is not None:
            writer.add_scalar("losses/ae_recon_loss",   ae_recon_loss_val,               global_step)
            writer.add_scalar("losses/ae_sat_loss",     ae_sat_loss_val,                 global_step)

        if iteration % args.checkpoint_interval == 0 or iteration == num_iterations:
            ckpt_path = os.path.join(args.checkpoint_dir, run_name, f"ckpt_{iteration:06d}.pt")
            _save_checkpoint(ckpt_path, iteration, global_step, agent, ae_model, agent_optimizer, ae_optimizer, args)

    envs.close()
    writer.close()


if __name__ == "__main__":
    train()
