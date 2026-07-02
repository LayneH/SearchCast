"""Automated hyperparameter search for Ridge-regression time-series forecasting.

Searches over scaling strategies, lookback windows, regularization, and data
augmentation with Optuna, and compares the resulting per-series/horizon "local"
models against a global baseline. See README.md for usage and CLI arguments.
"""

import argparse
import csv
import hashlib
import json
import os
import math
import time
from collections import OrderedDict
from contextlib import contextmanager

import torch
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import optuna

# ==========================================
# 0. Environment Setup
# ==========================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
optuna.logging.set_verbosity(optuna.logging.WARNING)


class StageProfiler:
    """Coarse wall-time profiler for pipeline stages (prep/gram/solve/predict).

    Disabled by default; when enabled it brackets each stage with
    torch.cuda.synchronize() so GPU work is attributed to the right stage.
    """

    def __init__(self):
        self.enabled = False
        self.totals = {}
        self.counts = {}

    @contextmanager
    def stage(self, name):
        if not self.enabled:
            yield
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            self.totals[name] = self.totals.get(name, 0.0) + dt
            self.counts[name] = self.counts.get(name, 0) + 1

    def report(self):
        return {name: {"total_s": self.totals[name], "count": self.counts[name]}
                for name in sorted(self.totals)}


PROFILER = StageProfiler()

# Run-wide knobs configured once in main(). aug_base_seed makes augmentation
# noise a deterministic function of (config, series, row) — reproducible under
# --seed and consistent across cache prefix extensions.
RUN_CONFIG = {"aug_base_seed": 0}


def _stable_seed(key):
    """Deterministic 63-bit seed from an arbitrary (repr-able) key.

    Python's hash() is salted per process, so it cannot seed reproducible RNGs.
    """
    digest = hashlib.blake2b(repr(key).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") >> 1


class GramCache:
    """Byte-budgeted LRU cache of prefix-Gram checkpoints.

    Window row i of a series depends only on (series, lookback, transform,
    noise) — not on the fold or horizon-group cutoff — and expanding-window
    folds / horizon groups only vary the number of leading rows used. So a
    Gram matrix is cached as checkpoints at row cutoffs and any request is
    served by extending the nearest checkpoint below with a delta matmul over
    only the new rows (never subtracting).

    Checkpoints are only stored when at least `min_gap` rows from an existing
    one, so an entry holds ~n_folds checkpoints instead of one per
    (fold x horizon-group) cutoff; requests between checkpoints pay a small
    delta extension instead of storage.
    """

    def __init__(self, max_bytes=4 << 30, enabled=True, min_gap=1024, verify_frac=0.0):
        self.max_bytes = max_bytes
        self.enabled = enabled
        self.min_gap = min_gap
        self.verify_frac = verify_frac
        self._entries = OrderedDict()  # key -> {n_rows: tensor}
        self._bytes = 0
        self.hits = 0
        self.partial_hits = 0
        self.misses = 0
        self._verify_countdown = int(1 / verify_frac) if verify_frac > 0 else 0

    def get_or_build(self, key, n_rows, make_empty, extend):
        """Return the accumulator for exactly the first n_rows window rows.

        make_empty() -> zeroed accumulator tensor
        extend(acc, lo, hi) -> accumulate rows [lo, hi) into acc in place

        The returned tensor may be a live cache entry — callers must not
        mutate it.
        """
        if not self.enabled:
            acc = make_empty()
            extend(acc, 0, n_rows)
            return acc

        cps = self._entries.get(key)
        base_n = max((n for n in cps if n <= n_rows), default=None) if cps else None

        if base_n == n_rows:
            self.hits += 1
            self._entries.move_to_end(key)
            acc = cps[base_n]
            if self.verify_frac > 0:
                # deterministic 1-in-N sampling; avoids touching global RNG state
                self._verify_countdown -= 1
                if self._verify_countdown <= 0:
                    self._verify_countdown = int(1 / self.verify_frac)
                    ref = make_empty()
                    extend(ref, 0, n_rows)
                    err = (acc - ref).abs().max().item()
                    scale = max(ref.abs().max().item(), 1e-30)
                    assert err / scale < 1e-9, \
                        f"GramCache self-check failed for {key}: rel err {err / scale:.3e}"
            return acc

        if base_n is None:
            self.misses += 1
            acc = make_empty()
            lo = 0
        else:
            self.partial_hits += 1
            self._entries.move_to_end(key)
            acc = cps[base_n].clone()
            lo = base_n
        extend(acc, lo, n_rows)

        if cps is None or all(abs(n_rows - n) >= self.min_gap for n in cps):
            self._store(key, n_rows, acc)
        return acc

    def _store(self, key, n_rows, tensor):
        entry_bytes = tensor.numel() * tensor.element_size()
        if entry_bytes > self.max_bytes:
            return
        cps = self._entries.setdefault(key, {})
        cps[n_rows] = tensor
        self._bytes += entry_bytes
        self._entries.move_to_end(key)
        while self._bytes > self.max_bytes and len(self._entries) > 1:
            old_key, old_cps = self._entries.popitem(last=False)
            self._bytes -= sum(t.numel() * t.element_size() for t in old_cps.values())

    def stats(self):
        return {"hits": self.hits, "partial_hits": self.partial_hits,
                "misses": self.misses, "bytes": self._bytes,
                "entries": len(self._entries)}


GRAM_CACHE = GramCache()

# ==========================================
# 1. Scaling Strategies & Scalers
# ==========================================
class ScalerStrategy:
    """Base class for computing center and scale statistics separately."""

    def compute_center(self, tensor, dim):
        raise NotImplementedError

    def compute_scale(self, tensor, dim):
        raise NotImplementedError


class StandardStrategy(ScalerStrategy):
    """Mean for center, std for scale."""

    def compute_center(self, tensor, dim):
        return torch.mean(tensor, dim=dim, keepdim=True)

    def compute_scale(self, tensor, dim):
        # Biased variance (correction=0) to match numpy/reference
        var = torch.var(tensor, dim=dim, keepdim=True, correction=0)
        return torch.sqrt(var + 1e-5)


class RobustStrategy(ScalerStrategy):
    """Median for center, IQR for scale."""

    def __init__(self, q_min=0.25, q_max=0.75):
        self.q_min = q_min
        self.q_max = q_max

    def compute_center(self, tensor, dim):
        return torch.quantile(tensor, 0.50, dim=dim, keepdim=True)

    def compute_scale(self, tensor, dim):
        q_l = torch.quantile(tensor, self.q_min, dim=dim, keepdim=True)
        q_h = torch.quantile(tensor, self.q_max, dim=dim, keepdim=True)
        return (q_h - q_l).clamp_min(1e-5)


class IdentityStrategy(ScalerStrategy):
    """No normalization."""

    def compute_center(self, tensor, dim):
        return torch.zeros(1, device=tensor.device)

    def compute_scale(self, tensor, dim):
        return torch.ones(1, device=tensor.device)


class GlobalScaler:
    """
    Global normalization: (X - center) / scale.

    Both center and scale are computed on the original (non-centered) data.
    Computes global (scalar) statistics over all data for dataset-level preprocessing.
    """

    def __init__(self, strategy: ScalerStrategy):
        self.strategy = strategy
        self.center = None
        self.scale = None

    def fit(self, X: torch.Tensor):
        """Compute global center and scale statistics.

        X: (S, T) series × time
        Computes scalar statistics over all values.
        """
        # Flatten to compute global statistics
        X_flat = X.reshape(-1, 1)  # (S*T, 1)
        self.center = self.strategy.compute_center(X_flat, dim=0)  # (1, 1) scalar
        self.scale = self.strategy.compute_scale(X_flat, dim=0)  # (1, 1) scalar

    def transform(self, X: torch.Tensor):
        """Apply (X - center) / scale."""
        if self.center is None:
            raise RuntimeError("Scaler not fitted")
        if isinstance(self.strategy, IdentityStrategy):
            return X
        center = self.center.to(X.device, non_blocking=True)
        scale = self.scale.to(X.device, non_blocking=True)
        return (X - center) / scale

    def inv_transform(self, X: torch.Tensor):
        """Reverse: X * scale + center."""
        if self.center is None:
            raise RuntimeError("Scaler not fitted")
        center = self.center.to(X.device, non_blocking=True)
        scale = self.scale.to(X.device, non_blocking=True)
        return X * scale + center


class LocalNormScaler:
    """
    Local normalization (instance-norm style).

    Center is computed on original data, scale is computed on centered data.
    Transform subtracts center and appends scale as a feature.

    Args:
        strategy: ScalerStrategy (Standard, Robust, or Identity)
        lookback: context length
        last_k: trailing samples to compute stats from (default: lookback)
    """

    def __init__(self, strategy: ScalerStrategy, lookback: int, last_k: int = None):
        self.strategy = strategy
        self.lookback = lookback
        self.last_k = last_k or lookback
        self.center = None
        self.scale = None

    def fit(self, X: torch.Tensor):
        """Compute center on original data, scale on centered data."""
        wins = X[:, -self.last_k:] if X.dim() == 2 and X.shape[-1] >= self.last_k else X
        self.center = self.strategy.compute_center(wins, dim=-1)
        # Compute scale on centered data
        centered = wins - self.center
        # self.scale = self.strategy.compute_scale(centered, dim=-1)
        # the scale is std deviation regardless of strategy for local norm
        var = torch.var(centered, dim=-1, keepdim=True, correction=0)
        self.scale = torch.sqrt(var + 1e-5)

    def transform(self, X: torch.Tensor):
        """Subtract center, append scale as feature. Returns (N, L+1)."""
        if self.center is None:
            raise RuntimeError("Scaler not fitted")
        center = self.center.to(X.device, non_blocking=True)
        scale = self.scale.to(X.device, non_blocking=True)
        X_centered = X - center
        return torch.cat([X_centered, scale], dim=-1)

    def transform_target(self, Y: torch.Tensor):
        """Subtract center from target (no scale appending)."""
        if self.center is None:
            raise RuntimeError("Scaler not fitted")
        center = self.center.to(Y.device, non_blocking=True)
        return Y - center

    def inv_transform(self, pred: torch.Tensor):
        """Add center back to predictions."""
        if self.center is None:
            raise RuntimeError("Scaler not fitted")
        center = self.center.to(pred.device, non_blocking=True)
        return pred + center

    @property
    def output_dim(self):
        """Feature dimension after transform (lookback + 1)."""
        return self.lookback + 1


class Augmentor:
    def apply(self, X, generator=None): return X

    @staticmethod
    def _randn(shape, like, generator):
        """Gaussian draws matching `like`'s device/dtype, optionally seeded.

        torch.randn_like does not accept a generator, so seeded draws go
        through torch.randn explicitly.
        """
        if generator is None:
            return torch.randn(shape, device=like.device, dtype=like.dtype)
        return torch.randn(shape, device=like.device, dtype=like.dtype,
                           generator=generator)

class TimeDomainNoise(Augmentor):
    def __init__(self, sigma): self.sigma = sigma
    def apply(self, X, generator=None):
        if self.sigma <= 0: return X
        return X.add_(self._randn(X.shape, X, generator), alpha=self.sigma)

class FreqDomainNoise(Augmentor):
    def __init__(self, sigma, mode='amplitude'):
        """
        mode: 'amplitude' (scale magnitude) or 'phase' (shift phase)
        """
        self.sigma = sigma
        self.mode = mode

    def apply(self, X, generator=None):
        # X shape: (Batch, Time, Feat) or (Batch, Time)
        # 1. FFT
        # rfft computes the real-input FFT (faster, gives only positive freqs)
        X_freq = torch.fft.rfft(X, dim=1)

        # 2. Perturb
        if self.mode == 'amplitude':
            # Perturb magnitude: Multiply by random scale ~ N(1, sigma)
            noise = self._randn(X_freq.shape, X, generator) * self.sigma
            # Apply to amplitude, keep phase
            # New_Complex = (Old_Abs + Noise) * e^(i * Old_Phase)
            # Scale magnitude directly
            scale = 1.0 + noise
            X_freq = X_freq * scale

        elif self.mode == 'phase':
            # Perturb phase: Add random angle ~ N(0, sigma)
            phase_noise = self._randn(X_freq.shape, X, generator) * self.sigma
            # New_Complex = Old_Complex * e^(i * phase_noise)
            # Euler's formula: e^(ix) = cos(x) + i*sin(x)
            rotation = torch.polar(torch.ones_like(X_freq.abs()), phase_noise)
            X_freq = X_freq * rotation

        # 3. Inverse FFT
        X_aug = torch.fft.irfft(X_freq, n=X.shape[1], dim=1)
        return X_aug


def has_augmentation(config):
    return (config is not None and config.get('noise_type') in ('time', 'freq')
            and config.get('sigma', 0.0) > 0)


def apply_augmentation(X, config=None, generator=None):
    if config is None: return X
    X_aug = X.clone()
    if config['noise_type'] == 'time':
        X_aug = TimeDomainNoise(config['sigma']).apply(X_aug, generator)
    elif config['noise_type'] == 'freq':
        X_aug = FreqDomainNoise(config['sigma']).apply(X_aug, generator)
    return X_aug

# ==========================================
# 2. Solver
# ==========================================
# compute dtype (chunk transforms + big matmuls), accumulate/solve dtype
PRECISION_CONFIGS = {
    "fp64": (torch.float64, torch.float64),
    "mixed": (torch.float32, torch.float64),
    "fp32": (torch.float32, torch.float32),
}


class RidgeSolver:
    def __init__(self, device, precision="fp64"):
        """
        Args:
            device: torch device
            precision: 'fp64' (exact, default), 'mixed' (fp32 Gram matmuls
                accumulated and solved in fp64 — the big-FLOPs win on GPUs
                with slow fp64), or 'fp32' (everything fp32)
        """
        self.device = device
        self.precision = precision
        self.compute_dtype, self.accum_dtype = PRECISION_CONFIGS[precision]

    def _dtypes(self, dtype):
        """Resolve (compute, accumulate) dtypes; an explicit dtype overrides both."""
        if dtype is None:
            return self.compute_dtype, self.accum_dtype
        return dtype, dtype

    @staticmethod
    def _iter_row_chunks(X, Y, chunk_size):
        """Yield 2D (rows, L) / (rows, H) chunks from 2D (N, ...) or 3D (S, N, ...) inputs.

        3D inputs are consumed in series-major order, i.e. equivalent to
        X.reshape(-1, L) without materializing the flattened copy (the windowed
        views produced by unfold cannot be reshaped for free).
        """
        if X.dim() == 2:
            for start in range(0, X.shape[0], chunk_size):
                end = min(start + chunk_size, X.shape[0])
                yield X[start:end], (Y[start:end] if Y is not None else None)
        else:
            for s in range(X.shape[0]):
                for start in range(0, X.shape[1], chunk_size):
                    end = min(start + chunk_size, X.shape[1])
                    yield X[s, start:end], (Y[s, start:end] if Y is not None else None)

    def _aug_generator(self, aug_config):
        """Deterministic per-configuration noise generator for the non-cached
        solve paths (refit, global baseline, per-series batched)."""
        if not has_augmentation(aug_config):
            return None
        gen = torch.Generator(device=self.device)
        gen.manual_seed(_stable_seed((
            "aug", RUN_CONFIG["aug_base_seed"],
            aug_config["noise_type"], round(float(aug_config["sigma"]), 9))))
        return gen

    def _transform_train_chunk(self, X_chunk, Y_chunk, scaler, aug_config, cdt,
                               generator=None, noise_slice=None):
        """Scaler transform + augmentation + intercept/scale feature for one
        raw training chunk, exactly as the Gram accumulation consumes it.

        noise_slice: pre-generated noise rows (from the block-deterministic
        path) applied instead of drawing from `generator`.
        """
        use_local_norm = isinstance(scaler, LocalNormScaler)
        if use_local_norm:
            scaler.fit(X_chunk)
            X_chunk = scaler.transform(X_chunk)
            Y_chunk = scaler.transform_target(Y_chunk)
            X_feat = X_chunk[:, :-1]
            if noise_slice is not None:
                X_feat = self._apply_block_noise(X_feat, aug_config, noise_slice)
            else:
                X_feat = apply_augmentation(X_feat, aug_config, generator)
            X_chunk = torch.cat([X_feat, X_chunk[:, -1:]], dim=-1)
        else:
            if scaler is not None:
                X_chunk = scaler.transform(X_chunk)
                Y_chunk = scaler.transform(Y_chunk)
            if noise_slice is not None:
                X_chunk = self._apply_block_noise(X_chunk, aug_config, noise_slice)
            else:
                X_chunk = apply_augmentation(X_chunk, aug_config, generator)
            ones = torch.ones((X_chunk.shape[0], 1), dtype=cdt, device=self.device)
            X_chunk = torch.cat([ones, X_chunk], dim=1)
        return X_chunk, Y_chunk

    @staticmethod
    def _apply_block_noise(X_feat, aug_config, noise):
        """Apply pre-drawn noise rows (same math as the Augmentor classes)."""
        sigma = aug_config["sigma"]
        if aug_config["noise_type"] == "time":
            return X_feat + sigma * noise
        # freq (amplitude mode): scale rfft magnitudes by N(1, sigma)
        X_freq = torch.fft.rfft(X_feat, dim=1)
        X_freq = X_freq * (1.0 + sigma * noise)
        return torch.fft.irfft(X_freq, n=X_feat.shape[1], dim=1)

    @torch.no_grad()
    def accumulate_gram(self, X_wins, Y_wins, scaler, aug_config, lo, hi,
                        XTX=None, XTY=None, seed_ctx=None, block=8192):
        """Accumulate Gram contributions of window rows [lo, hi) of each series
        into XTX (F, F) and/or XTY (F, H), in place.

        Chunks are aligned to fixed `block`-row boundaries in absolute row
        index, and augmentation noise is drawn per (seed_ctx, series, block)
        with full-block draws sliced to the covered rows — so the noise seen
        by row i is identical no matter which prefix range builds it. That is
        what makes cached prefix Grams extendable for noisy configurations,
        and makes XTY passes consistent with the cached XTX.
        """
        if X_wins.dim() == 2:
            X_wins = X_wins.unsqueeze(0)
            Y_wins = Y_wins.unsqueeze(0)
        S = X_wins.shape[0]
        cdt = self.compute_dtype
        adt = (XTX if XTX is not None else XTY).dtype
        noisy = has_augmentation(aug_config)
        use_local_norm = isinstance(scaler, LocalNormScaler)

        with PROFILER.stage("gram"):
            for s in range(S):
                for b in range(lo // block * block, hi, block):
                    r0, r1 = max(lo, b), min(hi, b + block)
                    if r0 >= r1:
                        continue
                    X_chunk = X_wins[s, r0:r1].to(dtype=cdt, device=self.device,
                                                  non_blocking=True)
                    Y_chunk = Y_wins[s, r0:r1].to(dtype=cdt, device=self.device,
                                                  non_blocking=True)

                    noise_slice = None
                    if noisy:
                        # feature count the noise applies to (excludes the
                        # appended scale feature / prepended intercept)
                        Lf = X_chunk.shape[1]
                        n_cols = (Lf // 2 + 1 if aug_config["noise_type"] == "freq"
                                  else Lf)
                        gen = torch.Generator(device=self.device)
                        gen.manual_seed(_stable_seed(
                            (seed_ctx, RUN_CONFIG["aug_base_seed"], s, b)))
                        block_noise = torch.randn((block, n_cols), device=self.device,
                                                  dtype=cdt, generator=gen)
                        noise_slice = block_noise[r0 - b:r1 - b]

                    X_chunk, Y_chunk = self._transform_train_chunk(
                        X_chunk, Y_chunk, scaler, aug_config, cdt,
                        noise_slice=noise_slice)

                    if XTX is not None:
                        gram = X_chunk.T @ X_chunk
                        XTX.add_(gram if gram.dtype == adt else gram.to(adt))
                    if XTY is not None:
                        cross = X_chunk.T @ Y_chunk
                        XTY.add_(cross if cross.dtype == adt else cross.to(adt))

    @torch.no_grad()
    def solve_from_gram(self, XTX, XTY, alphas, fit_intercept):
        """Solve the regularized systems for every alpha from precomputed
        Gram matrices (see accumulate_gram / GramCache)."""
        with PROFILER.stage("solve"):
            return self._solve_alpha_chunked(XTX, XTY, alphas, fit_intercept)

    @torch.no_grad()
    def solve(self, X_train, Y_train, alphas, scaler=None, chunk_size=50000, aug_config=None, dtype=None):
        """
        Solve ridge regression for multiple alpha values.

        Args:
            X_train: (N, L) or (S, N_s, L) input features (3D is treated as the
                     pooled row-concatenation of the S series)
            Y_train: (N, H) or (S, N_s, H) targets
            alphas: (K,) regularization strengths
            scaler: normalization scaler (LocalNormScaler or GlobalScaler or None)
                    - LocalNormScaler: per-sample normalization, appends scale feature, no intercept
                    - GlobalScaler: dataset-level normalization, adds intercept
                    - None: no normalization, adds intercept
            chunk_size: batch size for memory-efficient computation
            aug_config: augmentation configuration
            dtype: computation dtype

        Returns:
            Theta: (K, F, H) weight matrices for each alpha
        """
        L = X_train.shape[-1]
        H = Y_train.shape[-1]
        K = alphas.shape[0]

        # Determine feature dimension and intercept based on scaler type
        use_local_norm = isinstance(scaler, LocalNormScaler)
        if use_local_norm:
            F = L + 1  # context features + scale feature (no intercept)
            fit_intercept = False
        else:
            F = L + 1  # features + intercept
            fit_intercept = True

        cdt, adt = self._dtypes(dtype)
        XTX = torch.zeros(F, F, device=self.device, dtype=adt)
        XTY = torch.zeros(F, H, device=self.device, dtype=adt)
        aug_gen = self._aug_generator(aug_config)

        with PROFILER.stage("gram"):
            for X_chunk, Y_chunk in self._iter_row_chunks(X_train, Y_train, chunk_size):
                X_chunk = X_chunk.to(dtype=cdt, device=self.device, non_blocking=True)
                Y_chunk = Y_chunk.to(dtype=cdt, device=self.device, non_blocking=True)

                X_chunk, Y_chunk = self._transform_train_chunk(
                    X_chunk, Y_chunk, scaler, aug_config, cdt, generator=aug_gen)

                # Accumulate the gram matrix (per-chunk matmul in the compute
                # dtype, accumulation in the higher-precision dtype)
                if cdt == adt:
                    XTX.add_(X_chunk.T @ X_chunk)
                    XTY.add_(X_chunk.T @ Y_chunk)
                else:
                    XTX.add_((X_chunk.T @ X_chunk).to(adt))
                    XTY.add_((X_chunk.T @ Y_chunk).to(adt))

        with PROFILER.stage("solve"):
            Theta = self._solve_alpha_chunked(XTX, XTY, alphas, fit_intercept)
        return Theta

    def _solve_alpha_chunked(self, XTX, XTY, alphas, fit_intercept, alpha_chunk_size=None):
        """
        Solve (XTX + alpha*D) Theta = XTY for every alpha, chunking over alphas
        so the transient (Kb, ..., F, F) factorization workspace stays bounded
        instead of materializing all K systems at once.

        D = diag(0, 1, ..., 1) when fit_intercept else the identity.

        Args:
            XTX: (F, F) or (Sb, F, F) Gram matrices
            XTY: (F, H) or (Sb, F, H) cross-products
            alphas: (K,) regularization strengths

        Returns:
            Theta: (K, F, H) or (K, Sb, F, H), same device/dtype as XTY
        """
        K = alphas.shape[0]
        F = XTX.shape[-1]
        batched = XTX.dim() == 3
        if alpha_chunk_size is None:
            # ~1GB budget for A plus its Cholesky factor per chunk
            n_systems = XTX.shape[0] if batched else 1
            per_alpha = 2 * n_systems * F * F * XTX.element_size()
            alpha_chunk_size = max(1, min(K, (1 << 30) // max(1, per_alpha)))

        Theta = torch.empty((K,) + XTY.shape, device=XTY.device, dtype=XTY.dtype)
        for k0 in range(0, K, alpha_chunk_size):
            k1 = min(k0 + alpha_chunk_size, K)
            alpha_chunk = alphas[k0:k1].to(device=XTX.device, dtype=XTX.dtype)
            if batched:
                diag_vals = alpha_chunk.view(-1, 1, 1).expand(k1 - k0, XTX.shape[0], F).clone()
                if fit_intercept:
                    diag_vals[:, :, 0] = 0.0  # don't regularize the intercept
            else:
                diag_vals = alpha_chunk.view(-1, 1).expand(k1 - k0, F).clone()
                if fit_intercept:
                    diag_vals[:, 0] = 0.0
            A = XTX.unsqueeze(0) + torch.diag_embed(diag_vals)
            B = XTY.unsqueeze(0).expand((k1 - k0,) + XTY.shape)

            try:
                Lchol = torch.linalg.cholesky(A)
                Theta[k0:k1] = torch.cholesky_solve(B, Lchol)
            except RuntimeError:
                try:
                    Theta[k0:k1] = torch.linalg.solve(A, B)
                except RuntimeError:
                    Theta[k0:k1] = torch.linalg.pinv(A) @ B
        return Theta

    @torch.no_grad()
    def predict(self, X, theta, scaler=None, chunk_size=50000, dtype=None):
        """
        Make predictions using trained weights.

        Args:
            X: (N, L) or (S, N_s, L) input features (raw, before any normalization);
               3D input is consumed in series-major order and returned flattened
            theta: (K, F, H) weight matrices
            scaler: normalization scaler (LocalNormScaler or GlobalScaler or None)
                    - LocalNormScaler: fit, transform (appends scale), predict, inv_transform
                    - GlobalScaler: transform only, predict (caller handles inv_transform)
                    - None: add intercept, predict
            chunk_size: batch size for memory-efficient prediction
            dtype: computation dtype

        Returns:
            Y_pred: (K, N_total, H) predictions on the solver device
        """
        K, _, H = theta.shape
        dtype, _ = self._dtypes(dtype)
        theta = theta.to(dtype=dtype, device=self.device, non_blocking=True)

        use_local_norm = isinstance(scaler, LocalNormScaler)

        # Process in chunks for memory efficiency
        with PROFILER.stage("predict"):
            Y_pred_chunks = []
            for X_chunk, _ in self._iter_row_chunks(X, None, chunk_size):
                X_chunk = X_chunk.to(dtype=dtype, device=self.device, non_blocking=True)

                if use_local_norm:
                    # LocalNormScaler: fit, transform, predict, then inv_transform
                    scaler.fit(X_chunk)
                    X_transformed = scaler.transform(X_chunk)
                    Y_pred_norm = torch.einsum('nf, kfh -> knh', X_transformed, theta)
                    Y_pred_chunk = scaler.inv_transform(Y_pred_norm)
                else:
                    # GlobalScaler or None: add intercept and predict
                    if scaler is not None:
                        X_chunk = scaler.transform(X_chunk)
                    ones = torch.ones((X_chunk.shape[0], 1), dtype=dtype, device=self.device)
                    X_chunk = torch.cat([ones, X_chunk], dim=1)
                    Y_pred_chunk = torch.einsum('nf, kfh -> knh', X_chunk, theta)

                Y_pred_chunks.append(Y_pred_chunk)

            Y_pred = torch.cat(Y_pred_chunks, dim=1)
        return Y_pred

    def _predict_chunk(self, X_chunk, theta, scaler, dtype):
        """Transform one raw (rows, L) chunk, apply theta, undo the normalization.

        Unlike predict(), this also applies the GlobalScaler inverse so the
        result is always in the original data scale.
        """
        if isinstance(scaler, LocalNormScaler):
            scaler.fit(X_chunk)
            X_t = scaler.transform(X_chunk)
            pred = torch.einsum('nf, kfh -> knh', X_t, theta)
            return scaler.inv_transform(pred)
        if scaler is not None:
            X_chunk = scaler.transform(X_chunk)
        ones = torch.ones((X_chunk.shape[0], 1), dtype=dtype, device=self.device)
        X_t = torch.cat([ones, X_chunk], dim=1)
        pred = torch.einsum('nf, kfh -> knh', X_t, theta)
        if isinstance(scaler, GlobalScaler):
            pred = scaler.inv_transform(pred)
        return pred

    @torch.no_grad()
    def val_mse(self, X, Y, theta, scaler=None, chunk_size=50000, dtype=None):
        """
        Fused validation MSE per alpha: prediction and squared error in one
        chunked pass, never materializing the (K, N, H) prediction tensor and
        never leaving the solver device until the final (K,) vector.

        Equivalent to predict() -> inv_transform -> ((pred - Y)**2).mean((-2,-1)).

        Args:
            X: (N, L) or (S, N_s, L) raw validation inputs
            Y: (N, H) or (S, N_s, H) raw validation targets
            theta: (K, F, H) weight matrices

        Returns:
            mse_per_alpha: (K,) tensor on CPU
        """
        K = theta.shape[0]
        H = Y.shape[-1]
        cdt, adt = self._dtypes(dtype)
        theta = theta.to(dtype=cdt, device=self.device, non_blocking=True)

        sse = torch.zeros(K, device=self.device, dtype=adt)
        n_rows = 0
        with PROFILER.stage("predict"):
            for X_chunk, Y_chunk in self._iter_row_chunks(X, Y, chunk_size):
                X_chunk = X_chunk.to(dtype=cdt, device=self.device, non_blocking=True)
                Y_chunk = Y_chunk.to(dtype=cdt, device=self.device, non_blocking=True)
                pred = self._predict_chunk(X_chunk, theta, scaler, cdt)
                sse += ((pred - Y_chunk.unsqueeze(0)) ** 2).sum(dim=(-2, -1), dtype=adt)
                n_rows += X_chunk.shape[0]
        return (sse / (n_rows * H)).cpu()

    @torch.no_grad()
    def val_mse_batched(self, X_batch, Y_batch, theta, scaler=None, chunk_size=50000, dtype=None):
        """
        Fused validation MSE for per-series models: per-series MSE averaged
        across series, per alpha.

        Equivalent to predict_batched() -> inv_transform ->
        mean over (N, H) per series -> mean over series.

        Args:
            X_batch: (S, N_s, L) raw validation inputs
            Y_batch: (S, N_s, H) raw validation targets
            theta: (K, S, F, H) weight matrices

        Returns:
            mse_per_alpha: (K,) tensor on CPU
        """
        S, N_s, _ = X_batch.shape
        K = theta.shape[0]
        H = Y_batch.shape[-1]
        cdt, adt = self._dtypes(dtype)

        sse = torch.zeros(K, S, device=self.device, dtype=adt)
        with PROFILER.stage("predict"):
            for s in range(S):
                theta_s = theta[:, s].to(dtype=cdt, device=self.device, non_blocking=True)
                for start in range(0, N_s, chunk_size):
                    end = min(start + chunk_size, N_s)
                    X_chunk = X_batch[s, start:end].to(dtype=cdt, device=self.device, non_blocking=True)
                    Y_chunk = Y_batch[s, start:end].to(dtype=cdt, device=self.device, non_blocking=True)
                    pred = self._predict_chunk(X_chunk, theta_s, scaler, cdt)
                    sse[:, s] += ((pred - Y_chunk.unsqueeze(0)) ** 2).sum(dim=(-2, -1), dtype=adt)
        mse_per_series = sse / (N_s * H)  # (K, S)
        return mse_per_series.mean(dim=1).cpu()

    @torch.no_grad()
    def solve_batched(self, X_train_batch, Y_train_batch, alphas, scaler=None, chunk_size=50000, aug_config=None, dtype=None):
        """
        Solve ridge regression for multiple series simultaneously (batched).

        Each series gets its own separate Ridge model with unique weights.
        Adaptively batches over series and alphas to fit GPU memory.

        Args:
            X_train_batch: (S, N_s, L) input features for S series
            Y_train_batch: (S, N_s, H) targets for S series
            alphas: (K,) regularization strengths
            scaler: normalization scaler (LocalNormScaler or GlobalScaler or None)
            chunk_size: batch size for memory-efficient computation per series
            aug_config: augmentation configuration
            dtype: computation dtype

        Returns:
            Theta: (K, S, F, H) weight matrices on the solver device
        """
        S, N_s, L = X_train_batch.shape
        _, _, H = Y_train_batch.shape
        K = alphas.shape[0]

        # Determine feature dimension and intercept based on scaler type
        use_local_norm = isinstance(scaler, LocalNormScaler)
        if use_local_norm:
            F = L + 1  # context features + scale feature (no intercept)
            fit_intercept = False
        else:
            F = L + 1  # features + intercept
            fit_intercept = True

        cdt, adt = self._dtypes(dtype)
        Theta = torch.zeros(K, S, F, H, device=self.device, dtype=adt)

        # Determine series_batch: must fit Gram matrices + at least 1 alpha solve
        # Gram: Sb * (F*F + F*H) * 8 bytes for XTX + XTY
        # Solve (min): 1 * Sb * F*F * 8 * 3 bytes for A, L, workspace
        if self.device.type == "cuda":
            free_mem, _ = torch.cuda.mem_get_info(self.device)
            gpu_budget = int(free_mem * 0.7)
        else:
            gpu_budget = 4 << 30  # fixed working budget on CPU
        bytes_per_series = (F * F + F * H + F * F * 3) * 8  # Gram + 1-alpha solve
        series_batch = max(1, min(S, gpu_budget // bytes_per_series))

        for s_start in range(0, S, series_batch):
            s_end = min(s_start + series_batch, S)
            Sb = s_end - s_start

            # --- Phase 1: Accumulate Gram matrices for this series batch ---
            XTX = torch.zeros(Sb, F, F, device=self.device, dtype=adt)
            XTY = torch.zeros(Sb, F, H, device=self.device, dtype=adt)

            with PROFILER.stage("gram"):
                for si, s in enumerate(range(s_start, s_end)):
                    X_s = X_train_batch[s]  # (N_s, L)
                    Y_s = Y_train_batch[s]  # (N_s, H)
                    aug_gen = self._aug_generator(aug_config)

                    for start in range(0, N_s, chunk_size):
                        end = min(start + chunk_size, N_s)
                        X_chunk = X_s[start:end].to(dtype=cdt, device=self.device, non_blocking=True)
                        Y_chunk = Y_s[start:end].to(dtype=cdt, device=self.device, non_blocking=True)

                        X_chunk, Y_chunk = self._transform_train_chunk(
                            X_chunk, Y_chunk, scaler, aug_config, cdt, generator=aug_gen)

                        if cdt == adt:
                            XTX[si].add_(X_chunk.T @ X_chunk)
                            XTY[si].add_(X_chunk.T @ Y_chunk)
                        else:
                            XTX[si].add_((X_chunk.T @ X_chunk).to(adt))
                            XTY[si].add_((X_chunk.T @ Y_chunk).to(adt))

            # --- Phase 2: Solve for all alphas, chunked over alphas ---
            with PROFILER.stage("solve"):
                Theta[:, s_start:s_end] = self._solve_alpha_chunked(
                    XTX, XTY, alphas, fit_intercept)

            del XTX, XTY

        return Theta

    @torch.no_grad()
    def predict_batched(self, X_batch, theta, scaler=None, chunk_size=50000, dtype=None):
        """
        Make predictions using batched trained weights (one model per series).

        Args:
            X_batch: (S, N_s, L) input features for S series
            theta: (K, S, F, H) weight matrices
            scaler: normalization scaler (LocalNormScaler or GlobalScaler or None)
            chunk_size: batch size for memory-efficient prediction per series
            dtype: computation dtype

        Returns:
            Y_pred: (K, S, N_s, H) predictions
        """
        S, N_s, L = X_batch.shape
        K, _, _, H = theta.shape
        dtype, _ = self._dtypes(dtype)

        use_local_norm = isinstance(scaler, LocalNormScaler)

        Y_pred_all = []

        # Process each series
        with PROFILER.stage("predict"):
            for s in range(S):
                X_s = X_batch[s]  # (N_s, L)
                theta_s = theta[:, s, :, :].to(dtype=dtype, device=self.device, non_blocking=True)  # (K, F, H)

                # Process in chunks for memory efficiency
                Y_pred_chunks = []
                for start in range(0, N_s, chunk_size):
                    end = min(start + chunk_size, N_s)
                    X_chunk = X_s[start:end].to(dtype=dtype, device=self.device, non_blocking=True)

                    if use_local_norm:
                        # LocalNormScaler: fit, transform, predict, then inv_transform
                        scaler.fit(X_chunk)
                        X_transformed = scaler.transform(X_chunk)
                        Y_pred_norm = torch.einsum('nf, kfh -> knh', X_transformed, theta_s)
                        Y_pred_chunk = scaler.inv_transform(Y_pred_norm)
                    else:
                        # GlobalScaler or None: add intercept and predict
                        if scaler is not None:
                            X_chunk = scaler.transform(X_chunk)
                        ones = torch.ones((X_chunk.shape[0], 1), dtype=dtype, device=self.device)
                        X_chunk = torch.cat([ones, X_chunk], dim=1)
                        Y_pred_chunk = torch.einsum('nf, kfh -> knh', X_chunk, theta_s)

                    Y_pred_chunks.append(Y_pred_chunk)

                Y_pred_s = torch.cat(Y_pred_chunks, dim=1)  # (K, N_s, H)
                Y_pred_all.append(Y_pred_s)

            Y_pred = torch.stack(Y_pred_all, dim=1)  # (K, S, N_s, H)
        return Y_pred

# ==========================================
# 3. Data Pipeline
# ==========================================
def get_context_and_horizons(data, lookback, horizons):
    if isinstance(horizons, int):
        H_max = horizons
        idx_tensor = torch.tensor([horizons], dtype=torch.long, device=data.device)
    else:
        H_max = max(horizons)
        idx_tensor = torch.tensor(horizons, dtype=torch.long, device=data.device)

    # Dynamic Window: lookback + max_horizon required
    window_size = lookback + H_max
    if data.shape[1] < window_size:
        return None, None

    data_windowed = data.unfold(1, window_size, 1)
    X = data_windowed[:, :, :lookback]
    target_indices = lookback + idx_tensor - 1
    Y = torch.index_select(data_windowed, dim=2, index=target_indices)
    return X, Y


def create_strategy(method: str, scaler_config: dict) -> ScalerStrategy:
    """Create a ScalerStrategy based on method name."""
    if method == "mean":
        return StandardStrategy()
    elif method == "robust":
        q_min = scaler_config.get('q_min', 0.25)
        q_max = scaler_config.get('q_max', 0.75)
        return RobustStrategy(q_min, q_max)
    elif method == "none":
        return IdentityStrategy()
    else:
        raise ValueError(f"Unknown scaler method: {method}")


def create_scaler(strategy: ScalerStrategy, scope: str, lookback: int,
                  scaler_config: dict):
    """Create GlobalScaler or LocalNormScaler based on scope."""
    if scope == 'global':
        return GlobalScaler(strategy)
    elif scope == 'local':
        ratio = scaler_config.get('local_ratio', 1.0)
        last_k = max(1, int(lookback * ratio))
        return LocalNormScaler(strategy, lookback, last_k)
    else:
        raise ValueError(f"Unknown scope: {scope}")


def maybe_transform(X: torch.Tensor, scaler, force_no_fit: bool = False):
    """
    Apply scaler transformation, fitting if needed for LocalNormScaler.

    For GlobalScaler: assumes already fitted, just transforms.
    For LocalNormScaler: fits on X then transforms (unless force_no_fit).
    """
    if X is None:
        return None
    if isinstance(scaler, LocalNormScaler):
        if not force_no_fit:
            scaler.fit(X)
        return scaler.transform(X)
    elif isinstance(scaler, GlobalScaler):
        return scaler.transform(X)
    else:
        return X


def get_prepared_data(series_data, lookback, horizons, split_idx_1, split_idx_2,
                      scaler_config, include_val=True, include_test=True):
    """
    Prepare train/val/test data with appropriate scalers.

    include_val/include_test skip building window sets the caller will not use
    (fold evaluation never touches test windows; refit never touches val).

    Returns:
        X_train, Y_train, X_val, Y_val, X_test, Y_test, scalers_dict
    """
    if not isinstance(horizons, (list, tuple)):
        horizons = [horizons]

    method = scaler_config['method']
    scope = scaler_config['scope']

    strategy = create_strategy(method, scaler_config)

    # Slice boundaries
    test_slice_start = max(0, split_idx_2 - lookback)

    # Get windows
    X_train, Y_train = get_context_and_horizons(
        series_data[:, :split_idx_1], lookback, horizons)

    if include_val and split_idx_2 > split_idx_1:
        X_val, Y_val = get_context_and_horizons(
            series_data[:, split_idx_1 - lookback:split_idx_2],
            lookback, horizons)
    else:
        X_val, Y_val = None, None

    if include_test:
        X_test, Y_test = get_context_and_horizons(
            series_data[:, test_slice_start:], lookback, horizons)
    else:
        X_test, Y_test = None, None

    # Create scalers
    if scope == 'global':
        scaler_train = GlobalScaler(strategy)
        scaler_train.fit(series_data[:, :split_idx_1])
        scaler_val = scaler_test = scaler_train
    elif scope == 'local':
        ratio = scaler_config.get('local_ratio', 1.0)
        last_k = max(1, int(lookback * ratio))
        scaler_train = LocalNormScaler(strategy, lookback, last_k)
        scaler_val = LocalNormScaler(strategy, lookback, last_k)
        scaler_test = LocalNormScaler(strategy, lookback, last_k)
    else:
        raise ValueError(f"Unknown scope: {scope}")

    scalers = {'train': scaler_train, 'val': scaler_val, 'test': scaler_test}
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, scalers

# ==========================================
# 5. Objective Wrappers
# ==========================================
class SingleObjectiveWrapper:
    def __init__(self, data, series_idx, lookbacks, horizon, alphas, device,
                 train_ratio=0.4, test_ratio=0.2, split_starts=None, split_ends=None,
                 n_folds=1, fold_reg_lambda=0.0,
                 scaler_scope="search", scaler_method="search",
                 fixed_local_ratio=None, fixed_noise_type=None, fixed_aug_sigma=None,
                 pool_series=False, precision="fp64"):
        """
        Args:
            data: (T, S) time series data
            series_idx: index of target series, or None for all series
            lookbacks: list of valid lookback values for Optuna to search over
            horizon: forecast horizon(s)
            alphas: regularization strengths
            device: torch device
            train_ratio: fraction of data for training during search
            test_ratio: fraction of data for test (not used during search)
            split_starts: list of split start indices
            split_ends: list of split end indices
            n_folds: number of folds for expanding window CV (1 = single split)
            fold_reg_lambda: regularization weight for fold variance (0 = no regularization)
            scaler_scope: "global", "local", or "search" (default: "search")
            scaler_method: "mean", "robust", or "search" (default: "search")
            fixed_local_ratio: if set, use this value instead of searching
            fixed_noise_type: if set, use this augmentation type instead of searching
            fixed_aug_sigma: if set, use this augmentation sigma instead of searching
            pool_series: if True, pool training data across series (single model per group)
        """
        if series_idx is None:
            self.data = data.transpose(0, 1)
            self.is_batched = True
            self.n_series = data.shape[1]
        elif isinstance(series_idx, (list, tuple)):
            self.data = data[:, series_idx].transpose(0, 1)
            self.is_batched = len(series_idx) > 1
            self.n_series = len(series_idx)
        else:
            self.data = data[:, series_idx].unsqueeze(0)
            self.is_batched = False
            self.n_series = 1
        # Keep the (small) series data resident on the compute device so all
        # windowing/unfold operations below are device-side views.
        self.data = self.data.to(device).contiguous()

        # Identity of this wrapper's data slice for GramCache keys, plus a
        # memo of already-evaluated hyperparameter combinations (duplicate
        # Optuna trials are common in categorical-heavy spaces).
        if series_idx is None:
            self._series_key = ("all",)
        elif isinstance(series_idx, (list, tuple)):
            self._series_key = tuple(series_idx)
        else:
            self._series_key = (series_idx,)
        self._memo = {}
        # Store lookback bounds for log-scale search
        self.min_lookback = min(lookbacks)
        self.max_lookback = max(lookbacks)
        self.horizon = horizon
        self.alphas = alphas
        self.device = device
        self.solver = RidgeSolver(device, precision=precision)
        self.n_folds = n_folds
        self.fold_reg_lambda = fold_reg_lambda
        self.pool_series = pool_series

        # Ablation controls
        self.scaler_scope = scaler_scope
        self.scaler_method = scaler_method
        self.fixed_local_ratio = fixed_local_ratio
        self.fixed_noise_type = fixed_noise_type
        self.fixed_aug_sigma = fixed_aug_sigma

        T = self.data.shape[1]

        # first train_ratio data is used as training set during search
        self.split_idx_1 = int(T * train_ratio)
        # the final test_ratio is test, which is not used during search
        self.split_idx_2 = int(T * (1 - test_ratio))

        # use default 0.7/0.1/0.2 split for benchmark if the splits is not provided
        self.split_starts = split_starts
        self.split_ends = split_ends

        # Precompute fold boundaries for expanding window k-fold CV
        if n_folds > 1:
            trainval_len = self.split_idx_2
            fold_size = trainval_len // (n_folds + 1)
            # Boundaries: [fold_size, 2*fold_size, ..., (n_folds+1)*fold_size]
            self.fold_boundaries = [fold_size * (i + 1) for i in range(n_folds + 1)]
        else:
            self.fold_boundaries = None

    def _evaluate_fold(self, train_end, val_end, scaler_config, aug_config, lookback):
        """
        Evaluate a single train/val split.

        Args:
            train_end: end index of training data
            val_end: end index of validation data
            scaler_config: scaler configuration dict
            aug_config: augmentation configuration dict
            lookback: context length for this evaluation

        Returns:
            mse_per_alpha: (K,) tensor of MSE for each alpha, or None if fold is invalid
        """
        try:
            with PROFILER.stage("prep"):
                X_train_w, Y_train, X_val_w, Y_val, _, _, scalers = \
                    get_prepared_data(self.data, lookback, self.horizon,
                                      train_end, val_end, scaler_config,
                                      include_test=False)
        except ValueError:
            return None

        if X_train_w is None or X_val_w is None:
            return None

        if self.is_batched and not self.pool_series:
            # Batched mode: keep series separate (S, N_s, L), one model per
            # series. Per-series (S, F, F) Gram stacks are too large to cache,
            # so this path stays uncached.
            Theta = self.solver.solve_batched(
                X_train_w, Y_train, self.alphas,
                scaler=scalers['train'],
                aug_config=aug_config
            )  # (K, S, F, H)

            # Fused per-series validation MSE (never materializes predictions)
            mse_per_alpha = self.solver.val_mse_batched(
                X_val_w, Y_val, Theta, scaler=scalers['val'])  # (K,)
        else:
            # Single-series / pooled mode: one model over the pooled window
            # rows — served from prefix-Gram checkpoints when cached
            mse_per_alpha = self._eval_fold_from_gram(
                X_train_w, Y_train, X_val_w, Y_val, scalers,
                scaler_config, aug_config, lookback, train_end)

        return mse_per_alpha

    def _eval_fold_from_gram(self, X_train_w, Y_train, X_val_w, Y_val, scalers,
                             scaler_config, aug_config, lookback, train_end):
        """Pooled/single-series fold evaluation through the GramCache.

        XTX depends only on (series, lookback, transform, noise) and the row
        cutoff — folds and horizon groups are nested prefixes of the same
        window sequence — so it is cached across trials, folds, and
        horizon-group studies. XTY additionally depends on the horizon tuple
        (a few columns) and gets its own entries.
        """
        if X_train_w.dim() == 2:
            X_train_w = X_train_w.unsqueeze(0)
            Y_train = Y_train.unsqueeze(0)
        n_rows = X_train_w.shape[1]
        H = Y_train.shape[-1]
        F = lookback + 1  # + scale feature (local) or intercept (global)
        scaler_train = scalers['train']
        use_local_norm = isinstance(scaler_train, LocalNormScaler)

        if use_local_norm:
            # per-window transform: prefix-safe across folds
            xform_key = ("local", scaler_config["method"], scaler_train.last_k)
        else:
            # global scaler stats are fit on [:train_end] — fold-dependent,
            # so entries are only shared within a fold (still across the
            # horizon-group studies and duplicate trials)
            xform_key = ("global", scaler_config["method"], train_end)
        if has_augmentation(aug_config):
            aug_key = (aug_config["noise_type"], round(float(aug_config["sigma"]), 9))
        else:
            aug_key = ("clean",)
        base_key = (self._series_key, self.pool_series, lookback, xform_key, aug_key)
        adt = self.solver.accum_dtype

        def make_xtx():
            return torch.zeros(F, F, device=self.device, dtype=adt)

        def extend_xtx(acc, lo, hi):
            self.solver.accumulate_gram(X_train_w, Y_train, scaler_train,
                                        aug_config, lo, hi, XTX=acc,
                                        seed_ctx=base_key)

        def make_xty():
            return torch.zeros(F, H, device=self.device, dtype=adt)

        def extend_xty(acc, lo, hi):
            self.solver.accumulate_gram(X_train_w, Y_train, scaler_train,
                                        aug_config, lo, hi, XTY=acc,
                                        seed_ctx=base_key)

        horizon_t = (tuple(self.horizon) if isinstance(self.horizon, (list, tuple))
                     else (self.horizon,))
        XTX = GRAM_CACHE.get_or_build(("xtx",) + base_key, n_rows,
                                      make_xtx, extend_xtx)
        XTY = GRAM_CACHE.get_or_build(("xty",) + base_key + (horizon_t,), n_rows,
                                      make_xty, extend_xty)

        Theta = self.solver.solve_from_gram(XTX, XTY, self.alphas,
                                            fit_intercept=not use_local_norm)
        return self.solver.val_mse(X_val_w, Y_val, Theta, scaler=scalers['val'])

    def __call__(self, trial):
        # Suggest lookback using log-scale search (more efficient for context length)
        lookback = trial.suggest_int("lookback", self.min_lookback, self.max_lookback, log=True)

        # Scaler scope (global vs local)
        if self.scaler_scope == "search":
            scaler_scope = trial.suggest_categorical("scaler_scope", ["global", "local"])
        else:
            scaler_scope = self.scaler_scope

        # Scaler method (mean vs robust)
        if self.scaler_method == "search":
            scaler_method = trial.suggest_categorical("scaler_method", ["mean", "robust"])
        else:
            scaler_method = self.scaler_method

        # Local ratio (only relevant for local normalization)
        if scaler_scope == "local":
            if self.fixed_local_ratio is not None:
                local_ratio = self.fixed_local_ratio
            else:
                local_ratio = trial.suggest_float("local_ratio", 1e-3, 1.0, log=True)
        else:
            local_ratio = 1.0  # Not used for global, but needed for config dict

        scaler_config = {
            "method": scaler_method, "scope": scaler_scope,
            "local_ratio": local_ratio, "q_min": 0.25, "q_max": 0.75
        }

        # Augmentation hyperparameters
        if self.fixed_noise_type is not None:
            noise_type = self.fixed_noise_type
        else:
            noise_type = trial.suggest_categorical("noise_type", ["none", "time", "freq"])

        if noise_type == "none":
            sigma = 0.0
        else:
            if self.fixed_aug_sigma is not None:
                sigma = self.fixed_aug_sigma
            else:
                # Log-scale search for sigma (small values matter more)
                sigma = trial.suggest_float("aug_sigma", 1e-3, 0.5, log=True)
        aug_config = {"noise_type": noise_type, "sigma": sigma}

        # Exact-duplicate trials (common with categorical-heavy spaces and
        # shared startup trials) are answered from a memo without recompute.
        memo_key = (lookback, scaler_config["scope"], scaler_config["method"],
                    round(scaler_config["local_ratio"], 12),
                    aug_config["noise_type"], round(aug_config["sigma"], 12))

        if self.n_folds == 1:
            # Single split evaluation
            if memo_key in self._memo:
                mse_per_alpha = self._memo[memo_key]
            else:
                mse_per_alpha = self._evaluate_fold(
                    self.split_idx_1, self.split_idx_2, scaler_config, aug_config, lookback)
                self._memo[memo_key] = mse_per_alpha
            if mse_per_alpha is None:
                raise optuna.TrialPruned()
        else:
            # Expanding window k-fold CV
            if memo_key in self._memo:
                fold_mses_tensor = self._memo[memo_key]
            else:
                fold_mses = []
                for fold_idx in range(self.n_folds):
                    train_end = self.fold_boundaries[fold_idx]
                    val_end = self.fold_boundaries[fold_idx + 1]
                    mse = self._evaluate_fold(
                        train_end, val_end, scaler_config, aug_config, lookback)
                    if mse is not None:
                        fold_mses.append(mse)
                # Stack fold MSEs: (n_folds, K)
                fold_mses_tensor = torch.stack(fold_mses) if fold_mses else None
                self._memo[memo_key] = fold_mses_tensor

            if fold_mses_tensor is None:
                raise optuna.TrialPruned()

            fold_mses = fold_mses_tensor  # (n_valid_folds, K)
            mean_mse = fold_mses_tensor.mean(dim=0)  # (K,)

            # Apply fold variance regularization if lambda > 0
            if self.fold_reg_lambda > 0 and len(fold_mses) > 1:
                std_mse = fold_mses_tensor.std(dim=0)  # (K,)
                mse_per_alpha = mean_mse + self.fold_reg_lambda * std_mse
                # Log fold statistics for analysis
                best_mean_idx = mean_mse.argmin()
                trial.set_user_attr("fold_mean", mean_mse[best_mean_idx].item())
                trial.set_user_attr("fold_std", std_mse[best_mean_idx].item())
            else:
                mse_per_alpha = mean_mse

        # Find best alpha from (averaged) MSE
        mse_per_alpha = mse_per_alpha.squeeze()
        best_val_mse, best_idx = torch.min(mse_per_alpha, dim=0)

        trial.set_user_attr("best_alpha", self.alphas[best_idx].item())
        trial.set_user_attr("mse_per_alpha", mse_per_alpha.tolist())
        return best_val_mse.item()

    def refit_test(self, best_params, best_alpha_val, use_train_val=False):
        """
        Returns DICTIONARY containing raw predictions/targets for full test set.
        """
        # Get hyperparameters: use fixed values if set, otherwise from best_params
        lookback = best_params.get("lookback", self.min_lookback)

        # Scaler config: use fixed values if set, otherwise from best_params
        if self.scaler_method == "search":
            scaler_method = best_params.get("scaler_method", "mean")
        else:
            scaler_method = self.scaler_method

        if self.scaler_scope == "search":
            scaler_scope = best_params.get("scaler_scope", "local")
        else:
            scaler_scope = self.scaler_scope

        if self.fixed_local_ratio is not None:
            local_ratio = self.fixed_local_ratio
        else:
            local_ratio = best_params.get("local_ratio", 1.0)

        scaler_config = {
            "method": scaler_method, "scope": scaler_scope,
            "local_ratio": local_ratio, "q_min": 0.25, "q_max": 0.75
        }

        # Augmentation config: use fixed values if set, otherwise from best_params
        if self.fixed_noise_type is not None:
            noise_type = self.fixed_noise_type
        else:
            noise_type = best_params.get("noise_type", "none")

        if noise_type == "none":
            sigma = 0.0
        else:
            if self.fixed_aug_sigma is not None:
                sigma = self.fixed_aug_sigma
            else:
                sigma = best_params.get("aug_sigma", 0.0)
        aug_config = {"noise_type": noise_type, "sigma": sigma}

        # Use Train+Val (split_idx_2)
        start = self.split_ends[1] if use_train_val else self.split_starts[1]
        end = self.split_ends[1]
        X_train_w, Y_train, _, _, X_test_w, Y_test, scalers = \
            get_prepared_data(self.data, lookback, self.horizon,
                              start, end, scaler_config, include_val=False)

        S = self.n_series
        N_test = X_test_w.shape[1] if X_test_w.dim() == 3 else X_test_w.shape[0]

        # Y windows are contiguous, so this reshape is a view; X windows stay
        # 3D and are consumed chunk-wise by the solver without flattening.
        Y_test = Y_test.reshape(-1, Y_test.shape[-1])

        single_alpha = torch.tensor([best_alpha_val], device=self.device)

        # Solve with scaler
        Theta = self.solver.solve(
            X_train_w, Y_train, single_alpha,
            scaler=scalers['test'],
            aug_config=aug_config,
        )

        # Predict with scaler
        Y_pred = self.solver.predict(X_test_w, Theta, scaler=scalers['test'])
        Y_pred = Y_pred.reshape_as(Y_test)  # (S*N_test, H) or (N_test, H)

        # For GlobalScaler, apply inv_transform to get back to original scale
        if isinstance(scalers['test'], GlobalScaler):
            Y_pred = scalers['test'].inv_transform(Y_pred)
        Y_pred = Y_pred.to(torch.float32)

        if self.pool_series and self.is_batched:
            # Reshape to per-series: (S*N_test, H) -> (S, N_test, H)
            H = Y_pred.shape[-1]
            Y_pred = Y_pred.reshape(S, N_test, H)
            Y_test_per = Y_test.reshape(S, N_test, H)
            per_series_mse = ((Y_pred - Y_test_per)**2).mean(dim=(-2, -1)).cpu()  # (S,)
            return {
                'test_mse': per_series_mse.mean().item(),
                'per_series_mse': per_series_mse,  # (S,)
                'raw_preds': Y_pred.cpu(),  # (S, N_test, H)
            }

        # Calculate simple mean for logging
        mse = ((Y_pred - Y_test)**2).mean().item()
        return {
            'test_mse': mse,
            'raw_preds': Y_pred.cpu(),
        }

class MultiOutputGlobalSeriesObjectiveWrapper:
    def __init__(self, data, lookback, horizons, alphas, device, use_local_norm=True):
        """
        Global ridge regression wrapper for multi-output forecasting.

        Args:
            data: (T, S) time series data
            lookback: context length
            horizons: list of forecast horizons
            alphas: regularization strength (scalar)
            device: torch device
            use_local_norm: if True, use LocalNormScaler (default True for reference compatibility)
        """
        self.data = data.to(device)
        self.lookback = lookback
        self.horizon = horizons
        self.alphas = alphas
        self.device = device
        self.use_local_norm = use_local_norm
        self.solver = RidgeSolver(device)

    def refit_test(self, n_train, n_test):
        """
        Train on training data and predict on test data.

        Returns:
            dict with 'raw_preds': (S, N_test, H) predictions
        """
        T, S = self.data.shape
        L, H = self.lookback, len(self.horizon)
        max_horizon = max(self.horizon)

        # Create scaler based on setting
        if self.use_local_norm:
            scaler = LocalNormScaler(StandardStrategy(), L, L)
        else:
            scaler = None  # No scaler, use intercept

        # Create training windows (device-side views; the solver consumes the
        # 3D windows chunk-wise without materializing a flattened copy)
        train_wins = self.data[:n_train].transpose(0, 1).unfold(1, L + max_horizon, 1)
        # train_wins shape: (S, N_train_windows, L + max_horizon)

        X_tr = train_wins[:, :, :L]  # (S, N, L)
        Y_tr = train_wins[:, :, L:L+H]  # (S, N, H)

        # Solve ridge regression using unified solver
        single_alpha = torch.tensor([self.alphas], device=self.device)
        Theta = self.solver.solve(X_tr, Y_tr, single_alpha, scaler=scaler)

        # Create test windows: start from test_start - L to include context
        test_start = T - n_test
        test_wins = self.data[test_start - L:].transpose(0, 1).unfold(1, L + max_horizon, 1)
        # test_wins shape: (S, N_test_windows, L + max_horizon)

        X_te = test_wins[:, :, :L]  # (S, N_test, L)
        N_test = test_wins.shape[1]

        # Predict using unified solver
        Y_pred = self.solver.predict(X_te, Theta, scaler=scaler)

        # Reshape to (S, N_test, H)
        Y_pred = Y_pred.squeeze(0).reshape(S, N_test, H)
        Y_pred = Y_pred.detach().to(torch.float32).cpu()

        return {'raw_preds': Y_pred}

# ==========================================
# 6. Main Loop
# ==========================================
class TrialLogWriter:
    """Appends one CSV row per finished Optuna trial (used by scripts/bench_parity.py)."""

    FIELDS = ["study", "trial", "state", "value", "duration_s", "best_alpha",
              "params", "mse_per_alpha"]

    def __init__(self, path):
        self.path = path
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, "w", newline="") as f:
            csv.writer(f).writerow(self.FIELDS)

    def callback(self, study_key):
        def _cb(study, trial):
            row = [
                study_key,
                trial.number,
                trial.state.name,
                "" if trial.value is None else repr(trial.value),
                "" if trial.duration is None else f"{trial.duration.total_seconds():.4f}",
                repr(trial.user_attrs["best_alpha"]) if "best_alpha" in trial.user_attrs else "",
                json.dumps(trial.params, sort_keys=True),
                json.dumps(trial.user_attrs.get("mse_per_alpha", [])),
            ]
            with open(self.path, "a", newline="") as f:
                csv.writer(f).writerow(row)
        return _cb


def search_local_models(data, series_names, lookbacks, horizons, alphas, n_trials, device,
                        horizon_group_size=1, series_group_size=1,
                        search_train_ratio=0.4, search_test_ratio=0.2,
                        split_starts=None, split_ends=None, n_folds=1, fold_reg_lambda=0.0,
                        scaler_scope="search", scaler_method="search",
                        fixed_local_ratio=None, fixed_noise_type=None, fixed_aug_sigma=None,
                        pool_series=False, seed=None, precision="fp64",
                        horizon_group_subset=None, trial_log=None):
    best_results = []
    raw_results = {}  # Storage for alignment
    T, S = data.shape

    trial_log_writer = TrialLogWriter(trial_log) if trial_log else None

    # Handle special value: -1 means all series
    if series_group_size <= 0:
        series_group_size = S

    # Valid lookback calculation
    valid_lookbacks = [int(lb) for lb in lookbacks if lb < int(T * 0.3) - max(horizons)]
    print(f"Valid Lookbacks: {valid_lookbacks}")

    # Timing statistics
    search_times = []
    total_start_time = time.time()

    # Iterate over series groups
    for sg_idx in range(0, S, series_group_size):
        series_group = list(range(sg_idx, min(sg_idx + series_group_size, S)))
        series_names_in_group = [series_names[i] for i in series_group]
        group_display = (series_names_in_group[0] if len(series_group) == 1
                         else f"[{series_names_in_group[0]}..{series_names_in_group[-1]}]")
        print(f"\n[Series Group: {group_display}]")

        for group_idx, h_idx in enumerate(range(0, len(horizons), horizon_group_size)):
            if horizon_group_subset is not None and group_idx not in horizon_group_subset:
                continue
            horizon_group = horizons[h_idx:h_idx + horizon_group_size]

            # HP Search: use series group (grouped objective for robust HP selection)
            sampler = optuna.samplers.TPESampler(seed=seed) if seed is not None else None
            study = optuna.create_study(direction="minimize", sampler=sampler)
            wrap = SingleObjectiveWrapper(
                data, series_group if len(series_group) > 1 else series_group[0],
                valid_lookbacks, horizon_group, alphas, device,
                train_ratio=search_train_ratio, test_ratio=search_test_ratio,
                split_starts=split_starts, split_ends=split_ends,
                n_folds=n_folds, fold_reg_lambda=fold_reg_lambda,
                scaler_scope=scaler_scope, scaler_method=scaler_method,
                fixed_local_ratio=fixed_local_ratio, fixed_noise_type=fixed_noise_type,
                fixed_aug_sigma=fixed_aug_sigma,
                pool_series=pool_series, precision=precision)

            # Time the optimization
            callbacks = ([trial_log_writer.callback(f"sg{sg_idx}_hg{group_idx}")]
                         if trial_log_writer else None)
            search_start = time.time()
            study.optimize(wrap, n_trials=n_trials, callbacks=callbacks)
            search_elapsed = time.time() - search_start
            search_times.append(search_elapsed)

            # Extract best results from study
            params = study.best_trial.params
            noise_type = params.get('noise_type', 'none')
            aug_sigma = params.get('aug_sigma', 0.0) if noise_type != 'none' else 0.0
            local_ratio = params.get('local_ratio', fixed_local_ratio if fixed_local_ratio else 1.0)
            best = {
                'lookback': params['lookback'],
                'val_mse': study.best_value,
                'params': params,
                'best_alpha': study.best_trial.user_attrs['best_alpha'],
                'scaler_method': params.get('scaler_method', 'mean'),
                'scaler_scope': params.get('scaler_scope', 'local'),
                'noise_type': noise_type,
                'aug_sigma': aug_sigma,
                'local_ratio': local_ratio
            }

            if pool_series and len(series_group) > 1:
                # REFIT: single pooled model for entire series group
                metrics = wrap.refit_test(best['params'], best['best_alpha'])
                # metrics['raw_preds'] shape: (S_group, N_test, H)
                for i, s_idx in enumerate(series_group):
                    raw_results[(s_idx, horizon_group[0])] = metrics['raw_preds'][i]
                    rec = best.copy()
                    rec.update({
                        'series': series_names[s_idx],
                        'horizon': horizon_group,
                        'test_mse': metrics['per_series_mse'][i].item()
                    })
                    best_results.append(rec)
            else:
                # REFIT: train individual model per series with shared HPs
                for s_idx in series_group:
                    refitter = SingleObjectiveWrapper(
                        data, s_idx, valid_lookbacks, horizon_group, alphas, device,
                        train_ratio=search_train_ratio, test_ratio=search_test_ratio,
                        split_starts=split_starts, split_ends=split_ends,
                        scaler_scope=scaler_scope, scaler_method=scaler_method,
                        fixed_local_ratio=fixed_local_ratio, fixed_noise_type=fixed_noise_type,
                        fixed_aug_sigma=fixed_aug_sigma, precision=precision)
                    metrics = refitter.refit_test(best['params'], best['best_alpha'])

                    # Save Raw Vectors
                    raw_results[(s_idx, horizon_group[0])] = metrics['raw_preds']

                    # Save Summary
                    rec = best.copy()
                    rec.update({
                        'series': series_names[s_idx],
                        'horizon': horizon_group,
                        'test_mse': metrics['test_mse']
                    })
                    best_results.append(rec)

            display_horizon = (horizon_group if len(horizon_group) == 1
                               else f"[{horizon_group[0]}-{horizon_group[-1]}]")
            aug_str = f"aug={noise_type}" + (f"({aug_sigma:.3f})" if noise_type != 'none' else "")
            print(f"  > H={display_horizon}: L={best['lookback']}, "
                  f"α={best['best_alpha']:.2e}, scaler={best['scaler_scope']}, "
                  f"norm={best['scaler_method']}, {aug_str}. "
                  f"Val={best['val_mse']:.4f}, Time={search_elapsed:.2f}s")

        # Release cached allocator blocks once per series group (per-solve
        # empty_cache calls were removed — they forced a device sync each fit)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Print timing summary
    total_elapsed = time.time() - total_start_time
    avg_search_time = sum(search_times) / len(search_times) if search_times else 0
    print(f"\n[Timing Summary]")
    print(f"  Total searches: {len(search_times)}")
    print(f"  Average search time: {avg_search_time:.2f}s")
    print(f"  Total time: {total_elapsed:.2f}s ({total_elapsed/60:.1f}min)")

    return best_results, raw_results

def search_global_baseline(data, lookback, alphas, device, split_starts=None, split_ends=None,
                           use_local_norm=True):
    """
    Run global ridge regression baseline for multiple forecast horizons.

    Args:
        data: (T, S) time series data
        lookback: context length
        alphas: regularization strength
        device: torch device
        split_starts: list of split start indices [train_start, val_start, test_start]
        split_ends: list of split end indices [train_end, val_end, test_end]
        use_local_norm: if True, use LocalNormScaler (default True for reference compatibility)

    Returns:
        dict mapping cutoff -> (S, N_test, cutoff) predictions
    """
    raw_results = {}
    horizons_of_interest = [96, 192, 336, 720]
    if split_starts is not None and split_ends is not None:
        n_train = split_starts[1]
        n_val = split_ends[1] - n_train
        n_test = split_ends[2] - split_starts[2]
    else:
        n_train = int(data.shape[0] * 0.7)
        n_test = int(data.shape[0] * 0.2)
        n_val = data.shape[0] - n_train - n_test

    for cutoff in horizons_of_interest:
        horizons_list = list(range(1, cutoff + 1))
        wrapper = MultiOutputGlobalSeriesObjectiveWrapper(
            data, lookback, horizons_list, alphas, device, use_local_norm=use_local_norm
        )
        results = wrapper.refit_test(n_train=n_train, n_test=n_test)
        raw_results[cutoff] = results['raw_preds']

    return raw_results

# ==========================================
# 7. Alignment & Comparison Logic
# ==========================================
def align_and_compare(data: torch.Tensor, local_raw: torch.Tensor, global_raw: torch.Tensor, L: int, H: int, horizon_group_size):
    """
    local_raw: list of tensors [(S, N_test[1]), (S, N_test[2]), ..., (S, N_test[720])] if we assume the max horizon is H
    global_raw: a stensor of shape (S, N_test[H], H)
    """
    print("\n[Final Alignment & Benchmarking]")
    assert global_raw.shape[-1] == H

    N_test_at_cutoff = global_raw.shape[1]
    global_diff = global_raw - data
    global_mse = (global_diff**2).mean().item()
    global_mae = global_diff.abs().mean().item()

    local_raw = torch.cat([pred[:, :N_test_at_cutoff] for pred in local_raw[:math.ceil(H/horizon_group_size)]], dim=-1)
    local_diff = local_raw - data
    local_mse = (local_diff**2).mean().item()
    local_mae = local_diff.abs().mean().item()
    
    print(f"  > H={H}: Local {local_mse:.4f} vs Global {global_mse:.4f}")
    return dict(mse=local_mse, mae=local_mae), dict(mse=global_mse, mae=global_mae)


def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_csv", type=str, default="data/weather.csv")
        parser.add_argument("--output_dir", type=str, default="results")
        parser.add_argument("--local_horizon_group_size", type=int, default=24)
        parser.add_argument("--local_series_group_size", type=int, default=1,
                            help="Number of series to group for HP search (1 = per-series, -1 = all)")
        parser.add_argument("--instance_norm", action="store_true", default=False,
                            help="Use instance normalization for global baseline (default: False)")
        parser.add_argument("--no_instance_norm", action="store_false", dest="instance_norm",
                            help="Disable instance normalization for global baseline")
        parser.add_argument("--n_folds", type=int, default=1,
                            help="Number of folds for expanding window CV (1 = single split, no CV)")
        parser.add_argument("--fold_reg_lambda", type=float, default=0.0,
                            help="Regularization weight for fold variance (0 = no regularization)")
        parser.add_argument("--n_trials", type=int, default=30)

        # Ablation controls
        parser.add_argument("--scaler_scope", type=str, default="search",
                            choices=["global", "local", "search"],
                            help="Normalization scope: global (dataset-level), local (per-sample), or search both")
        parser.add_argument("--scaler_method", type=str, default="search",
                            choices=["mean", "robust", "search"],
                            help="Normalization method: mean (standard), robust (median/IQR), or search both")
        parser.add_argument("--fixed_local_ratio", type=float, default=None,
                            help="Fix local_ratio instead of searching (only relevant for local scope)")
        parser.add_argument("--fixed_noise_type", type=str, default=None,
                            choices=["none", "time", "freq"],
                            help="Fix augmentation type instead of searching")
        parser.add_argument("--fixed_aug_sigma", type=float, default=None,
                            help="Fix augmentation sigma instead of searching")
        parser.add_argument("--pool_series", action="store_true", default=False,
                            help="Pool training data across series in a group (single model per group)")
        parser.add_argument("--fixed_lookback", type=int, default=None,
                            help="Fix lookback to a single value instead of searching")
        parser.add_argument("--seed", type=int, default=None,
                            help="Seed for numpy/torch and Optuna TPE sampler (for multi-seed runs)")

        # Performance controls
        parser.add_argument("--precision", type=str, default="fp64",
                            choices=["fp64", "mixed", "fp32"],
                            help="Solver precision for the local-model search: fp64 (exact, "
                                 "default), mixed (fp32 Gram matmuls, fp64 accumulate/solve), "
                                 "or fp32. The global baseline always runs fp64.")
        parser.add_argument("--tf32", action="store_true", default=False,
                            help="Allow TF32 tensor-core matmuls (only affects fp32 compute)")
        parser.add_argument("--cache_gb", type=float, default=4.0,
                            help="GramCache budget in GiB (prefix-Gram checkpoints shared "
                                 "across trials, folds, and horizon-group studies)")
        parser.add_argument("--no_cache", action="store_true", default=False,
                            help="Disable the GramCache (Grams are rebuilt every evaluation)")
        parser.add_argument("--cache_verify", type=float, default=0.0,
                            help="Fraction of cache hits to re-verify from scratch (debug)")

        # Debug/benchmark controls (scripts/bench_parity.py)
        parser.add_argument("--horizon_subset", type=str, default=None,
                            help="Comma-separated horizon-group indices to search (debug/benchmark; "
                                 "skips the global baseline and alignment stages)")
        parser.add_argument("--trial_log", type=str, default=None,
                            help="Path to a CSV logging every finished Optuna trial")
        parser.add_argument("--profile", action="store_true", default=False,
                            help="Record coarse per-stage timings (adds CUDA syncs; benchmark only)")
        args = parser.parse_args()

        if args.seed is not None:
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)

        os.makedirs(args.output_dir, exist_ok=True)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if args.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        RUN_CONFIG["aug_base_seed"] = args.seed if args.seed is not None else 0
        GRAM_CACHE.max_bytes = int(args.cache_gb * (1 << 30))
        GRAM_CACHE.enabled = not args.no_cache
        GRAM_CACHE.verify_frac = args.cache_verify

        PROFILER.enabled = args.profile
        horizon_subset = (set(int(i) for i in args.horizon_subset.split(","))
                          if args.horizon_subset else None)

        # Config
        if args.fixed_lookback is not None:
            LOOKBACKS = np.array([args.fixed_lookback], dtype=int)
        else:
            LOOKBACKS = np.logspace(5, 11, 19, base=2, dtype=int)
        HORIZONS = list(range(1, 721))
        HORIZON_GROUP_SIZE = args.local_horizon_group_size
        SERIES_GROUP_SIZE = args.local_series_group_size
        ALPHAS = torch.logspace(-6, 4, 21, device=DEVICE)
        N_TRIALS = args.n_trials

        # Load
        df = pd.read_csv(args.input_csv)
        scaler = StandardScaler()
        if 'ETT' not in args.input_csv:
            n_train = int(len(df) * 0.7)
            n_test = int(len(df) * 0.2)
            n_val = len(df) - n_train - n_test
            total_len = len(df)
            search_train_ratio = 0.4
            search_test_ratio = 0.2
        else:
            # Match reference: ETT datasets are truncated to exactly 20 months
            # Train: 12 months, Val: 4 months, Test: 4 months
            if 'h' in args.input_csv:
                n_train = 12 * 30 * 24          # 8640
                n_val = 4 * 30 * 24             # 2880
                n_test = 4 * 30 * 24            # 2880
                total_len = 12 * 30 * 24 + 8 * 30 * 24  # 14400 (20 months)
            else:
                n_train = 12 * 30 * 24 * 4      # 34560
                n_val = 4 * 30 * 24 * 4         # 11520
                n_test = 4 * 30 * 24 * 4        # 11520
                total_len = 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4  # 57600 (20 months)
            search_train_ratio = (n_train + n_val) / total_len * 0.5   # use half of the data as validation
            search_test_ratio = n_test / total_len
        split_starts = [0, n_train, n_train + n_val]
        split_ends = [n_train, n_train + n_val, total_len]
        # Truncate data to match reference implementation
        df_values = df.drop(columns=['date']).values[:total_len]
        scaler.fit(df_values[:n_train])
        data = torch.tensor(scaler.transform(df_values), dtype=torch.float32)
        print("Split start: ", split_starts)
        print("Split end: ", split_ends)

        # 1. Local Search (Returns Raw Preds)
        best_df, local_raw_ind = search_local_models(
            data, df.columns[1:], LOOKBACKS, HORIZONS, ALPHAS, n_trials=N_TRIALS, device=DEVICE,
            horizon_group_size=HORIZON_GROUP_SIZE, series_group_size=SERIES_GROUP_SIZE,
            search_train_ratio=search_train_ratio, search_test_ratio=search_test_ratio,
            split_starts=split_starts, split_ends=split_ends,
            n_folds=args.n_folds, fold_reg_lambda=args.fold_reg_lambda,
            scaler_scope=args.scaler_scope, scaler_method=args.scaler_method,
            fixed_local_ratio=args.fixed_local_ratio, fixed_noise_type=args.fixed_noise_type,
            fixed_aug_sigma=args.fixed_aug_sigma,
            pool_series=args.pool_series, seed=args.seed, precision=args.precision,
            horizon_group_subset=horizon_subset, trial_log=args.trial_log
        )
        pd.DataFrame(best_df).to_csv(f"{args.output_dir}/local_results.csv", index=False)

        if args.profile:
            profile = PROFILER.report()
            profile["gram_cache"] = GRAM_CACHE.stats()
            with open(os.path.join(args.output_dir, "profile.json"), "w") as f:
                json.dump(profile, f, indent=2)
            for name, rec in profile.items():
                print(f"  [profile] {name}: {rec}")

        if horizon_subset is not None:
            # Debug/benchmark mode: the remaining stages need every horizon group.
            print(f"--horizon_subset set; skipping global baseline and alignment stages.")
            return

        local_raw = []
        for h_idx in range(0, len(HORIZONS), HORIZON_GROUP_SIZE):
            horizon_group = HORIZONS[h_idx:h_idx + HORIZON_GROUP_SIZE]
            local_raw.append(
                torch.stack(
                    [local_raw_ind[(s_idx, horizon_group[0])] for s_idx in range(data.shape[1])], 
                    dim=0))

        # 2. Global Baseline (Returns Raw Preds)
        global_lookback = 720
        global_raw = search_global_baseline(
            data, global_lookback, 1e-5, DEVICE,
            split_starts=split_starts, split_ends=split_ends,
            use_local_norm=args.instance_norm
        )

        # 3. Align and Compare
        cutoffs = [96, 192, 336, 720]
        final_bench = {}
        test_start = split_starts[2]
        assert test_start > global_lookback
        test_data = data[test_start - global_lookback:]

        for cutoff in [96, 192, 336, 720]:
            test_wins = test_data.transpose(0, 1).unfold(1, global_lookback + cutoff, 1)
            Y_te = test_wins[:, :, global_lookback:global_lookback+cutoff] #.reshape(-1, cutoff)
            local_metrics, global_metrics = align_and_compare(Y_te, local_raw, global_raw[cutoff], global_lookback, cutoff, HORIZON_GROUP_SIZE)
            final_bench[cutoff] = [local_metrics['mse'], global_metrics['mse'], local_metrics['mae'], global_metrics['mae']]
            print(f"Cutoff = {cutoff}: ", [local_metrics['mse'], global_metrics['mse'], local_metrics['mae'], global_metrics['mae']])
        bench_df = pd.DataFrame(final_bench).transpose()
        bench_df.to_csv(f"{args.output_dir}/benchmark_comparison.csv", header=['Local MSE', 'Global MSE', 'Local MAE', 'Global MAE'])

        # save the first predictions for visualization
        MAX_HORIZONS = max(HORIZONS)
        VIS_IDX = 336
        local_pred = torch.cat([local_p[:, VIS_IDX, :] for local_p in local_raw], dim=-1).numpy()   # (S, H)
        global_pred = global_raw[MAX_HORIZONS][:, VIS_IDX].numpy()   # (S, H)
        gt = data[test_start+VIS_IDX:test_start+MAX_HORIZONS+VIS_IDX, :].transpose(0, 1)
        np.save(os.path.join(args.output_dir, "predictions.npy"), dict(gt=gt, local_pred=local_pred, global_pred=global_pred))

        # Plot
        S, H = gt.shape
        fig, axes = plt.subplots(S, 1, figsize=(12, 3*S), sharex=True)
        if S == 1: axes = [axes]

        for i in range(S):
            ax = axes[i]
            ax.plot(gt[i], label='Ground Truth', color='black', linewidth=1.5, alpha=0.7)
            ax.plot(local_pred[i], label='Local (Ours)', color='blue', linestyle='--', alpha=0.8)
            ax.plot(global_pred[i], label='Global (Baseline)', color='red', linestyle=':', alpha=0.8)

            ax.set_ylabel(f'Series {i}')
            ax.set_title(f'Forecast Comparison: Series {i}')
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(loc='upper right')

        plt.xlabel('Horizon Step (1-720)')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "forecast_comparison_all_horizons.png"))


        # the predition at H=336
        VIS_HORIZON = 336
        gt = data[test_start + VIS_HORIZON-1:, :].transpose(0, 1) # (S, N)
        n_test = gt.shape[1]
        global_pred = global_raw[VIS_HORIZON][:, :, VIS_HORIZON - 1].numpy()   # (S, N)
        assert global_pred.shape[1] == n_test
        local_pred = torch.cat([local_p[:, :n_test, :] for local_p in local_raw[:math.ceil(VIS_HORIZON/HORIZON_GROUP_SIZE)]], dim=-1)
        local_pred = local_pred[:, :, VIS_HORIZON - 1].numpy()   # (S, H)
        # Plot
        S, H = gt.shape
        fig, axes = plt.subplots(S, 1, figsize=(12, 3*S), sharex=True)
        if S == 1: axes = [axes]

        for i in range(S):
            ax = axes[i]
            ax.plot(gt[i], label='Ground Truth', color='black', linewidth=1.5, alpha=0.7)
            ax.plot(local_pred[i], label='Local (Ours)', color='blue', linestyle='--', alpha=0.8)
            ax.plot(global_pred[i], label='Global (Baseline)', color='red', linestyle=':', alpha=0.8)

            ax.set_ylabel(f'Series {i}')
            ax.set_title(f'Forecast Comparison: Series {i}')
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(loc='upper right')

        plt.xlabel('Time')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"forecast_comparison_horizon{VIS_HORIZON}.png"))


if __name__ == "__main__":
    main()
