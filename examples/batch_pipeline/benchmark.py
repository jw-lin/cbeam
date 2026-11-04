"""Timing helpers for the batch-vs-one-at-a-time propagation comparison
demonstrated in examples/Batched.ipynb.

Pure processing -- no matplotlib, no specula. Propagation only needs the
characterized ChainPropagator (``prop12``), never the influence-function /
aberration machinery, so this module carries none of those dependencies
either.
"""
from __future__ import annotations

import time
import numpy as np

from .constants import backend


def _block_until_ready(x):
    """No-op for numpy; on JAX, wait for the (possibly async-dispatched)
    device computation so the timer measures real device time, not just how
    long it took to queue the work."""
    ready = getattr(x, "block_until_ready", None)
    if ready is not None:
        ready()
    return x


def make_random_modal_batch(n_fields, n_modes, seed=0):
    """Random unit-L2-norm complex modal-coefficient vectors.

    Plain numpy regardless of backend -- Propagator.propagate() accepts a
    host array and moves it to the active backend itself.

    Returns
    -------
    np.ndarray, shape (n_fields, n_modes), complex128
    """
    rng = np.random.default_rng(seed)
    u0 = rng.standard_normal((n_fields, n_modes)) + 1j * rng.standard_normal((n_fields, n_modes))
    u0 /= np.linalg.norm(u0, axis=1, keepdims=True)
    return u0


def time_one_at_a_time(prop12, u0_batch):
    """Propagate each row of ``u0_batch`` with its own ``propagate()`` call.

    Returns elapsed wall-clock seconds for the whole loop.
    """
    t0 = time.perf_counter()
    for row in u0_batch:
        uf = prop12.propagate(row)[2]
        _block_until_ready(uf)
    return time.perf_counter() - t0


def time_batched(prop12, u0_batch):
    """Propagate the whole batch in a single ``propagate()`` call.

    Returns elapsed wall-clock seconds.
    """
    t0 = time.perf_counter()
    uf = prop12.propagate(u0_batch)[2]
    _block_until_ready(uf)
    return time.perf_counter() - t0


def benchmark_propagation(prop12, batch_sizes, seed=0, warmup=True):
    """
    Time propagating random modal-coefficient batches through ``prop12``,
    both one field at a time and as a single batched call, for each ``n`` in
    ``batch_sizes``.

    warmup : bool, optional (default True)
        Run (and discard the time of) one call of every shape used below
        before it is timed. Needed on the JAX backend, where each distinct
        input shape triggers its own trace + XLA compile the first time it
        is seen -- without a warmup call, the *first* batch size measured
        would include that one-off compile cost, made to look like a
        per-call cost it isn't. Harmless (a little redundant work) on the
        numpy backend, which has nothing to compile.

    Returns
    -------
    dict : {"backend": "numpy" | "jax", "n_modes": int, "results": [
        {"n_fields": n, "loop_seconds": ..., "batch_seconds": ...}, ...
    ]}
    """
    n_modes = prop12.Nmax
    if warmup:
        # The one-at-a-time loop always propagates single-field (n_modes,)
        # arrays, whatever n is -- compile that shape once, up front.
        time_one_at_a_time(prop12, make_random_modal_batch(1, n_modes, seed=2**31 - 1))

    results = []
    for n in batch_sizes:
        u0_batch = make_random_modal_batch(n, n_modes, seed=seed)
        if warmup:
            time_batched(prop12, u0_batch)  # compiles this batch shape; discarded
        loop_seconds = time_one_at_a_time(prop12, u0_batch)
        batch_seconds = time_batched(prop12, u0_batch)
        results.append({
            "n_fields": n,
            "loop_seconds": loop_seconds,
            "batch_seconds": batch_seconds,
        })

    return {"backend": backend, "n_modes": int(n_modes), "results": results}


if __name__ == "__main__":
    # CLI entry point used by examples/Batched.ipynb to run this benchmark in
    # a subprocess under CBEAM_BACKEND=jax: cbeam.backend picks numpy vs. jax
    # once, at import time (see tests/conftest.py for the same constraint in
    # the test suite), so a notebook already running under one backend
    # cannot exercise the other in the same process.
    import json
    import sys

    from .config import get_simulation_parameters
    from .engine import build_and_characterize_lantern

    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: python -m batch_pipeline.benchmark "
            "<nrings> <wavelength_um> <batch_sizes_csv>"
        )
    nrings = int(sys.argv[1])
    wavelength_um = float(sys.argv[2])
    batch_sizes = [int(n) for n in sys.argv[3].split(",")]

    p = get_simulation_parameters(nrings, wavelength_um)
    # reuse_cache=True: falls back to full characterization automatically if
    # no cached run matches (nrings, wavelength) yet -- slow the first time,
    # fast on every later call.
    prop12 = build_and_characterize_lantern(p, reuse_cache=True)
    result = benchmark_propagation(prop12, batch_sizes)
    # Last line of stdout only: the notebook ignores everything printed
    # before it (characterize()/load() progress messages, jax/diffrax's own
    # startup banner, ...) and parses just this line as JSON.
    print(json.dumps(result))
