from __future__ import annotations
from typing import Optional, Tuple, Dict
import numpy as np 

Array = np.ndarray

def _rng_from_seed(seed: Optional[int | np.random.Generator]) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)

def simulate_gbm_averages(
    S0: float, ##Initial price
    r: float, ## risk free rate continuously copounded
    sigma: float, #annualized vol
    T: float, #maturity in years
    N: int, #number of time steps
    M: int, #number of paths
    *,
    seed: Optional[int | np.random.Generator] = None,
    dtype: np.dtype = np.float64,
    batch_size: Optional[int] = None,
    return_terminal: bool = False,
    return_paths: bool = False,
)   -> Tuple[Array, Dict[str, float] | Tuple[Array, Dict[str, float]] | Tuple[Array, Dict[str, float]]]:

    ##error stuff
    if N <= 0 or M <= 0:
        raise ValueError("N and M must be positive")
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")
    if T <= 0:
        raise ValueError("T must be positive")

    rng = _rng_from_seed(seed)
    dt = T / N
    drift = (r - 0.5 * sigma * sigma) * dt
    vol = sigma * np.sqrt(dt)

    ##deciding batching
    if batch_size is None:
        batch_size = min(M, max(10_000, 1_000_000 // max(1, N)))

    ##outputs
    Sbar = np.empty(M, dtype=dtype)
    S_T = np.empty(M, dtype=dtype) if return_terminal else None
    paths_full = np.empty((M, N+1), dtype=dtype) if return_paths else None

    done = 0
    while done < M:
        b = min(batch_size, M - done)
        S = np.full(b, S0, dtype=dtype)
        sum_S = np.zeros(b, dtype=dtype)

        if return_paths:
            paths_full[done:done + b, 0] = S

        for k in range(1, N+1):
            Z = rng.standard_normal(b).astype(dtype, copy=False)
            np.multiply(S, np.exp(drift + vol * Z, dtype=dtype), out=S)
            sum_S += S
            if return_paths:
                paths_full[done:done + b, k] = S0

        Sbar[done:done + b] = sum_S / N 
        if return_terminal:
            S_T[done:done + b] = S

        done += b 
    
    meta = {"S0": float(S0), "r": float(r), "sigma": float(sigma), "T": float(T), "N": int(N), "M": int(M), "dt": float(dt)}

    if return_paths and return_terminal:
        return Sbar, S_T, paths_full, meta
    if return_paths:
        return Sbar, paths_full, meta
    if return_terminal:
        return Sbar, S_T, meta
    return Sbar, meta


Sbar, S_T, meta = simulate_gbm_averages(
    S0 = 100, r=0.02, sigma=0.2, T=1.0, N=252, M=20_000, seed=123, return_terminal=true
)
print("Meta:", meta)
print("E[bar S] ~", float(Sbar.mean()))
print("E[S_T] ~", float(S_T.mean()))
