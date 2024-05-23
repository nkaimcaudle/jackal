from functools import partial
from typing import Callable
from collections import namedtuple
import jax
import jax.numpy as jnp
import jax.random as jrnd


def evolve_heston(carry: tuple, X: tuple, covL: jnp.ndarray) -> tuple:
    perf_prev, v_prev, t_prev = carry
    Z, t_curr, r, q, kappa, theta, sigma = X
    dt = t_curr - t_prev
    sdt = jnp.sqrt(dt)

    W = covL @ Z
    Z_var_process = W[1::2]
    Z_spot_process = W[0::2]
    v_curr = jnp.maximum(
        v_prev
        + kappa * (theta - v_prev) * dt
        + sigma * jnp.sqrt(v_prev) * sdt * Z_var_process,
        0.0,
    )
    drift = dt * ((r - q) - 0.5 * v_curr)
    move = drift + jnp.sqrt(v_curr) * Z_spot_process * sdt

    perf_curr = perf_prev + move
    return (perf_curr, v_curr, t_curr), (move, v_curr)


def mcpaths_single_heston(
    key: jrnd.PRNGKey,
    i: int,
    JTs: jnp.ndarray,
    JT0: float,
    Nassets: int,
    fwd_r: jnp.ndarray,
    fwd_q: jnp.ndarray,
    covL: jnp.ndarray,
    S0: jnp.ndarray,
    v0: jnp.ndarray,
    kappa: jnp.ndarray,
    theta: jnp.ndarray,
    sigma: jnp.ndarray,
) -> jnp.ndarray:
    _key = jrnd.fold_in(key, i)
    Z = jrnd.normal(_key, shape=(JTs.size, 2 * Nassets))
    init = (jnp.zeros(Nassets), v0, JT0)
    Xs = (Z, JTs, fwd_r, fwd_q, kappa, theta, sigma)
    _evolve_heston = partial(evolve_heston, covL=covL)
    _, (paths, vs) = jax.lax.scan(_evolve_heston, init, Xs)
    paths = jnp.vstack((jnp.zeros(Nassets), paths)).T
    return S0[:, None] * jnp.exp(jnp.cumsum(paths, axis=-1))


def mcpaths_heston(
    key: jrnd.PRNGKey,
    NRuns: int,
    NIter: int,
    JTs: jnp.ndarray,
    JT0: float,
    Nassets: int,
    fwd_r: jnp.ndarray,
    fwd_q: jnp.ndarray,
    covL: jnp.ndarray,
    S0: jnp.ndarray,
    v0: jnp.ndarray,
    kappa: jnp.ndarray,
    theta: jnp.ndarray,
    sigma: jnp.ndarray,
) -> jnp.ndarray:
    paths_func = partial(
        mcpaths_single_heston,
        key=key,
        JTs=JTs,
        JT0=JT0,
        Nassets=Nassets,
        fwd_r=fwd_r,
        fwd_q=fwd_q,
        covL=covL,
        S0=S0,
        v0=v0,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
    )
    paths = jax.vmap(paths_func)(i=jnp.arange(NRuns * NIter))
    return paths.reshape(NRuns, NIter, Nassets, -1)


def a():
    asof = 0.0
    JTs = jnp.linspace(0.01, 1, 64)
    pegs = jnp.hstack((0.0, JTs))
    paths = mcpaths_heston(
        jrnd.PRNGKey(0),
        NRuns=16,
        NIter=10_000,
        JTs=JTs,
        JT0=0.0,
        Nassets=1,
        fwd_r=jnp.full(64, 0.05),
        fwd_q=jnp.zeros((64, 1)),
        covL=jnp.eye(2),
        S0=jnp.array(
            [
                100.0,
            ]
        ),
        v0=jnp.array(
            [
                0.2**2,
            ]
        ),
        kappa=jnp.zeros((64, 1)),
        theta=jnp.full((64, 1), 0.0**2),
        sigma=jnp.zeros((64, 1)),
    )
    return PricingInfo(asof, paths, pegs, ("BABA",))


def payoff_call(ul: str, expiry: float, strike: float) -> Callable:
    def fn1(pi: PricingInfo) -> float:
        ul_idx = pi.uls.index(ul)
        JT_idx = jnp.argwhere(pi.pegs == expiry).item()
        ST = pi.paths[ul_idx, JT_idx]
        return jnp.exp(-0.05 * (expiry - pi.asof)) * jnp.maximum(ST - strike, 0.0)

    return fn1


def payoff_put(ul: str, expiry: float, strike: float) -> Callable:
    def fn1(pi: PricingInfo) -> float:
        ul_idx = pi.uls.index(ul)
        JT_idx = jnp.argwhere(pi.pegs == expiry).item()
        ST = pi.paths[ul_idx, JT_idx]
        return jnp.exp(-0.05 * (expiry - pi.asof)) * jnp.maximum(strike - ST, 0.0)

    return fn1


def payoff_fwd(ul: str, expiry: float, strike: float) -> Callable:
    def fn1(pi: PricingInfo) -> float:
        ul_idx = pi.uls.index(ul)
        JT_idx = jnp.argwhere(pi.pegs == expiry).item()
        ST = pi.paths[ul_idx, JT_idx]
        return jnp.exp(-0.05 * (expiry - pi.asof)) * (ST - strike)

    return fn1


PricingInfo = namedtuple("PricingInfo", ["asof", "paths", "pegs", "uls"])
if __name__ == "__main__":
    pi = a()
    opt_call = payoff_call("BABA", 1.0, 115.0)
    opt_put = payoff_put("BABA", 1.0, 115.0)
    opt_fwd = payoff_fwd("BABA", 1.0, 115.0)

    def px_call(path) -> float:
        _pi = PricingInfo(pi.asof, path, pi.pegs, pi.uls)
        return opt_call(_pi)

    def px_put(path) -> float:
        _pi = PricingInfo(pi.asof, path, pi.pegs, pi.uls)
        return opt_put(_pi)

    def px_fwd(path) -> float:
        _pi = PricingInfo(pi.asof, path, pi.pegs, pi.uls)
        return opt_fwd(_pi)

    a = jax.vmap(jax.vmap(px_call))(pi.paths).mean()
    b = jax.vmap(jax.vmap(px_put))(pi.paths).mean()
    c = jax.vmap(jax.vmap(px_fwd))(pi.paths).mean()
    print(a, b, c)
