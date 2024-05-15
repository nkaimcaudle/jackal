import jax
import jax.numpy as jnp
import jax.random as jrnd


def evolve_heston(carry, X):
    perf_prev, v_prev, tprev = carry
    Z, t_curr, r, q, kappa, theta, sigma = X
    dt = t_curr - tprev

    # drift = dt * (r + -0.5 * sigma**2)
    # diffusion = sigma * jnp.sqrt(dt)
    # move = drift + diffusion * Z
    Z_var_process = Z[1::2]
    Z_spot_process = Z[0::2]
    v_curr = jnp.maximum(
        v_prev + kappa * (theta - v_prev) * dt + sigma * jnp.sqrt(dt) * Z_var_process,
        0.0,
    )
    drift = dt * (r + q - 0.5 * v_curr)
    move = drift + jnp.sqrt(v_curr) * Z_spot_process * jnp.sqrt(dt)

    perf_curr = perf_prev + move
    return (perf_curr, v_curr, t_curr), move


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
    Z = jrnd.normal(key, shape=(JTs.size, 2 * Nassets))
    init = (jnp.zeros(Nassets), v0, JT0)
    Xs = (Z, JTs, fwd_r, fwd_q, kappa, theta, sigma)
    _, paths = jax.lax.scan(evolve_heston, init, Xs)
    return S0 * jnp.exp(jnp.hstack(((0.0), jnp.cumsum(paths))))


if __name__ == "__main__":
    paths = mcpaths_heston(
        jrnd.PRNGKey(3),
        NRuns=2,
        NIter=1000,
        JTs=jnp.linspace(0.01, 1, 25),
        JT0=0.0,
        Nassets=1,
        fwd_r=jnp.full((25, 1), 0.05),
        fwd_q=jnp.zeros((25, 1)),
        covL=jnp.zeros((1, 1)),
        S0=jnp.array(
            [
                100.0,
            ]
        ),
        v0=jnp.sqrt(jnp.array([0.2])),
        kappa=jnp.zeros((25, 1)),
        theta=jnp.sqrt(jnp.full((25, 1), 0.2)),
        sigma=jnp.zeros((25, 1)),
    )
    print(paths)
