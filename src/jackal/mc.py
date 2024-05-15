import jackal.models as models
import datetime
from typing import Callable
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jrnd
import jackal.utils
import pydantic
from rich import print


def combine_dates(
    engine: models.MCEngineModel,
    product: models.ProductModel,
    asof: pydantic.AwareDatetime,
) -> tuple[pydantic.AwareDatetime, ...]:
    euro_dates = product.get_modeling_european_dates
    amer_dates = product.get_modeling_american_dates
    if len(amer_dates):
        amer_dates_begin, amer_dates_end = min(amer_dates), max(amer_dates)
        dts = (amer_dates_begin,)
        new_date = amer_dates_begin + engine.TimeStep
        while new_date < amer_dates_end:
            dts = dts + (new_date,)
            new_date += engine.TimeStep
    else:
        dts = ()
    return tuple(sorted(set((asof,) + euro_dates + dts)))


def paths_fn(
    key: jrnd.PRNGKey,
    i: int,
    M: int,
    D: int,
    NUnderlyings: int,
    Ts: jnp.array,
    _evolve: Callable,
    Tdate0: float,
) -> jnp.array:
    path_key = jrnd.fold_in(key, i)
    W = jrnd.normal(key=path_key, shape=(M,))
    init = (0.0, Tdate0)
    _, paths = jax.lax.scan(_evolve, init, (W, Ts))
    return jnp.exp(jnp.hstack(((0.0), jnp.cumsum(paths))))


def price_mono_product(
    engine: models.MCEngineModel,
    product: models.ProductModel,
    curve: models.IRCurveModel,
    eq: models.EqMarketData,
    vol: models.LogNormalVolModel,
) -> float:
    assert product.get_underlyings[0] == eq.Underlying
    assert curve.ValuationDate == eq.ValuationDate

    asof = curve.AsOf
    modeling_dates = combine_dates(engine, product, asof)
    modeling_Tdates = jnp.asarray(list(map(jackal.utils.to_Tdate, modeling_dates)))

    key = jrnd.PRNGKey(engine.Seed)
    _paths = partial(
        paths_fn,
        key=key,
        M=len(modeling_dates) - 1,
        D=vol.NFactors,
        NUnderlyings=1,
        Ts=modeling_Tdates[1:],
        _evolve=vol.evolve_fn(),
        Tdate0=modeling_Tdates[0],
    )
    paths = jax.vmap(_paths)(i=jnp.arange(engine.NRuns * engine.NIter))
    paths = jnp.reshape(paths, (engine.NRuns, engine.NIter, 1, -1))
    paths = mkt.Spot * paths
    _payoff = partial(
        product.payoff,
        underlyings=(product.Underlying,),
        pegs=modeling_dates,
        ircurve=curve,
    )
    payoffs = jax.vmap(
        jax.vmap(
            _payoff,
        ),
    )(paths)
    return payoffs.mean(), payoffs.std() / jnp.sqrt(engine.NRuns * engine.NIter)


if __name__ == "__main__":
    opt1 = models.EuroVanillaOption(
        Currency="USD",
        Underlying="BABA",
        Strike=105.0,
        OptionType="Call",
        ExerciseDate=datetime.datetime(2025, 4, 10, 16, 30, tzinfo=datetime.UTC),
    )
    opt2 = opt1.copy(update=dict(OptionType="Put", pk=None))
    engine = models.MCEngineModel(
        NRuns=20,
        NIter=80_000,
    )
    curve = models.FlatIRCurve(Currency="USD", Underlying="USD", Rate=0.05)
    mkt = models.EqMarketData(Currency="USD", Underlying="BABA", Spot=100.0)
    vol = models.FlatEqVol(Underlying="BABA", Currency="USD", RefSpot=100.0, Vol=0.2)
    print(
        curve,
        engine,
        mkt,
        vol,
    )

    print()
    print(opt1)
    a = price_mono_product(engine, opt1, curve, mkt, vol)
    print(a)

    print()
    print(opt2)
    b = price_mono_product(engine, opt2, curve, mkt, vol)
    print(b)
