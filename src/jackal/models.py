from pydantic import BaseModel, Field
from functools import partial
from typing import Callable
import pydantic
import string
import numpy as np
import jax.numpy as jnp
import jax
from rich import print
import datetime
from typing import Literal
from jackal.utils import date_diff


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def generate_pk(n: int = 6) -> str:
    alphabet = list(string.ascii_letters + string.digits)
    return "".join(np.random.choice(alphabet) for _ in range(n))


class PricingItem(BaseModel):
    pk: str = Field(default_factory=generate_pk)

    class Config:
        frozen: bool = True


class MarketDataModel(PricingItem):
    Underlying: str
    Currency: str = Field(max_length=3, min_length=3)


class MarketDataDatedModel(MarketDataModel):
    AsOf: pydantic.AwareDatetime = Field(default_factory=_utc_now)

    @property
    def ValuationDate(self) -> datetime.date:
        return self.AsOf.date()


class IRCurveModel(MarketDataDatedModel):
    def discfact(self, dt: pydantic.AwareDatetime) -> float:
        pass


class FlatIRCurve(IRCurveModel):
    Rate: float

    def discfact(self, dt: pydantic.AwareDatetime) -> float:
        t = date_diff(self.AsOf, dt)
        return jnp.exp(-self.Rate * t)


class ProductModel(PricingItem):
    Currency: str = Field(min_length=3, max_length=3)

    @property
    def get_underlyings(self) -> tuple[str, ...]:
        raise NotImplementedError()

    @property
    def get_modeling_european_dates(self) -> tuple[pydantic.AwareDatetime, ...]:
        raise NotImplementedError()

    @property
    def get_modeling_american_dates(self) -> tuple[pydantic.AwareDatetime, ...]:
        return ()


class VanillaOption(ProductModel):
    Underlying: str
    Strike: float
    OptionType: Literal["Call", "Put"]

    @property
    def get_underlyings(self) -> tuple[str, ...]:
        return (self.Underlying,)


class EuroVanillaOption(VanillaOption):
    ExerciseDate: pydantic.AwareDatetime

    @property
    def get_modeling_european_dates(self) -> tuple[pydantic.AwareDatetime, ...]:
        return (self.ExerciseDate,)

    def payoff(
        self,
        paths: jnp.array,
        underlyings: tuple[str, ...],
        pegs: list[pydantic.AwareDatetime],
        ircurve: IRCurveModel,
    ) -> float:
        ul_idx = underlyings.index(self.Underlying)
        peg_idx = pegs.index(self.ExerciseDate)
        alpha = 10e4
        ST = paths[ul_idx, peg_idx]
        a = ST - self.Strike if self.OptionType == "Call" else self.Strike - ST
        cashflow = jax.nn.softplus(a * alpha) / alpha
        return ircurve.discfact(self.ExerciseDate) * cashflow


class AmerVanillaOption(VanillaOption):
    ExerciseStart: pydantic.AwareDatetime
    ExerciseEnd: pydantic.AwareDatetime


class RFQ(PricingItem):
    Product: ProductModel
    AsOf: pydantic.AwareDatetime
    Bid: float | None = None
    Ask: float | None = None
    BidSize: float | None = None
    AskSize: float | None = None

    @property
    def mid(self) -> float:
        return 0.5 * (self.Bid + self.Ask)

    @property
    def weighted_mid(self) -> float:
        nom = self.AskSize * self.Bid + self.BidSize * self.Ask
        denom = self.BidSize + self.AskSize
        return nom / denom


class Execution(PricingItem):
    Product: ProductModel
    Side: Literal["Buy", "Sell"]
    Size: float = Field(gt=0)
    Price: float
    When: pydantic.AwareDatetime


class Holding(PricingItem):
    Product: ProductModel
    Executions: list[Execution]

    @property
    def Notional(self) -> float:
        long = sum(
            execution.Size for execution in self.Executions if execution.Side == "Buy"
        )
        short = sum(
            execution.Size for execution in self.Executions if execution.Side == "Sell"
        )
        return long - short


class Portfolio(PricingItem):
    Holdings: list[Holding]


class EqMarketData(MarketDataDatedModel):
    Spot: float
    DivYield: float = 0.0
    RepoYield: float = 0.0


class VolModel(MarketDataDatedModel):
    @property
    def NFactors(self) -> int:
        raise NotImplementedError()


class LogNormalVolModel(VolModel):
    RefSpot: float


class FlatEqVol(LogNormalVolModel):
    Vol: float = Field(gt=0.0)

    @property
    def NFactors(self) -> int:
        return 1

    def evolve_fn(self) -> Callable:
        def evolve(carry, X, r, sigma):
            perf_prev, tprev = carry
            Z, tnew = X
            dt = tnew - tprev

            drift = dt * (r + -0.5 * sigma**2)
            diffusion = sigma * jnp.sqrt(dt)
            move = drift + diffusion * Z

            perf_new = perf_prev + move
            return (perf_new, tnew), move

        return partial(evolve, r=0.05, sigma=self.Vol)


class CorrelationModel(BaseModel):
    Correls: list[tuple[str, str, float]]


class EngineModel(BaseModel):
    pass


class MCEngineModel(EngineModel):
    NRuns: int = Field(ge=1)
    NIter: int = Field(ge=1)
    Seed: int = 0
    TimeStep: datetime.timedelta = datetime.timedelta(days=1)


class MarketDataManager(BaseModel):
    curves: dict[str, IRCurveModel]
    eqmkdata: dict[str, EqMarketData]
    eqvols: dict[str, LogNormalVolModel]
