from pydantic import BaseModel, Field
import pydantic
import string
import numpy as np
import jax.numpy as jnp
from rich import print
import datetime
from typing import Literal


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def generate_pk(n: int = 6) -> str:
    alphabet = list(string.ascii_letters + string.digits)
    return "".join(np.random.choice(alphabet) for _ in range(n))


class PricingItem(BaseModel):
    pk: str = Field(default_factory=generate_pk)


class MarketDataModel(PricingItem):
    Underlying: str
    Currency: str = Field(max_length=3, min_length=3)


class MarketDataDatedModel(MarketDataModel):
    ValuationDate: datetime.date
    AsOf: pydantic.AwareDatetime = Field(default_factory=_utc_now)


class IRCurveModel(MarketDataDatedModel):
    pass


class FlatIRCurve(IRCurveModel):
    Rate: float


class ProductModel(PricingItem):
    Currency: str = Field(min_length=3, max_length=3)

    @property
    def get_underlyings(self) -> tuple[str, ...]:
        raise NotImplementedError()

    @property
    def get_modeling_dates(self) -> tuple[datetime.datetime, ...]:
        raise NotImplementedError()

    def payoff(
        self, paths: jnp.array, dates: jnp.array, ircurve: IRCurveModel
    ) -> float:
        pass


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
    def get_modeling_dates(self) -> tuple[datetime.datetime, ...]:
        return (self.ExerciseDate,)


class AmerVanillaOption(VanillaOption):
    ExerciseStart: pydantic.AwareDatetime
    ExerciseEnd: pydantic.AwareDatetime


class RFQ(PricingItem):
    Product: ProductModel
    Bid: float | None = None
    Ask: float | None = None
    BidSize: float | None = None
    AskSize: float | None = None


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
    def NFactors(self) -> int:
        raise NotImplementedError()


class LogNormalVolModel(VolModel):
    pass


class FlatEqVol(LogNormalVolModel):
    Vol: float = Field(gt=0.0)

    def NFactors(self) -> int:
        return 1


class EngineModel(BaseModel):
    pass


class MCEngineModel(EngineModel):
    NRuns: int = Field(ge=1)
    NIter: int = Field(ge=1)
    Seed: int = 0
