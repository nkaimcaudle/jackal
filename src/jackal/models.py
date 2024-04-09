from pydantic import BaseModel, Field
import pydantic
import string
import numpy as np
from rich import print
import datetime
from typing import Literal


def generate_pk(n: int = 6) -> str:
    alphabet = list(string.ascii_letters + string.digits)
    return "".join(np.random.choice(alphabet) for _ in range(n))


class PricingItem(BaseModel):
    pk: str = Field(default_factory=generate_pk)


class ProductModel(PricingItem):
    Currency: str = Field(min_length=3, max_length=3)


class VanillaOption(ProductModel):
    Underlying: str
    Strike: float
    OptionType: Literal["Call", "Put"]


class EuroVanillaOption(VanillaOption):
    ExerciseDate: pydantic.AwareDatetime


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

