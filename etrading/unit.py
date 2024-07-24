from pydantic import BaseModel, Field
from typing import List, Dict, Literal, Type, Optional, Union
import numpy as np
import time


class Declaration(BaseModel):
    id: Optional[str] = Field(default=None)
    unitId: str
    unitName: str
    point: int = Field(description="报价段序号")
    start: Union[int, float]
    end: Union[int, float]
    price: Union[int, float]
    engId: str


class Unit(BaseModel):
    id: str = Field(description="unit id")
    engId: str = Field(description="eng id")
    unitName: str
    unitType: str = Field(description="101火电, 201水电, 301风电, 302光伏, 901 placeholder")
    unitCap: float = Field(description="额定负荷")
    busNode: str = Field(description="所属母线节点")
    forecast: Optional[List] = Field(default_factory=list)

    def calc_reward(self, prices, quantities):
        rewards = []
        for p, q in zip(prices, quantities):
            rewards.append(self._calc_reward(p, q))
        return np.sum(rewards)

    def _calc_reward(self, price, quantity):
        # 边际度电成本函数
        def cost_fn_1000(x):
            return 2.0538e-7*(x -790.2424773590417)**2 + 0.24790364592462752 - 0.1
        def cost_fn_600(x):
            return 6.1614e-7*(x -527.2340701788555)**2 + 0.20482801230239875 - 0.1
        def cost_fn_400(x):
            return 6.1614e-7*(x -327.2340701788555)**2 + 0.20482801230239875 - 0.1
        def cost_fn_200(x):
            return 6.1614e-7*(x -157.2340701788555)**2 + 0.21482801230239875 - 0.1
        def cost_fn_100(x):
            return 6.1614e-7*(x -80.2340701788555)**2 + 0.21482801230239875 - 0.1
        if self.unitType == "101":
            if self.unitCap < 100:
                cost_fn = cost_fn_100
            elif self.unitCap < 200:
                cost_fn = cost_fn_200
            elif self.unitCap < 400:
                cost_fn = cost_fn_400
            elif self.unitCap < 600:
                cost_fn = cost_fn_600
            else:
                cost_fn = cost_fn_1000
            if quantity == 0:
                return -1e8
            else:
                return price * quantity - 1e3 * cost_fn(quantity) * quantity
        else:
            return price * quantity

    def zero_declare(self):
        valid_forecast =  max(self.forecast) if self.forecast else self.unitCap
        declaration = Declaration(
                unitId=self.id,
                unitName=self.unitName,
                point=0,
                start=0,
                end=min(self.unitCap, valid_forecast),
                price=0,
                engId=self.engId
            )
        return [declaration.model_dump()]

    def random_declare(self, mu=250, bias=100):
        res = []
        num_bins = 3
        bins = np.linspace(0, self.unitCap*0.8, num_bins+1)
        prices = np.sort(np.random.random(num_bins) * mu + bias)
        for point, (start, end, price) in enumerate(zip(bins[:num_bins], bins[1:num_bins+1], prices)):
            declaration = Declaration(
            unitId=self.id,
            unitName=self.unitName,
            point=point,
            start=start,
            end=end,
            price=price,
            engId=self.engId
        )
            res.append(declaration.model_dump())
        return res
    
    def opt_declare(self, trial):
        num_bins = 4
        mu = trial.suggest_float("mu", 200, 400)
        std = trial.suggest_float("std", 0, 50)
        prices = np.clip(np.random.normal(mu, std, num_bins), a_min=200, a_max=500)
        prices = np.sort(prices)
        prices = prices + np.array([i*1e-6 for i in range(num_bins)])
        bins = np.linspace(0, self.unitCap, num_bins)

        res = []
        for point, (start, end, price) in enumerate(zip(bins[:num_bins], bins[1:num_bins+1], prices)):
            declaration = Declaration(
            unitId=self.id,
            unitName=self.unitName,
            point=point,
            start=start,
            end=end,
            price=price,
            engId=self.engId
        )
            res.append(declaration.model_dump())
        return res
