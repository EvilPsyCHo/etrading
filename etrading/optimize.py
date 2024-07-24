
from pathlib import Path
import os
import pandas as pd
import time
from datetime import datetime
import pickle
import json
import logging

import optuna
import numpy as np 

from etrading.api import DataAPI
from etrading.unit import Unit


def get_logger(path):
    logger = logging.getLogger('etrading')
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H:%M:%S")
    file_handler = logging.FileHandler(os.path.join(path, f"{formatted_datetime}.log"))
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


api = DataAPI()
cols = [f"p{i}" for i in range(1, 97)]


class Environment:

    def __init__(self, eng_id, path, units_df, bus_load_df, sys_load_df, new_energy_df, linemap_df, studies, optimize_info):
        self.eng_id = eng_id
        self.path = Path(path)
        self.units_df = units_df
        self.units = self.convert_unit_dataframe_to_unit_obj(units_df, new_energy_df)
        self.bus_load = bus_load_df
        self.sys_load = sys_load_df
        self.new_energy = new_energy_df
        self.linemap_df = linemap_df
        self.studies = studies
        self.optimize_info = optimize_info
        self.logger = get_logger(path)
    
    @classmethod
    def convert_unit_dataframe_to_unit_obj(cls, units_df, new_energy_df):
        units = []
        for _, row in units_df.iterrows():
            unit = Unit(**row.to_dict())
            if unit.id in set(new_energy_df.id):
                unit.forecast = new_energy_df.loc[new_energy_df.id == unit.id, cols].values[0].tolist()
            units.append(unit)
        return units_df

    
    @classmethod
    def load(cls, env_dir):
        with open(os.path.join(env_dir, "data.pkl"), "rb") as f:
            data = pickle.load(f)
        with open(os.path.join(env_dir, "optimize_info.pkl"), "rb") as f:
            optimize_info = pickle.load(f)
        with open(os.path.join(env_dir, "studies.pkl"), "rb") as f:
            studies = pickle.load(f)
        return cls(optimize_info=optimize_info, studies=studies, **data)
        

    @classmethod
    def create(cls, save_dir, eng_name, remark):
        eng_id = api.create_eng(eng_name, remark)["data"]
        api.load_eng_base_data(eng_id)
        path = Path(save_dir) / eng_name
        path.mkdir(exist_ok=True, parents=True)
        units_df = pd.DataFrame(api.get_eng_unit_info(eng_id)["data"])
        # units.to_csv(path / "units.csv", index=False)
        bus_load_df = pd.DataFrame(api.get_eng_bus_load(eng_id)["data"])
        # bus_load.to_csv(path / "bus_load.csv", index=False)
        sys_load_df = pd.DataFrame(api.get_eng_system_load(eng_id)["data"])
        # sys_load.to_csv(path / "sys_load.csv", index=False)
        new_energy_df = pd.DataFrame(api.get_eng_new_energy_forecast(eng_id)["data"])
        # new_energy.to_csv(path / "new_energy.csv", index=False)
        linemap_df = pd.DataFrame(api.get_eng_line_map(eng_id)["data"])
        
        units = cls.convert_unit_dataframe_to_unit_obj(units_df, new_energy_df)
        # studies
        studies = {}
        for unit in units:
            if unit.unitType == "101":
                studies[unit.id] = optuna.create_study(direction="maximize")



        with open(path / "data.pkl", "wb") as f:
            data = {
                "eng_id": eng_id,
                "path": str(path),
                "units_df": units_df,
                "bus_load_df": bus_load_df,
                "sys_load_df": sys_load_df,
                "new_energy_df": new_energy_df,
                "linemap_df": linemap_df,
            }
            pickle.dump(data, f)
        with open(path / "studies.pkl", "wb") as f:
            pickle.dump(studies, f)
        
        optimize_info = []
        with open(path / "optimize_info.pkl", "wb") as f:
            pickle.dump(optimize_info, f)
        return cls(eng_id, str(path), units_df, units, bus_load_df, sys_load_df, new_energy_df, studies, optimize_info)

    def random_step(self, mu, bias):
        round = len(self.optimize_info)
        path = self.path / f"round_{round}"
        path.mkdir(exist_ok=True, parents=True)

        self.logger.info(f"<<random step {round}>>  generate declarations")
        declares = []
        for unit in self.units:
            if unit.unitType != "101":
                declares.extend(unit.zero_declare())
            else:
                declares.extend(unit.random_declare(mu, bias))
        declares_df = pd.DataFrame(declares)
        declares_df.to_csv(path / "declaration.csv", index=False)

        non_zero_mean = declares_df[declares_df.price > 0].price.mean()
        nan_zero_max = declares_df[declares_df.price > 0].price.max()
        non_zero_min = declares_df[declares_df.price > 0].price.min()
        self.logger.info(f"<<optimization step {round}>>  declarations price non zero min {non_zero_min:.2f}, mean {non_zero_mean:.2f}, max {nan_zero_max:.2f}")

        self.logger.info(f"<<optimization step {round}>>  submit declarations")
        while True:
            content = api.create_eng_declaration(declares)
            if content["status"] == 200:
                break
            else:
                self.logger.info(f"<<optimization step {round}>>  {content}, waiting 30 seconds and submit declarations again...")
                time.sleep(30)
        
        self.logger.info(f"<<optimization step {round}>>  submit clearing")
        while True:
            content = api.create_eng_clearing(self.eng_id)
            if content["status"] == 200:
                break
            else:
                self.logger.info(f"<<optimization step {round}>>  {content}, waiting 30 seconds and submit clearing again ...")
                time.sleep(30)


        time.sleep(60 * 15)
        self.logger.info(f"<<optimization step {round}>>  get clearing result")
        while True:
            content = api.get_eng_clearing_power_and_price(self.eng_id)
            if content["data"]:
                break
            else:
                self.logger.info(f"<<optimization step {round}>>  waiting 60 seconds and get clearing again ...")
                time.sleep(60)
        

        clear_result = content["data"]
        clear_result_df = pd.DataFrame(clear_result)
        clear_result_df.to_csv(path / "clear_result.csv", index=False)

        prices_df = clear_result_df[clear_result_df.type=="电价"].reset_index(drop=True)
        quantity_df = clear_result_df[clear_result_df.type=="电力"].reset_index(drop=True)
        
        max_quantity = quantity_df[cols].sum().max()
        max_price = prices_df[cols].max().max()
        declares_df["q"] = declares_df["end"] - declares_df["start"]

        prices_df = prices_df.merge(self.units_df.rename(columns={"id": "unitId"})[["unitId", "unitType", "unitCap"]], on="unitId", how="left")
        quantity_df = quantity_df.merge(self.units_df.rename(columns={"id": "unitId"})[["unitId", "unitType", "unitCap"]], on="unitId", how="left")

        thermal_avg_price = np.sum(prices_df.loc[prices_df.unitType=="101", cols].values * quantity_df.loc[quantity_df.unitType=="101", cols].values) / np.sum(quantity_df.loc[quantity_df.unitType=="101", cols].values)
        thermal_quantity = np.sum(quantity_df.loc[quantity_df.unitType=="101", cols].values)

        max_prices_declare = declares_df.loc[declares_df.price <= max_price, "q"].sum()
        self.logger.info(f"<<optimization step {round}>>  max quantity {max_quantity:.3e}, max price {max_price:.2f}, max price declaration {max_prices_declare:.2f}, thermal quantity {thermal_quantity:.3e} avg price {thermal_avg_price:.2f}")
        self.optimize_info.append(round)
        with open(self.path / "optimize_info.pkl", "wb") as f:
            pickle.dump(self.optimize_info, f)
    

    def step(self):
        round = len(self.optimize_info)
        path = self.path / f"round_{round}"
        path.mkdir(exist_ok=True, parents=True)

        self.logger.info(f"<<optimization step {round}>>  generate declarations")
        declares = []
        trials = {}
        for unit in self.units:
            if unit.unitType != "101":
                declares.extend(unit.zero_declare())
            else:
                study = self.studies[unit.id]
                trial = study.ask()
                trials[unit.id] = trial
                declares.extend(unit.opt_declare(trial))
        declares_df = pd.DataFrame(declares)
        declares_df.to_csv(path / "declaration.csv", index=False)

        non_zero_mean = declares_df[declares_df.price > 0].price.mean()
        nan_zero_max = declares_df[declares_df.price > 0].price.max()
        non_zero_min = declares_df[declares_df.price > 0].price.min()
        self.logger.info(f"<<optimization step {round}>>  declarations price non zero min {non_zero_min:.2f}, mean {non_zero_mean:.2f}, max {nan_zero_max:.2f}")

        self.logger.info(f"<<optimization step {round}>>  submit declarations")
        while True:
            content = api.create_eng_declaration(declares)
            if content["status"] == 200:
                break
            else:
                self.logger.info(f"<<optimization step {round}>>  {content}, waiting 30 seconds and submit declarations again...")
                time.sleep(30)
        
        self.logger.info(f"<<optimization step {round}>>  submit clearing")
        while True:
            content = api.create_eng_clearing(self.eng_id)
            if content["status"] == 200:
                break
            else:
                self.logger.info(f"<<optimization step {round}>>  {content}, waiting 30 seconds and submit clearing again ...")
                time.sleep(30)


        time.sleep(60 * 15)
        self.logger.info(f"<<optimization step {round}>>  get clearing result")
        while True:
            content = api.get_eng_clearing_power_and_price(self.eng_id)
            if content["data"]:
                break
            else:
                self.logger.info(f"<<optimization step {round}>>  waiting 60 seconds and get clearing again ...")
                time.sleep(60)
        

        clear_result = content["data"]
        clear_result_df = pd.DataFrame(clear_result)
        clear_result_df.to_csv(path / "clear_result.csv", index=False)

        prices_df = clear_result_df[clear_result_df.type=="电价"].reset_index(drop=True)
        quantity_df = clear_result_df[clear_result_df.type=="电力"].reset_index(drop=True)
        
        max_quantity = quantity_df[cols].sum().max()
        max_price = prices_df[cols].max().max()
        declares_df["q"] = declares_df["end"] - declares_df["start"]

        prices_df = prices_df.merge(self.units_df.rename(columns={"id": "unitId"})[["unitId", "unitType", "unitCap"]], on="unitId", how="left")
        quantity_df = quantity_df.merge(self.units_df.rename(columns={"id": "unitId"})[["unitId", "unitType", "unitCap"]], on="unitId", how="left")

        thermal_avg_price = np.sum(prices_df.loc[prices_df.unitType=="101", cols].values * quantity_df.loc[quantity_df.unitType=="101", cols].values) / np.sum(quantity_df.loc[quantity_df.unitType=="101", cols].values)
        thermal_quantity = np.sum(quantity_df.loc[quantity_df.unitType=="101", cols].values)

        max_prices_declare = declares_df.loc[declares_df.price <= max_price, "q"].sum()
        self.logger.info(f"<<optimization step {round}>>  max quantity {max_quantity:.3e}, max price {max_price:.2f}, max price declaration {max_prices_declare:.2f}, thermal quantity {thermal_quantity:.3e} avg price {thermal_avg_price:.2f}")
        usefull_ids = set(clear_result_df.unitId)

        total_reward = 0
        for unit in self.units:
            if unit.id in usefull_ids:
                if unit.unitType == "101":
                    study = self.studies[unit.id]
                    prices = prices_df.loc[prices_df.unitId == unit.id, cols].values[0].tolist()
                    quantities = quantity_df.loc[quantity_df.unitId == unit.id, cols].values[0].tolist()
                    reward = unit.calc_reward(prices, quantities)
                    study.tell(trials[unit.id], reward)
                    total_reward += reward
        self.logger.info(f"<<optimization step {round}>>  optimization total reward {total_reward:.3e}")
        with open(self.path / "studies.pkl", "wb") as f:
            pickle.dump(self.studies, f)
        self.optimize_info.append({"total_reward": total_reward})
        with open(self.path / "optimize_info.pkl", "wb") as f:
            pickle.dump(self.optimize_info, f)

    
    def run(self, n=1):
        for i in range(n):
            self.step()
    
    def random_run(self, mu=250, bias=50, n=1):
        for i in range(n):
            self.random_step(mu, bias)


if __name__ == "__main__":
    env = Environment.create("/Users/kky/project/etrading/data/bayes-run-60-min200", "bayes-run-60-min200", "bayes-run-60-min200")
    env.run(40)
