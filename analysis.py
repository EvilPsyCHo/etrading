import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import BayesianRidge, LinearRegression


import pickle
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import sys

from etrading.api import DataAPI
from etrading.optimize import Environment
from etrading import unit
from etrading.unit import Unit
from etrading.optimize import *



api = DataAPI()
ROOT = Path(__file__).parent
cols = [f"p{i}" for i in range(1, 97)]


def load_eng_info(path):
    api = DataAPI()
    with open(path, "rb") as f:
        data = pickle.load(f)
    engId = data["eng_id"]
    
    try:
        # line_map_df = data["line_map_df"]
        line_map_df = pd.read_csv(ROOT / "assets" / "mapping.csv")
        for col in line_map_df.columns:
            line_map_df[col] = line_map_df[col].astype(str)
    except:
        line_map = api.get_eng_line_map(engId)
        line_map_df = pd.DataFrame(line_map["data"])
    units_df = data["units_df"]
    new_energy_plan_df = data["new_energy_df"]
    bus_load_df = data["bus_load_df"]
    sys_load_df = data["sys_load_df"]

    units_df = units_df[["id", "engId", "unitName", "unitType", "unitCap", "busNode"]].rename(columns={"id": "unitId"})
    new_energy_plan_df = new_energy_plan_df[["id", "engId"] + cols].rename(columns={"id": "unitId"})
    bus_load_df = bus_load_df[["id", "engId", "ldName"] + cols].rename(columns={"id": "lineId", "ldName": "lineName"})
    bus_load_df = pd.melt(bus_load_df, id_vars=list(set(bus_load_df.columns) - set(cols)), value_name="power", var_name="time")
    new_energy_plan_df = pd.melt(new_energy_plan_df, id_vars=list(set(new_energy_plan_df.columns) - set(cols)), value_name="power", var_name="time")


    return engId, units_df, line_map_df, sys_load_df, bus_load_df, new_energy_plan_df

def load_eng_info_via_api(path):
    with open(Path(path) / "data.pkl", "rb") as f:
        data = pickle.load(f)
    eng_id = data["eng_id"]
    cols = [f"p{i}" for i in range(1, 97)]
    units = pd.DataFrame(api.get_eng_unit_info(eng_id)["data"]).rename(columns={"id": "unitId"})
    linemap = pd.DataFrame(api.get_eng_line_map(eng_id)["data"])

    busload = pd.DataFrame(api.get_eng_bus_load(eng_id)["data"])[["id", "engId", "ldName"] + cols].rename(columns={"id": "lineId", "ldName": "lineName"})
    busload = pd.melt(busload, id_vars=list(set(busload.columns) - set(cols)), value_name="power", var_name="time")

    sysload = pd.DataFrame(api.get_eng_system_load(eng_id)["data"])
    sysload = pd.melt(sysload, id_vars=list(set(sysload.columns) - set(cols)), value_name="power", var_name="time")

    new_energy_plan = pd.DataFrame(api.get_eng_new_energy_forecast(eng_id)["data"])[["id", "engId"] + cols].rename(columns={"id": "unitId"})
    new_energy_plan = pd.melt(new_energy_plan, id_vars=list(set(new_energy_plan.columns) - set(cols)), value_name="power", var_name="time")

    return eng_id, units, linemap, sysload, busload, new_energy_plan


def load_trade_result(path):
    cols = [f"p{i}" for i in range(1, 97)]
    df = pd.read_csv(path, dtype={"unitId": "str"})[["type", "unitId"] + cols]
    trade_price_df = df[df.type == "电价"].reset_index(drop=True).drop(columns=["type"])
    trade_price_df = pd.melt(trade_price_df, id_vars=list(set(trade_price_df.columns) - set(cols)), value_name="trade_price", var_name="time")
    trade_power_df = df[df.type == "电力"].reset_index(drop=True).drop(columns=["type"])
    trade_power_df = pd.melt(trade_power_df, id_vars=list(set(trade_power_df.columns) - set(cols)), value_name="trade_power", var_name="time")
    return trade_power_df, trade_price_df


def load_declare_result(path):
    declare_df = pd.read_csv(path, dtype={"unitId": "str"}).rename(columns={"price": "declare_price"})

    declare_pwoer = declare_df.groupby(["unitId", "unitName"])["end"].max().reset_index().rename(columns={"end": "declare_power"})
    declare_min_price = declare_df.groupby(["unitId", "unitName"])["declare_price"].min().reset_index().rename(columns={"declare_price": "declare_min_price"})
    declare_max_price = declare_df.groupby(["unitId", "unitName"])["declare_price"].max().reset_index().rename(columns={"declare_price": "declare_max_price"})
    declare_mean_price = declare_df.groupby(["unitId", "unitName"])["declare_price"].mean().reset_index().rename(columns={"declare_price": "declare_avg_price"})

    df = declare_pwoer.merge(declare_min_price, on=["unitId", "unitName"], how="left")\
        .merge(declare_max_price, on=["unitId", "unitName"], how="left")\
        .merge(declare_mean_price, on=["unitId", "unitName"], how="left")
    return df


def load_declare_and_trade_result(path, eng_id, round):
    declare_path = path / f"round_{round}" / "declaration.csv"
    declare_df = load_declare_result(declare_path)
    declare_df["engId"] = eng_id
    declare_df["roundId"] = round

    trade_path = path / f"round_{round}" / "clear_result.csv"
    trade_power_df, trade_price_df = load_trade_result(trade_path)
    trade_power_df["engId"] = eng_id
    trade_power_df["roundId"] = round
    trade_price_df["engId"] = eng_id
    trade_price_df["roundId"] = round

    merge_df = declare_df.merge(trade_power_df, how="outer", on=["unitId", "engId", "roundId"])
    merge_df = merge_df.merge(trade_price_df, how="outer", on=["unitId", "engId", "roundId", "time"])
    return merge_df



def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",props[col].dtype)
            print("******************************")
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eng_id")
    args = parser.parse_args()

    ##### Process data
    save_path = ROOT / "data" / args.eng_id
    save_path.mkdir(exist_ok=True, parents=True)
    eng_id, units, linemap, sysload, busload, new_energy_plan = load_eng_info(ROOT / "data" / args.eng_id / "data.pkl")
    trade_dfs = []
    rounds = [int(f.stem.replace("round_", "")) for f in Path(save_path).glob("round_*")]
    for round in rounds:
        try:
            trade_df = load_declare_and_trade_result(save_path, eng_id, round)
            trade_dfs.append(trade_df)
        except Exception as e:
            print(e)
    trade_dfs = pd.concat(trade_dfs, ignore_index=True)
    sysplan = pd.read_excel(Path(__file__).parent / "assets" / "联络线计划.xlsx", header=None)
    sysplan.columns = ["sysline"] + cols + ["create_time", "create_time2"]
    sysplan = pd.melt(sysplan, id_vars=list(set(sysplan.columns) - set(cols)), value_name="power", var_name="time")

    
    data = {
    "eng_id": eng_id,
    "sysplan": sysplan,
    "units": units,
    "trade_dfs": reduce_mem_usage(trade_dfs),
    "linemap": linemap,
    "sysload": sysload,
    "busload": busload,
    "new_energy_plan": new_energy_plan,
}

    
    with open(save_path / "process.pkl", "wb") as f:
        pickle.dump(data, f)

    #### Analysis
    eng_id = data["eng_id"]
    sysplan = data["sysplan"]
    units = data["units"]
    trade_dfs = data["trade_dfs"]
    linemap = data["linemap"]
    sysload = data["sysload"]
    busload = data["busload"]
    new_energy_plan = data["new_energy_plan"]

    unit_id_to_name = units.set_index("unitId")["unitName"].to_dict()
    num_rounds = trade_dfs.roundId.unique()
    cols = [f"p{i}" for i in range(1, 97)]

    ## 火电竞价空间与成交价格
    trade_space = busload.groupby("time")["power"].sum()[cols] - sysplan.groupby("time")["power"].sum()[cols] - new_energy_plan.groupby("time")["power"].sum()[cols]
    trade_space = trade_space.reset_index().rename(columns={"power": "space"})
    trade_dfs["trade_amount"] = trade_dfs["trade_power"] * trade_dfs["trade_price"]
    trade_amount = trade_dfs.groupby(["roundId", "time"])["trade_amount"].sum().reset_index()
    trade_power = trade_dfs.groupby(["roundId", "time"])["trade_power"].sum().reset_index()
    trade_avg_price = trade_power.merge(trade_amount, on=["roundId", "time"], how="left")
    trade_avg_price["trade_avg_price"] = trade_avg_price["trade_amount"] / trade_avg_price["trade_power"]
    trade_avg_price = trade_avg_price.merge(trade_space, how="left", on="time")

    corr_space_and_avg_price = trade_avg_price.groupby("roundId").apply(lambda x: x[["space", "trade_avg_price"]].corr().values[0][1], include_groups=False).fillna(0.)
    # filter valid exp
    corr_space_and_avg_price = corr_space_and_avg_price[corr_space_and_avg_price > 0]
    valid_rounds = corr_space_and_avg_price.index.tolist()
    trade_dfs = trade_dfs[trade_dfs.roundId.isin(valid_rounds)].reset_index(drop=True)
    trade_avg_price = trade_avg_price[trade_avg_price.roundId.isin(valid_rounds)].reset_index(drop=True)



    ## 平均报价与成交价格关系
    declare_price = []
    for _, row in trade_dfs.iterrows():
        if row.trade_power > 0:
            dp = None
            for p in [row.declare_min_price, row.declare_avg_price, row.declare_max_price]:
                if p+0.001 > row.trade_price:
                    dp = p
                    break
            if dp is None:
                dp = row.declare_max_price
        else:
            dp = row.declare_min_price
        declare_price.append(dp)
    trade_dfs["declare_price"] = declare_price

    ## 价格分解模型
    bayes_weight = {}
    for r in valid_rounds:
        mat = trade_avg_price.loc[trade_avg_price.roundId == r, ["space", "trade_avg_price"]].values
        lr = BayesianRidge().fit(mat[:, [0]], mat[:, 1])
        corr = np.corrcoef(mat[:,0], mat[:,1])[0][1]
        bayes_weight[r] = {"lr_coef": lr.coef_[0], "lr_intercept": lr.intercept_, "corr": corr}
    bayes_weight = pd.DataFrame(bayes_weight).T.reset_index().rename(columns={"index": "roundId"})

    unit_declare_intercept_corr = {}
    for uid in trade_dfs.unitId.unique():
        sub = trade_dfs[(trade_dfs.unitId == uid) & (trade_dfs.time == "p1")].reset_index(drop=True)
        sub = bayes_weight[bayes_weight.lr_coef > 0].merge(sub[["roundId", "declare_price"]], on="roundId")
        unit_declare_intercept_corr[uid] = sub[["lr_intercept", "declare_price"]].corr().values[0][1]
    
    unit_declare_intercept_corr = pd.Series(unit_declare_intercept_corr)
    margin_units = unit_declare_intercept_corr[unit_declare_intercept_corr>0.6].index.tolist()
    unit_declare_intercept_corr[unit_declare_intercept_corr<0] = np.clip(unit_declare_intercept_corr[unit_declare_intercept_corr<0] + 0.2, -0.5, 0)

    ## 全局价格模型
    margin_unit_avg_price = trade_dfs[trade_dfs.unitId.isin(margin_units)].groupby(["roundId", "time"])["declare_price"].mean().reset_index()
    trade_avg_price = trade_avg_price.merge(margin_unit_avg_price, on=["roundId", "time"])
    train_data = trade_avg_price[trade_avg_price.roundId.isin(corr_space_and_avg_price.index)]
    model_with_declare = BayesianRidge().fit(train_data[["space", "declare_price"]].values, train_data.trade_avg_price.values)
    model_without_declre = BayesianRidge().fit(train_data[["space"]].values, train_data.trade_avg_price.values)

    # 节点电价形成机制
    trade_dfs = trade_dfs.merge(units[["unitId", "busNode"]], on="unitId", how="left")
    node_trade_price = trade_dfs.groupby(["busNode", "roundId", "time"])["trade_price"].mean().reset_index()

    def get_node_unit_corr_mat():
        mat = {}
        for unit_id in units.unitId:
            sub = node_trade_price.merge(trade_dfs.loc[trade_dfs.unitId==unit_id, ["unitId", "roundId", "time", "declare_price"]], on=["roundId", "time"], how="left")
            sub = sub[sub.roundId.isin(corr_space_and_avg_price.index)]
            mat[unit_id] = sub.groupby("busNode").apply(lambda x: np.corrcoef(x["declare_price"], x["trade_price"])[0][1], include_groups=False)
        return pd.DataFrame(mat)

    node_unit_corr_mat = get_node_unit_corr_mat()
    valid_node_unit = node_unit_corr_mat.isnull().mean()[node_unit_corr_mat.isnull().mean() == 0].index.tolist()
    node_unit_corr_mat = node_unit_corr_mat[valid_node_unit]


    analysis = {
    "node_unit_corr_mat": node_unit_corr_mat,
    "trade_dfs": reduce_mem_usage(trade_dfs),
    "trade_avg_price": trade_avg_price,
    "corr_space_and_avg_price": corr_space_and_avg_price,
    "unit_declare_intercept_corr": unit_declare_intercept_corr,
    "bayes_weight": bayes_weight,
    "global_price_model_with_margin_declare": model_with_declare,
    "global_price_model": model_without_declre
}
    with open(save_path / "analysis.pkl", "wb") as f:
        pickle.dump(analysis, f)
    
    with open(save_path / "info.pkl", "rb") as f:
        info = pickle.load(f)
    info["valid_exp_rounds"] = analysis["trade_dfs"].roundId.nunique()
    with open(save_path / "info.pkl", "wb") as f:
        pickle.dump(info, f)
