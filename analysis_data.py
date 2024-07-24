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
    parser.add_argument("--data")
    args = parser.parse_args()

    with open(args.data, "rb") as f:
        data = pickle.load(f)
    print(data.keys())
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

    corr_space_and_avg_price = trade_avg_price.groupby("roundId").apply(lambda x: x[["space", "trade_avg_price"]].corr().values[0][1]).fillna(0.)
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
    model_with_declare = BayesianRidge().fit(train_data[["space", "declare_price"]], train_data.trade_avg_price)
    model_without_declre = BayesianRidge().fit(train_data[["space"]], train_data.trade_avg_price)

    # 节点电价形成机制
    trade_dfs = trade_dfs.merge(units[["unitId", "busNode"]], on="unitId", how="left")
    node_trade_price = trade_dfs.groupby(["busNode", "roundId", "time"])["trade_price"].mean().reset_index()

    def get_node_unit_corr_mat():
        mat = {}
        for unit_id in units.unitId:
            sub = node_trade_price.merge(trade_dfs.loc[trade_dfs.unitId==unit_id, ["unitId", "roundId", "time", "declare_price"]], on=["roundId", "time"], how="left")
            sub = sub[sub.roundId.isin(corr_space_and_avg_price.index)]
            mat[unit_id] = sub.groupby("busNode").apply(lambda x: np.corrcoef(x["declare_price"], x["trade_price"])[0][1])
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
    with open("/home/kky/project/etrading/output/841c8cf7-0941-4600-9276-4889fd71f163/analysis.pkl", "wb") as f:
        pickle.dump(analysis, f)
