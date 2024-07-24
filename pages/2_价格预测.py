import streamlit as st
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
import seaborn as sns
from sklearn.linear_model import BayesianRidge, LinearRegression

### config
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css("style.css")


with st.sidebar:
    prompt = f'''### 陕西电力现货模拟策略分析\n

工程ID: {st.session_state["data"]["eng_id"]}\n
机组数量: {len(st.session_state["data"]["units"])}\n
系统负荷(MW): {round(st.session_state["data"]["busload"].power.sum() / 96, 2)}\n
新能源出力(MW): {round(st.session_state["data"]["new_energy_plan"].power.sum() / 96,2)}\n
'''
    st.markdown(prompt)
cols = [f"p{i}" for i in range(1, 97)]


def daily_predict_global_price_with_margin_declare(space, margin_min, margin_max):
    preds = []
    for s in space:
        preds.append(_predict_global_price_with_margin_declare(s, margin_min, margin_max))
    return np.array(preds)


def daily_predict_global_price(space):
    preds = []
    for s in space:
        preds.append(_predict_global_price(s))
    return np.array(preds)
        


def _predict_global_price_with_margin_declare(space, margin_min, margin_max):
    margin_avg = (margin_min + margin_max) / 2
    margin_std = (margin_max - margin_min) / 4
    margin = np.random.normal(margin_avg, margin_std, 100).tolist()
    inputs = np.array([[space] * 100, margin]).T
    pred_avg, pred_std = st.session_state["analysis"]["global_price_model_with_margin_declare"].predict(inputs, return_std=True)
    preds = []
    for avg, std in zip(pred_avg, pred_std):
        preds.extend(np.random.normal(avg, std, 10).tolist())
    return preds


def _predict_global_price(space):
    inputs = np.array([[space] * 100]).T
    pred_avg, pred_std = st.session_state["analysis"]["global_price_model"].predict(inputs, return_std=True)
    preds = []
    for avg, std in zip(pred_avg, pred_std):
        preds.extend(np.random.normal(avg, std, 10).tolist())
    return preds


def plot_daily_price_prediction(preds):
    mu = preds.mean(axis=1)
    std = preds.std(axis=1)
    fig = plt.figure()
    plt.plot(mu)
    plt.fill_between(range(len(mu)), mu - std/4 + (np.random.random(len(mu)) - 0.5) * std/4, mu + std /4 + (np.random.random(len(mu)) - 0.5) * std/4, color='gray', alpha=0.3, label='Uncertainty') 
    # plt.ylim(0, max(mu) + max(std) + 20)
    plt.xlabel("时间段（96点）")
    plt.ylabel("价格（元/MWH）")
    st.pyplot(fig)
    plt.close(fig)


def simulate(pred_price, declare_price, declare_quantity, num_rounds=1000):
    day_simulate_trade_price = []
    day_simulate_trade_quantity = []

    for i in range(96):
        simulate_trade_quantity = []
        simulate_trade_price = []
        for _ in range(num_rounds):
            p = random.choice(pred_price[i])
            curr_q = 0
            for dp, dq in zip(declare_price, declare_quantity):
                if dp <= p:
                    curr_q += dq
            simulate_trade_quantity.append(curr_q)
            simulate_trade_price.append(p)
        day_simulate_trade_price.append(simulate_trade_price)
        day_simulate_trade_quantity.append(simulate_trade_quantity)
    day_simulate_trade_price = np.array(day_simulate_trade_price).mean(axis=0)
    day_simulate_trade_quantity = np.array(day_simulate_trade_quantity).sum(axis=0) / 4
    return day_simulate_trade_price, day_simulate_trade_quantity


def search(quantity):
    mu = []
    std = []
    for p in np.arange(0, 500, 5):
        day_simulate_trade_price, day_simulate_trade_quantity = simulate(st.session_state["predict"]["price"], [p], [quantity])
        day_simulate_trade_sale = np.round(day_simulate_trade_price * day_simulate_trade_quantity / 1e4, 1)
        mu.append(np.mean(day_simulate_trade_sale))
        std.append(np.std(day_simulate_trade_sale))
    return np.array(mu), np.array(std), np.arange(0, 500, 5)


def plot_simulate_trade_rate(trade_rate):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(trade_rate, ax=ax)
    plt.title("成交率分布")
    st.pyplot(fig)
    plt.close(fig)


def plot_simulate_trade_sale(trade_sale):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(trade_sale, ax=ax)
    plt.title("收益分布（万元）")
    st.pyplot(fig)
    plt.close(fig)



st.markdown("# 价格预测模型")

### 模型输入
space_df = pd.DataFrame(st.session_state["analysis"]["trade_avg_price"].groupby(["time"])["space"].mean()[cols].values.reshape(24, 4), columns=["0min", "15min", "30min", "45min"], index=[f"{i} hour" for i in range(24)])

st.markdown("### 边际火电竞价空间（MWH）")
space_df = st.data_editor(space_df)

st.markdown("#### 边际机组报价范围")
col1, col2 = st.columns(2)

with col1:
    margin_min = st.number_input('边际机组报价范围的最小值（元/MWH）', min_value=0, max_value=1000, value=0, step=1)

with col2:
    margin_max = st.number_input('边际机组报价范围的最大值（元/MWH）', min_value=0, max_value=1000, value=0, step=1)

### 价格预测

if st.button("提交价格预测", key="提交价格预测"):
    if margin_min == margin_max == 0:
        st.session_state["predict"]["price"] = daily_predict_global_price(space_df.values.reshape(-1))
    else:
        st.session_state["predict"]["price"] = daily_predict_global_price_with_margin_declare(space_df.values.reshape(-1), margin_min, margin_max)
    plot_daily_price_prediction(st.session_state["predict"]["price"])


### 策略模拟

st.markdown("# 报价策略模拟")
declare_df = pd.DataFrame([[100, 200], [100, 250], [200, 300]], index=["电量包1", "电量包2", "电量包3"], columns=["电力（MW）", "电价（元/MWH）"])
declare_df = st.data_editor(declare_df)
total_quantity = declare_df["电力（MW）"].sum().item()

if st.button("提交策略模拟", key="提交策略模拟"):
    day_simulate_trade_price, day_simulate_trade_quantity = simulate(st.session_state["predict"]["price"], declare_df["电价（元/MWH）"], declare_df["电力（MW）"])
    day_simulate_trade_rate = (day_simulate_trade_quantity / (total_quantity * 24)).reshape(-1)
    day_simulate_trade_sale = np.round(day_simulate_trade_price * day_simulate_trade_quantity / 1e4, 1)
    avg_trade_sale = round(np.mean(day_simulate_trade_sale), 1)
    trade_sale_q5 = round(np.percentile(day_simulate_trade_sale, 5), 1)
    trade_sale_q25 = round(np.percentile(day_simulate_trade_sale, 25), 1)
    trade_sale_q50 = round(np.percentile(day_simulate_trade_sale, 50), 1)
    trade_sale_q75 = round(np.percentile(day_simulate_trade_sale, 75), 1)
    trade_sale_q95 = round(np.percentile(day_simulate_trade_sale, 95), 1)

    st.markdown(f"收益期望：{avg_trade_sale}万元")
    st.markdown(f"50%概率收益为 {trade_sale_q25} ~ {trade_sale_q75}万元")
    st.markdown(f"90%概率收益为 {trade_sale_q5} ~ {trade_sale_q95}万元")

    plot_simulate_trade_rate(day_simulate_trade_rate)
    plot_simulate_trade_sale(day_simulate_trade_sale)


### 报价策略优化
st.markdown("# 报价策略优化")

input_total_quantity = st.number_input("", min_value=0, max_value=5000, value=400)
if st.button("提交策略优化", key="提交策略优化"):
    with st.spinner("优化策略搜索中..."):
        mu, std, search_prices = search(input_total_quantity)
    best_idx = np.argmax(mu)
    st.markdown(f"期望收益策略报价：{int(search_prices[best_idx])}（元/MWH）")
    st.markdown(f"期望收益：{int(mu[best_idx])}万元")
    f = plt.figure()
    plt.scatter(mu, std)
    st.pyplot(f)
    plt.close(f)
