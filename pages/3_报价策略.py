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


def display_sidebar():
    with st.sidebar:
        prompt = f'''### 陕西电力现货模拟策略分析\n

    模拟试验: {st.session_state["data"]["eng_id"]}\n
    机组数量: {len(st.session_state["data"]["units"])}\n
    系统平均负荷: {round(st.session_state["data"]["busload"].power.sum() / 96, 2)} (MW)\n
    新能源平均出力预测: {round(st.session_state["data"]["new_energy_plan"].power.sum() / 96,2)} (MW)\n
    '''
        st.markdown(prompt)

cols = [f"p{i}" for i in range(1, 97)]


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
    plt.xlabel("成交率")
    plt.ylabel("频率")
    st.pyplot(fig)
    plt.close(fig)


def plot_simulate_trade_sale(trade_sale):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(trade_sale, ax=ax)
    plt.title("收益分布（万元）")
    plt.xlabel("收益（万元）")
    plt.ylabel("频率")
    st.pyplot(fig)
    plt.close(fig)


def plot_search_mu_std(mu, std):
    best_idx = np.argmax(mu)
    f = plt.figure(figsize=(12, 6))
    plt.scatter(mu, std)
    plt.scatter([mu[best_idx]], [std[best_idx]], color="red", label="最优报价")
    plt.xlabel("期望收益（万元）")
    plt.ylabel("收益波动")
    plt.title("期望收益 vs. 收益波动")
    plt.legend()
    st.pyplot(f)
    plt.close(f)


def plot_search_price_mu(price, mu):
    best_idx = np.argmax(mu)
    f = plt.figure(figsize=(12, 6))
    plt.scatter(price, mu)
    plt.scatter([price[best_idx]], [mu[best_idx]], color="red", label="最优报价")
    plt.xlabel("报价（元/MW）")
    plt.ylabel("期望收益（万元）")
    plt.legend()
    plt.title("报价 vs. 期望收益")
    st.pyplot(f)
    plt.close(f)


def plot_search_price_std(price, std, best_idx):
    f = plt.figure(figsize=(12, 6))
    plt.scatter(price, std)
    plt.scatter([price[best_idx]], [std[best_idx]], color="red", label="最优报价")
    plt.xlabel("报价（元/MW）")
    plt.ylabel("收益波动")
    plt.title("报价 vs. 收益波动")
    plt.legend()
    st.pyplot(f)
    plt.close(f)


def display_search_result(price, mu, std):
    best_idx = np.argmax(mu)
    col1_1, col1_2 = st.columns(2)
    with col1_1:
        plot_search_price_mu(price, mu)
    with col1_2:
        plot_search_price_std(price, std, best_idx)
    plot_search_mu_std(mu, std)


### 策略模拟
display_sidebar()
st.markdown("# 报价策略模拟")
declare_df = pd.DataFrame([[100, 100], [100, 200], [200, 300]], index=["电量包1", "电量包2", "电量包3"], columns=["电力（MW）", "电价（元/MWH）"])
declare_df = st.data_editor(declare_df)
total_quantity = declare_df["电力（MW）"].sum().item()

if st.button("提交策略模拟", key="提交策略模拟"):
    try:
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
    except:
        st.write("请先完成价格预测再进行策略模拟。")


### 报价策略优化
st.markdown("# 报价策略优化")
input_total_quantity = st.number_input("电力（MW）", min_value=0, max_value=5000, value=400)
if st.button("搜索优化策略", key="搜索优化策略"):
    try:
        with st.spinner("优化策略搜索中..."):
            mu, std, search_prices = search(input_total_quantity)
        best_idx = np.argmax(mu)
        st.markdown(f"最优报价策略：{int(search_prices[best_idx])}（元/MWH）")
        st.markdown(f"最优期望收益：{int(mu[best_idx])}万元")
        display_search_result(search_prices, mu, std)
    except:
        st.write("请先完成价格预测再进行策略模拟。")
    