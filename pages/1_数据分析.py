import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import random
import matplotlib.font_manager as fm
import plotly.graph_objects as go
from pathlib import Path


##### Config 
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")
cols = [f"p{i}" for i in range(1, 97)]


def plot_daily_trade_space_vs_trade_avg_price(round):
    trade_avg_price = st.session_state["analysis"]["trade_avg_price"]
    sub_df = trade_avg_price[trade_avg_price.roundId == round].set_index("time").loc[cols]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    line1 = axes[0].plot(sub_df["space"], label="火电竞价空间", color="red")[0]
    ax0t = axes[0].twinx()
    line2 = ax0t.plot(sub_df["trade_avg_price"], label="平均成交价格")[0]
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax0t.set_xticks(ax0t.get_xticks()[::8])
    ax0t.legend(lines, labels, loc='lower left')

    axes[1].scatter(sub_df["space"], sub_df["trade_avg_price"])
    fig.suptitle(f"exp_{round} 竞价空间与成交价格", size=20)
    st.pyplot(fig)
    plt.close(fig)


def plot_daily_trade_space_vs_trade_avg_price_corr_distribution():
    fig, ax = plt.subplots(figsize=(12, 6))
    print(st.session_state["analysis"].keys(), st.session_state["analysis"]["corr_space_and_avg_price"].shape)
    sns.kdeplot(st.session_state["analysis"]["corr_space_and_avg_price"], ax=ax, fill=True)
    plt.title("（日内）火电竞价空间与成交价格相关系数分布", size=20)
    st.pyplot(fig)
    plt.close(fig)


def plot_unit_declare_vs_trade_avg_price_corr():
    ax = st.session_state["analysis"]["unit_declare_intercept_corr"].fillna(0.000).sort_values().plot(figsize=(16, 8))
    fig = ax.figure
    plt.xlabel("机组")
    plt.ylabel("相关系数")
    plt.title("机组报价与平均成交价格平台相关系数", size=20)
    plt.xticks([])
    st.pyplot(fig)
    plt.close(fig)


def plot_margin_unit_intercept_corr(unitId):
    trade_dfs = st.session_state["analysis"]["trade_dfs"]
    bayes_weight = st.session_state["analysis"]["bayes_weight"]
    margin_df = trade_dfs[(trade_dfs.unitId == unitId) & (trade_dfs.time=="p1")]
    merge_df = bayes_weight.merge(margin_df[["roundId", "declare_price"]], on="roundId")

    # f, ax = plt.subplots()
    g = sns.lmplot(
    data=merge_df[merge_df.lr_coef>0],
    x="declare_price", y="lr_intercept",
    height=5
)
    plt.xlabel("边际机组报价")
    plt.ylabel("出清价格平台")
    plt.title(f"机组-{st.session_state['unit_id_to_name'][unitId]}")
    st.pyplot(g.figure)
    plt.close(g.figure)

def display_sidebar():
    with st.sidebar:
        prompt = f'''### 陕西电力现货模拟策略分析\n

    模拟试验: {st.session_state["data"]["eng_id"]}\n
    机组数量: {len(st.session_state["data"]["units"])}\n
    系统平均负荷: {round(st.session_state["data"]["busload"].power.sum() / 96, 2)} (MW)\n
    新能源平均出力预测: {round(st.session_state["data"]["new_energy_plan"].power.sum() / 96,2)} (MW)\n
    '''
        st.markdown(prompt)


### Streamlit layout
display_sidebar()
st.markdown("## 成交价格影响因素分析")
st.markdown("### （日内）火电竞价空间与平均成交价格关系")
random_rounds = None
if st.button("刷新随机显示", key="刷新（日内）火电竞价空间与平均成交价格关系"):
    random_rounds = random.sample(st.session_state["analysis"]["trade_avg_price"].roundId.unique().tolist(), 4)
if not random_rounds:
    random_rounds = random.sample(st.session_state["analysis"]["trade_avg_price"].roundId.unique().tolist(), 4)

col3_1, col3_2 = st.columns(2)
with col3_1:
    plot_daily_trade_space_vs_trade_avg_price(random_rounds[0])

with col3_2:
    plot_daily_trade_space_vs_trade_avg_price(random_rounds[1])

col4_1, col4_2 = st.columns(2)
with col4_1:
    plot_daily_trade_space_vs_trade_avg_price(random_rounds[2])

with col4_2:
    plot_daily_trade_space_vs_trade_avg_price(random_rounds[3])

st.markdown("### （日内）火电竞价空间与平均成交价格相关系数分布")
plot_daily_trade_space_vs_trade_avg_price_corr_distribution()

st.markdown("### 机组报价与平均成交价格关系")
plot_unit_declare_vs_trade_avg_price_corr()


st.markdown("### 边际机组")
margin_units = st.session_state["analysis"]["unit_declare_intercept_corr"].sort_values(ascending=False).head(10).index.tolist()
margin_corrs = st.session_state["analysis"]["unit_declare_intercept_corr"].sort_values(ascending=False).head(10).values
margin_df = pd.DataFrame({"边际机组": margin_units, "平均成交价格相关系数": margin_corrs})
margin_df["边际机组"] = margin_df["边际机组"].map(lambda x: st.session_state["unit_id_to_name"].get(x))
st.dataframe(margin_df)

random_4_margin_units = None
if st.button("刷新随机显示", key="刷新边际机组报价与平均成交价格关系"):
    random_4_margin_units = random.sample(margin_units, 4)
if not random_4_margin_units:
    random_4_margin_units = random.sample(margin_units, 4)

col3_1, col3_2 = st.columns(2)
with col3_1:
    plot_margin_unit_intercept_corr(random_4_margin_units[0])

with col3_2:
    plot_margin_unit_intercept_corr(random_4_margin_units[1])

col4_1, col4_2 = st.columns(2)
with col4_1:
    plot_margin_unit_intercept_corr(random_4_margin_units[2])

with col4_2:
    plot_margin_unit_intercept_corr(random_4_margin_units[3])
