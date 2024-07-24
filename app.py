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

from etrading.api import DataAPI


##### Config 
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")
sns.set_theme(style="dark")
font_path = './assets/SimSun.ttf'
font_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = font_prop.get_name()
cols = [f"p{i}" for i in range(1, 97)]
api = DataAPI()

##### Initialization

def get_analysis_eng_ids():
    eng_ids = [subdir.stem for subdir in Path("./output").iterdir() if subdir.is_dir()]
    return eng_ids
st.session_state["analysis_eng_id"] = st.selectbox("模拟试验分析", options=get_analysis_eng_ids(), index=0)
with open(f"./output/{st.session_state['analysis_eng_id']}/data.pkl", "rb") as f:
    st.session_state["data"] = pickle.load(f)
with open(f"./output/{st.session_state['analysis_eng_id']}/analysis.pkl", "rb") as f:
    st.session_state["analysis"] = pickle.load(f)

st.session_state["unittype_map"] = {"101": "火电", "201": "水电", "301": "风电", "302": "光伏"}
st.session_state["data"]["units"] = st.session_state["data"]["units"][st.session_state["data"]["units"].unitType.isin(st.session_state["unittype_map"])]
st.session_state["unit_id_to_name"] = st.session_state["data"]["units"].set_index("unitId")["unitName"].to_dict()
st.session_state["predict"] = {"price": None}


def plot_unit_count():
    # Data for the pie chart
    unit_count = st.session_state["data"]["units"].unitType.value_counts()
    sizes = unit_count.values
    labels = list(map(lambda x: st.session_state["unittype_map"].get(x), unit_count.index.tolist()))
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(sizes, labels=[f'{label}: {size}' for label, size in zip(labels, sizes)], colors=colors, startangle=140)
    ax.set_title("机组类型数量")
    st.pyplot(fig)
    plt.close(fig)



def plot_unit_capacity():
    # Data for the pie chart
    unit_cap = st.session_state["data"]["units"].groupby("unitType")["unitCap"].sum()
    sizes = unit_cap.values
    labels = list(map(lambda x: st.session_state["unittype_map"].get(x), unit_cap.index.tolist()))
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(sizes, labels=[f'{label}: {size}' for label, size in zip(labels, sizes)], colors=colors, startangle=140)
    ax.set_title('机组装机容量（MW）')
    st.pyplot(fig)
    plt.close(fig)



def plot_system_load_and_new_energy():
    system_load = st.session_state["data"]["busload"].groupby("time")["power"].sum()
    new_energy = st.session_state["data"]["new_energy_plan"].groupby("time")["power"].sum()
    cols = [f"p{i}" for i in range(1, 97)]
    fig, ax = plt.subplots(figsize=(16, 8))

    ax.plot(system_load[cols], label="系统负荷（MW）")
    _ = ax.set_xticks(ax.get_xticks()[::4])
    
    ax.plot(new_energy[cols], label="新能源预测出力（MW）")

    # ax.set_title("System Load vs. New Energy Forecasting")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


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
    sns.kdeplot(st.session_state["analysis"]["corr_space_and_avg_price"], ax=ax, fill=True)
    plt.title("（日内）火电竞价空间与成交价格相关系数分布")
    st.pyplot(fig)
    plt.close(fig)


def plot_unit_declare_vs_trade_avg_price_corr():
    ax = st.session_state["analysis"]["unit_declare_intercept_corr"].fillna(0.000).sort_values().plot()
    fig = ax.figure
    plt.xlabel("机组")
    plt.ylabel("相关系数")
    plt.title("机组报价与平均成交价格平台相关系数")
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

with st.sidebar:
    prompt = f'''### 陕西电力现货模拟策略分析\n

工程ID: {st.session_state["data"]["eng_id"]}\n
机组数量: {len(st.session_state["data"]["units"])}\n
系统负荷(MW): {round(st.session_state["data"]["busload"].power.sum() / 96, 2)}\n
新能源出力(MW): {round(st.session_state["data"]["new_energy_plan"].power.sum() / 96,2)}\n
'''
    st.markdown(prompt)


st.markdown("## 市场概览")
st.markdown("### 机组数量与装机容量")
col1_1, col1_2 = st.columns(2)
with col1_1:
    plot_unit_count()

with col1_2:
    plot_unit_capacity()

st.markdown("### 系统负荷与新能源出力预测")
plot_system_load_and_new_energy()


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

st.markdown("### 边际机组报价与平均成交价格关系")
plot_unit_declare_vs_trade_avg_price_corr()


margin_units = st.session_state["analysis"]["unit_declare_intercept_corr"].sort_values(ascending=False).head(20).index.tolist()
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
