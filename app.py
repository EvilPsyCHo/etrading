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
import json
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
ROOT = Path(__file__).parent
try:
    res = api.test()
    code = json.loads(res.content)["status"]
    if json.loads(res.content)["status"] == 200:
        st.markdown("连接后台数据库与模拟交易接口成功")
    else:
        st.markdown(f"连接“出清系统”失败，部分功能无法使用，请排查，ERROR CODE {code}")
except:
    st.markdown(f"连接“出清系统”失败，部分功能无法使用，请排查")
##### Initialization

def load_simulation_info(data_dir):
    infos = []
    for info_path in Path(data_dir).rglob("info.pkl"):
        with open(info_path, "rb") as f:
            info = pickle.load(f)
        infos.append(info)
    infos = pd.DataFrame(infos)
    return infos

infos = load_simulation_info(ROOT / "data")
valid_infos = infos[infos.valid_exp_rounds >= 4]
st.session_state["analysis_eng_id"] = st.selectbox("选择模拟试验（仅显示有效试验轮次大于5次的试验）", options=valid_infos.eng_id.tolist(), index=0)
with open(f"./data/{st.session_state['analysis_eng_id']}/process.pkl", "rb") as f:
    st.session_state["data"] = pickle.load(f)
with open(f"./data/{st.session_state['analysis_eng_id']}/analysis.pkl", "rb") as f:
    st.session_state["analysis"] = pickle.load(f)

st.session_state["unittype_map"] = {"101": "火电", "201": "水电", "301": "风电", "302": "光伏"}
st.session_state["data"]["units"] = st.session_state["data"]["units"][st.session_state["data"]["units"].unitType.isin(st.session_state["unittype_map"])]
st.session_state["unit_id_to_name"] = st.session_state["data"]["units"].set_index("unitId")["unitName"].to_dict()
st.session_state["unit_name_to_id"] = {v:k for k,v in st.session_state["unit_id_to_name"].items()}
st.session_state["predict"] = {"price": None}


def plot_unit_count():
    # Data for the pie chart
    unit_count = st.session_state["data"]["units"].unitType.value_counts()
    sizes = unit_count.values
    labels = list(map(lambda x: st.session_state["unittype_map"].get(x), unit_count.index.tolist()))
    colormap = {"#ff9999": "火电", "#66b3ff": "水电", "#99ff99": "风电", "#ffcc99": "光伏"}
    colormap = {v:k for k,v in colormap.items()}
    colors = list(map(lambda x: colormap.get(x), labels))
    
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(sizes, labels=[f'{label}: {size}' for label, size in zip(labels, sizes)], colors=colors, startangle=140)
    ax.set_title("机组类型数量")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)



def plot_unit_capacity():
    # Data for the pie chart
    unit_cap = st.session_state["data"]["units"].groupby("unitType")["unitCap"].sum()
    sizes = unit_cap.values.astype(int)
    labels = list(map(lambda x: st.session_state["unittype_map"].get(x), unit_cap.index.tolist()))
    colormap = {"#ff9999": "火电", "#66b3ff": "水电", "#99ff99": "风电", "#ffcc99": "光伏"}
    colormap = {v:k for k,v in colormap.items()}
    colors = list(map(lambda x: colormap.get(x), labels))

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(sizes, labels=[f'{label}: {size}' for label, size in zip(labels, sizes)], colors=colors, startangle=140)
    ax.set_title('机组装机容量（MW）')
    fig.tight_layout()
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
st.markdown("## 市场概览")
st.markdown("### 机组数量与装机容量")
col1_1, col1_2 = st.columns([1, 1])
with col1_1:
    plot_unit_count()

with col1_2:
    plot_unit_capacity()

st.markdown("### 系统负荷与新能源出力预测")
plot_system_load_and_new_energy()
