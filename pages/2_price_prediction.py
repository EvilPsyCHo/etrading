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
cols = [f"p{i}" for i in range(1, 97)]

def display_sidebar():
    with st.sidebar:
        prompt = f'''### 陕西电力现货模拟策略分析\n

    模拟试验: {st.session_state["data"]["eng_id"]}\n
    机组数量: {len(st.session_state["data"]["units"])}\n
    系统平均负荷: {round(st.session_state["data"]["busload"].power.sum() / 96, 2)} (MW)\n
    新能源平均出力预测: {round(st.session_state["data"]["new_energy_plan"].power.sum() / 96,2)} (MW)\n
    '''
        st.markdown(prompt)


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


### Streamlit layout
display_sidebar()

### 价格分析
st.markdown("# 价格预测模型")

#### 模型输入
space_df = pd.DataFrame(st.session_state["analysis"]["trade_avg_price"].groupby(["time"])["space"].mean()[cols].values.reshape(24, 4), columns=["0min", "15min", "30min", "45min"], index=[f"{i} hour" for i in range(24)])

st.markdown("### 边际火电竞价空间（MWH）")
space_df = st.data_editor(space_df)

st.markdown("#### 边际机组报价范围（选填）")
col1, col2 = st.columns(2)

with col1:
    margin_min = st.number_input('边际机组报价范围的最小值（元/MWH）', min_value=0, max_value=1000, value=0, step=1)

with col2:
    margin_max = st.number_input('边际机组报价范围的最大值（元/MWH）', min_value=0, max_value=1000, value=0, step=1)
margin_max = max(margin_max, margin_min)
#### 价格预测

if st.button("提交价格预测", key="提交价格预测"):
    
    if margin_min == margin_max == 0:
        st.session_state["predict"]["price"] = daily_predict_global_price(space_df.values.reshape(-1))
    else:
        st.session_state["predict"]["price"] = daily_predict_global_price_with_margin_declare(space_df.values.reshape(-1), margin_min, margin_max)
    st.markdown("蓝色曲线为预测价格期望，灰色区域为价格50%概率区间。")
    plot_daily_price_prediction(st.session_state["predict"]["price"])
