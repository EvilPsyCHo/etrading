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
import time
from etrading.api import DataAPI
import json
import subprocess


api = DataAPI()
ROOT = Path(__file__).parent.parent

st.markdown("## 创建模拟试验")
simulate_eng_id = st.text_input("工程ID")
simulate_target_round = st.number_input(label="目标模拟次数（建议10次以上）", min_value=1, max_value=100, value=10)
simulate_min = st.number_input(label="报价区间（下限）", min_value=1, max_value=500, value=200)
simulate_max = st.number_input(label="报价区间（上限）", min_value=1, max_value=500, value=300)
if st.button("模拟试验"):
    with st.spinner("模拟试验启动中..."):
        try:
            res = api.test()
            assert json.loads(res.content)["status"] == 200
            process = subprocess.Popen([
            "python", "simulate.py",
            "--id", simulate_eng_id,
            "--target_rounds", str(simulate_target_round),
            "--min", str(simulate_min),
            "--max", str(simulate_max)
        ])
            time.sleep(5)
            retcode = process.poll()
            if retcode is None:
                estimate_time = simulate_target_round
                st.write(f"模拟试验启动成功，预计耗时{simulate_target_round}小时")
            else:
                st.write(f"模拟试验启动失败，请检查工程ID是否正确输入，目标模拟次数是否设置正确，或检查后台日志")
        except:
            st.write(f"无法连接交易出清接口，请排查。")


if st.button("模型训练&数据分析"):
    with st.spinner("模型训练 & 数据分析启动中..."):
        try:
            process = subprocess.Popen([
            "python", "analysis.py",
            "--eng_id", simulate_eng_id,
        ])
            time.sleep(5)
            retcode = process.poll()
            if retcode is None:
                estimate_time = simulate_target_round
                st.write(f"模型启动成功，预计耗时{simulate_target_round}分钟")
            else:
                st.write(f"模型训练启动失败，请检查工程ID是否正确输入，或检查后台日志")
        except Exception as e:
            st.write(f"模型训练启动失败，请检查工程ID是否正确输入，或检查后台日志")
            st.write(e)


def load_simulation_info(data_dir):
    infos = []
    for info_path in Path(data_dir).rglob("info.pkl"):
        with open(info_path, "rb") as f:
            info = pickle.load(f)
        infos.append(info)
    infos = pd.DataFrame(infos)
    return infos

st.markdown("## 模拟试验信息")
st.dataframe(load_simulation_info(ROOT / "data"))



st.markdown("## 创建工程")
create_eng_name = st.text_input("创建工程名称")
create_eng_remark = st.text_input("创建工程备注信息")
if st.button("创建工程"):
    try:

        eng_id = api.create_eng(create_eng_name, create_eng_remark)["data"]
        st.write(f"创建工程成功，工程ID：{eng_id}")
        with st.spinner("加载工程数据..."):
            res = api.load_eng_base_data(eng_id)
            st.write(res)
    except:
        st.markdown("无法连接交易出清接口，请排查。")



st.markdown("## 工程信息")
try:
    eng_info = pd.DataFrame(api.get_eng_info()["data"])
    st.dataframe(eng_info[["engName", "id", "remark", "createTime"]])
except:
    st.markdown("无法连接交易出清接口，请排查。")
