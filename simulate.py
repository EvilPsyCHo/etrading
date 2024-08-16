from etrading.optimize import Environment
from etrading.api import DataAPI
from pathlib import Path
import pickle
import pandas as pd

def load_simulation_info(data_dir):
    infos = []
    for info_path in Path(data_dir).rglob("info.pkl"):
        with open(info_path, "rb") as f:
            info = pickle.load(f)
        infos.append(info)
    infos = pd.DataFrame(infos)
    return infos


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--id")
    parser.add_argument("--target_rounds", type=int)
    parser.add_argument("--min", type=int)
    parser.add_argument("--max", type=int)
    

    args = parser.parse_args()

    api = DataAPI()

    assert args.max > args.min
    assert args.max > 0
    assert args.target_rounds >= 1

    ROOT = Path(__file__).parent
    infos = load_simulation_info(ROOT / "data")


    try:
        res = api.test()
        code = json.loads(res.content)["status"]
        if code == 200:
            print("出清模拟平台连接成功")
        else:
            raise ConnectionError(f"连接“出清系统”失败，部分功能无法使用，请排查，ERROR CODE {code}")
    except Exception as e:
        raise ConnectionError(f"连接“出清系统”失败，部分功能无法使用，请排查，ERROR CODE {code}.\n{e}")

    if args.id in infos.eng_id.to_list():
        print("load existing simulation environment ..")
        env = Environment.load(args.id)
    else:
        print("create existing simulation environment ..")
        env = Environment.create(args.id)
    mu = (args.min + args.max) / 2
    bias = (args.max - args.min) / 4
    env.random_run(args.target_rounds, mu, bias)
