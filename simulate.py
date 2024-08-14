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
    parser = argparse.ArgumentParser()
    parser.add_argument("--id")
    parser.add_argument("--target_rounds", type=int)
    parser.add_argument("--min", type=int)
    parser.add_argument("--max", type=int)
    

    args = parser.parse_args()

    assert args.max > args.min
    assert args.max > 0
    assert args.target_rounds >= 1

    ROOT = Path(__file__).parent
    infos = load_simulation_info(ROOT / "data")
    if args.id in infos.eng_id.to_list():
        print("load existing simulation environment ..")
        env = Environment.load(args.id)
    else:
        print("create existing simulation environment ..")
        env = Environment.create(args.id)
    mu = (args.min + args.max) / 2
    bias = (args.max - args.min) / 4
    env.random_run(args.target_rounds, mu, bias)
