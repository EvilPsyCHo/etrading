from etrading.optimize import Environment
from etrading.api import DataAPI


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

    env = Environment.create(args.id)
    mu = (args.min + args.max) / 2
    bias = (args.max - args.min) / 4
    env.random_run(args.target_rounds, mu, bias)