import argparse
import pprint
import time
import pandas as pd
import matplotlib.pyplot as plt

from amp.inference.inference import HYDRAmpGenerator

parser = argparse.ArgumentParser(description='Run template generation on multiple sequences')
parser.add_argument('-mp', '--model_path', type=str, required=True)
parser.add_argument('-dp', '--decomposer_path', type=str, required=True)

parser.add_argument('-m', '--mode', type=str, choices=['amp', 'nonamp'], default='amp')
parser.add_argument('-n', '--n_target', default=1000, help='How many accepted sequences must be obtained')
parser.add_argument('-e', '--seed', type=int, default=None, help='Generation seed')

args = parser.parse_args()

generator = HYDRAmpGenerator(model_path=args.model_path, decomposer_path=args.decomposer_path)

# cold start
generator.unconstrained_generation(mode=args.mode, n_target=10, seed=None)

times = {}
for n_target in [10, 100, 500, 1000, 5000]:
    cur_times = []
    print(n_target)
    for _ in range(20):
        start = time.perf_counter()
        generator.unconstrained_generation(mode=args.mode, n_target=n_target)
        end = time.perf_counter()
        cur_times.append(end - start)
    times[n_target] = cur_times

ax = pd.DataFrame.from_dict(times).boxplot()
plt.title(f'Unconstrained time for {args.mode}')
plt.show()
