import argparse
import pprint
import time

import matplotlib.pyplot as plt
import pandas as pd
from amp.inference.inference import HYDRAmpGenerator
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Run template generation on multiple sequences')
parser.add_argument('-mp', '--model_path', type=str, required=True)
parser.add_argument('-dp', '--decomposer_path', type=str, required=True)

parser.add_argument('-m', '--mode', type=str, choices=['improve', 'worsen'])
parser.add_argument('-n', '--n_target', default=1000, help='How many accepted sequences must be obtained')
parser.add_argument('-e', '--seed', type=int, default=None, help='Generation seed')
parser.add_argument('-a', '--data', type=str, default=None, help='Data')

args = parser.parse_args()

generator = HYDRAmpGenerator(model_path=args.model_path, decomposer_path=args.decomposer_path)
data = pd.read_csv(args.data)['Sequence']

times = {}
for n_attempts in tqdm([50, 100, 200, 400]):
    for n_seq in tqdm([5, 10, 20, 30, 50]):
        cur_times = []
        data_lim = data.iloc[:n_seq]
        for _ in range(10):
            start = time.perf_counter()
            generator.template_generation(padded_sequences=data_lim, constraint=args.mode, n_attempts=n_attempts)
            end = time.perf_counter()
            cur_times.append(end - start)
        times[(n_attempts, n_seq)] = cur_times
df = pd.DataFrame.from_dict(times)
df.columns.names = ['n_attempts', 'n_seq']
df = df.stack(level=0).reset_index(level=0, drop=True).reset_index()
df.boxplot(by='n_attempts')
plt.show()
