import argparse
import pprint

from amp.inference.inference import HYDRAmpGenerator

parser = argparse.ArgumentParser(description='Run template generation on multiple sequences')
parser.add_argument('-mp', '--model_path', type=str, required=True)
parser.add_argument('-dp', '--decomposer_path', type=str, required=True)

parser.add_argument('-m', '--mode', type=str, choices=['amp', 'nonamp'])
parser.add_argument('-s', '--softmax',  action='store_true')

parser.add_argument('-n', '--n_target', default=100, help='How many accepted sequences must be obtained')
parser.add_argument('-e', '--seed', type=int, default=None, help='Generation seed')

args = parser.parse_args()

generator = HYDRAmpGenerator(model_path=args.model_path, decomposer_path=args.decomposer_path, softmax=args.softmax)
generation_result = generator.unconstrained_generation(mode=args.mode,
                                                       n_target=args.n_target,
                                                       seed=args.seed, filter_out=False)

pprint.pprint(generation_result)
