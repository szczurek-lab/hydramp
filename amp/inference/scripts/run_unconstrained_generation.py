import argparse
import pprint

from amp.inference.inference import HydrAMPGenerator

parser = argparse.ArgumentParser(description='Run template generation on multiple sequences')
parser.add_argument('-mp', '--model_path', type=str, required=True)
parser.add_argument('-dp', '--decomposer_path', type=str, required=True)
parser.add_argument('-m', '--mode', type=str, choices=['amp', 'nonamp'], default='amp')
parser.add_argument('-s', '--softmax', action='store_true')
parser.add_argument('-n', '--n_target', default=100, help='How many accepted sequences must be obtained')
parser.add_argument('-e', '--seed', type=int, default=None, help='Generation seed')
parser.add_argument('--filter-out-cysteins', action='store_true')
parser.add_argument('--filter-out-known-amps', action='store_true')
parser.add_argument('--filter-out-hydrophobic-clusters', action='store_true')
parser.add_argument('--filter-out-positive-clusters', action='store_true')

args = parser.parse_args()

generator = HydrAMPGenerator(model_path=args.model_path, decomposer_path=args.decomposer_path, softmax=args.softmax)
generation_result = generator.unconstrained_generation(mode=args.mode,
                                                       n_target=args.n_target,
                                                       seed=args.seed,
                                                       filter_out=True,
                                                       filter_out_cysteins=args.filter_out_cysteins,
                                                       filter_out_hydrophobic_clusters=args.filter_out_hydrophobic_clusters,
                                                       filter_out_positive_clusters=args.filter_out_positive_clusters,
                                                       filter_out_known_amps=args.filter_out_known_amps)

pprint.pprint(generation_result)
