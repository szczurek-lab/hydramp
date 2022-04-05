import argparse
import pprint

from amp.inference.filtering import get_filtering_mask
from amp.inference.inference import HydrAMPGenerator

parser = argparse.ArgumentParser(description='Run analogue generation on multiple sequences')
parser.add_argument('-mp', '--model_path', type=str, required=True)
parser.add_argument('-dp', '--decomposer_path', type=str, required=True)
parser.add_argument('-f', '--filtering-criteria', type=str, choices=['improvement', 'discovery'])
parser.add_argument('-s', '--sequences', nargs='+', help='Sequences to improve upon', required=True)
parser.add_argument('-so', '--softmax', action='store_true')
parser.add_argument('-t', '--temperature', type=float, default=5.0, help='Creativity parameter')
parser.add_argument('-n', '--n_attempts', type=int, default=100, help=' Improving attempts per peptide')
parser.add_argument('--filter-out-cysteins', action='store_true')
parser.add_argument('--filter-out-known-amps', action='store_true')
parser.add_argument('--filter-out-hydrophobic-clusters', action='store_true')
parser.add_argument('--filter-out-positive-clusters', action='store_true')

args = parser.parse_args()
generator = HydrAMPGenerator(model_path=args.model_path, decomposer_path=args.decomposer_path, softmax=args.softmax)

generation_result = generator.analogue_generation(sequences=args.sequences,
                                                  seed=42,
                                                  filtering_criteria=args.filtering_criteria,
                                                  n_attempts=args.n_attempts,
                                                  temp=args.temperature,
                                                  filter_out_cysteins=args.filter_out_cysteins,
                                                  filter_out_hydrophobic_clusters=args.filter_out_hydrophobic_clusters,
                                                  filter_out_positive_clusters=args.filter_out_positive_clusters,
                                                  filter_out_known_amps=args.filter_out_known_amps)

pprint.pprint(generation_result)
