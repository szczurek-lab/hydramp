import argparse
import pprint

from amp.inference.inference import HydrAMPGenerator

parser = argparse.ArgumentParser(description='Run analogue generation on multiple sequences')
parser.add_argument('-mp', '--model_path', type=str, required=True)
parser.add_argument('-dp', '--decomposer_path', type=str, required=True)
parser.add_argument('-f', '--filtering-criteria', type=str, choices=['improvement', 'discovery'])
parser.add_argument('-s', '--sequences', nargs='+', help='Sequences to improve upon', required=True)
parser.add_argument('-so', '--softmax',  action='store_true')
parser.add_argument('-t', '--temperature', type=float, default=5.0, help='Creativity parameter')
parser.add_argument('-n', '--n_attempts', type=int, default=100, help=' Improving attempts per peptide')

args = parser.parse_args()

generator = HydrAMPGenerator(model_path=args.model_path, decomposer_path=args.decomposer_path, softmax=args.softmax)
generation_result = generator.analogue_generation(sequences=args.sequences,
                                                  filtering_criteria=args.filtering_criteria,
                                                  n_attempts=args.n_attempts,
                                                  temp=args.temperature)

pprint.pprint(generation_result)
