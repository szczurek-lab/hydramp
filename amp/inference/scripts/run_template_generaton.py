import argparse
import pprint

from amp.inference.inference import HYDRAmpGenerator

parser = argparse.ArgumentParser(description='Run template generation on multiple sequences')
parser.add_argument('-mp', '--model_path', type=str, required=True)
parser.add_argument('-dp', '--decomposer_path', type=str, required=True)

parser.add_argument('-c', '--constraint', type=str, choices=['absolute', 'relative'])
parser.add_argument('-m', '--mode', type=str, choices=['improve', 'worsen'])
parser.add_argument('-s', '--sequences', nargs='+', help='<Required> Sequences', required=True)
parser.add_argument('-so', '--softmax',  action='store_true')

parser.add_argument('-n', '--n_attempts', default=100, help=' Improving Attempts per peptide')
parser.add_argument('-e', '--seed', type=int, default=42, help='Generation seed ')

args = parser.parse_args()

generator = HYDRAmpGenerator(model_path=args.model_path, decomposer_path=args.decomposer_path, softmax=args.softmax)
generation_result = generator.template_generation(sequences=args.sequences,
                                                  mode=args.mode,
                                                  constraint=args.constraint,
                                                  n_attempts=args.n_attempts)

pprint.pprint(generation_result)
