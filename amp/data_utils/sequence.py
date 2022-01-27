import numpy as np
from keras import preprocessing

STD_AA = list('ACDEFGHIKLMNPQRSTVWY')


def check_if_std_aa(sequence):
    if all(aa in STD_AA for aa in sequence):
        return True
    return False


def check_length(sequence, min_length, max_length):
    if min_length <= len(sequence) <= max_length:
        return True
    return False


def to_one_hot(x):
    alphabet = list('ACDEFGHIKLMNPQRSTVWY')
    classes = range(1, 21)
    aa_encoding = dict(zip(alphabet, classes))
    return [[aa_encoding[aa] for aa in seq] for seq in x]


def pad(x, max_length: int = 25) -> np.ndarray:
    return preprocessing.sequence.pad_sequences(
        x,
        maxlen=max_length,
        padding='post',
        value=0.0
    )
