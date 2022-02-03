import numpy as np
import pandas as pd


def filter_out_known_AMP(positive_path, data: pd.DataFrame) -> pd.DataFrame:
    all_positives = pd.read_csv(positive_path)
    return data[~data.sequence.isin(list(set(all_positives.Sequence) & set(data.sequence)))]


def filter_out_cysteins(data: pd.DataFrame) -> pd.DataFrame:
    return data[~data['sequence'].str.contains('C')]


def check_sequence_for_positive_clusters(sequence: str, threshold: int = 3) -> bool:
    aa_windows = [sequence[i:i + 5] for i in range(len(sequence) - 4)]
    # Calculate the number of positively charged amino acids: R and K in each window
    # If all amino acids in threshold-sized window are positive, discard sequence
    return threshold in [aa_window.count('K') + aa_window.count('R') for aa_window in aa_windows]


def check_sequence_for_repetitive_clusters(sequence: str) -> bool:
    return any(a == b == c for a, b, c in zip(sequence, sequence[1:], sequence[2:]))


def check_sequence_for_hydrophobic_clusters(sequence: str) -> bool:
    ch = False
    for hydrophobic_aa in ['F', 'I', 'L', 'V', 'W', 'M', 'A']:
        ch |= any(a == b == c == hydrophobic_aa for a, b, c in zip(sequence, sequence[1:], sequence[2:]))
    return ch


def filter_out_positive_clusters(data: pd.DataFrame) -> pd.DataFrame:
    return data[~data['sequence'].apply(check_sequence_for_positive_clusters)]


def get_filtering_mask(sequences: np.ndarray, filtering_options):
    accept = True
    if filtering_options['filter_out_cysteins']:
        accept = np.apply_along_axis(sequences, check_sequence_for_repetitive_clusters)
    if filtering_options['filter_out_hydrophobic_clusters']:
        accept = np.apply_along_axis(sequences, check_sequence_for_repetitive_clusters)
    if filtering_options['filter_out_repetitive_clusters']:
        accept = np.apply_along_axis(sequences, check_sequence_for_repetitive_clusters)
    if filtering_options['filter_out_known_amps']:
        accept = np.apply_along_axis(sequences, check_sequence_for_repetitive_clusters)
    return accept


# STRINGENT

def filter_out_aa_clusters(data: pd.DataFrame) -> pd.DataFrame:
    return data[~data['sequence'].apply(lambda x: any(a == b == c for a, b, c in zip(x, x[1:], x[2:])))]


def amino_based_filtering(positive_path, dataset: pd.DataFrame) -> pd.DataFrame:
    dataset = filter_out_known_AMP(positive_path, dataset)
    dataset = filter_out_positive_clusters(dataset)
    dataset = filter_out_aa_clusters(dataset)
    dataset = filter_out_cysteins(dataset)
    return dataset


# RELAXED

def filter_out_hydrophobic_clusters(data: pd.DataFrame) -> pd.DataFrame:
    # Based on Eisenberg scale
    data = data[~data['sequence'].apply(check_sequence_for_hydrophobic_clusters)]
    return data


def amino_based_filtering_relaxed(positive_path, dataset: pd.DataFrame) -> pd.DataFrame:
    dataset = filter_out_known_AMP(positive_path, dataset)
    dataset = filter_out_positive_clusters(dataset)
    dataset = filter_out_hydrophobic_clusters(dataset)
    return dataset
