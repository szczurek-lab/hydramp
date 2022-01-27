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


def filter_out_positive_clusters(data: pd.DataFrame) -> pd.DataFrame:
    return data[~data['sequence'].apply(check_sequence_for_positive_clusters)]


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
    for hydrophobic_aa in ['F', 'I', 'L', 'V', 'W', 'M', 'A']:
         data = data[~data['sequence'].apply(lambda x: any(a == b == c == hydrophobic_aa for a, b, c in zip(x, x[1:], x[2:])))]
    return data

def amino_based_filtering_relaxed(positive_path, dataset: pd.DataFrame) -> pd.DataFrame:
    dataset = filter_out_known_AMP(positive_path, dataset)
    dataset = filter_out_positive_clusters(dataset)
    dataset = filter_out_hydrophobic_clusters(dataset)
    return dataset
