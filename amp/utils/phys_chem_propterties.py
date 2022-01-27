from typing import Dict, List

import modlamp.analysis as manalysis
import numpy as np

hydrophilic = ['R', 'N', 'D', 'Q', 'E', 'K']
aa_with_positive_charge = ['K', 'R', 'H']
aa_with_negative_charge = ['D', 'E']


def calculate_hydrophobicity(data: List[str]) -> np.ndarray:
    h = manalysis.GlobalAnalysis(data)
    h.calc_H(scale='eisenberg')
    return h.H[0]


def calculate_hydrophobicmoment(data: List[str]) -> np.ndarray:
    h = manalysis.PeptideDescriptor(data, 'eisenberg')
    h.calculate_moment()
    return h.descriptor.flatten()


def calculate_charge(data: List[str]) -> np.ndarray:
    h = manalysis.GlobalAnalysis(data)
    h.calc_charge()
    return h.charge[0]


def calculate_isoelectricpoint(data: List[str]) -> np.ndarray:
    h = manalysis.GlobalDescriptor(data)
    h.isoelectric_point()
    return h.descriptor.flatten()


def calculate_length(sequences: List[str]) -> np.ndarray:
    return np.array([len(seq) for seq in sequences])


def calculate_h_score(sequences: List[str]) -> np.ndarray:
    return np.array([helical_search(seq) for seq in sequences])


def helical_search(sequence: str) -> float:
    def hydro_part(half):
        return np.mean([a in hydrophilic for a in half])

    def charge_part(half):
        return np.mean([a in aa_with_positive_charge for a in half])

    def process_half(half):
        hydro_score = hydro_part(half)
        charge_score = charge_part(half)
        return {'hydro': hydro_score,
                'charge': charge_score}

    def balance_sides(one_half_s, other_half_s, key):
        return np.abs((one_half_s[key] - other_half_s[key]))

    def combine_scores(one_half_s, other_half_s):
        hydro = balance_sides(one_half_s, other_half_s, 'hydro')
        charge = balance_sides(one_half_s, other_half_s, 'charge')

        return hydro + charge

    sequence = sequence.upper()

    graph = [[] for _ in range(18)]
    for amino, angle in zip(sequence, range(0, 100 * len(sequence), 100)):
        a = ((angle % 360) // 20)
        graph[a].append(amino)

    best = 0.0

    for wall in range(9):
        one_half = [x for y in graph[wall:wall + 9] for x in y]
        other_half = [x for y in graph[9 + wall:] + graph[:wall] for x in y]

        one_half_scores = process_half(one_half)
        other_half_scores = process_half(other_half)

        combined_score = combine_scores(one_half_scores, other_half_scores)
        best = combined_score if combined_score > best else best

    return best


def calculate_physchem_prop(sequences: List[str]) -> Dict[str, object]:
    return {
        "length": calculate_length(sequences).tolist(),
        "hydrophobicity": calculate_hydrophobicity(sequences).tolist(),
        "hydrophobic_moment": calculate_hydrophobicmoment(sequences).tolist(),
        "charge": calculate_charge(sequences).tolist(),
        "isoelectric_point": calculate_isoelectricpoint(sequences).tolist(),
        "h_score": calculate_h_score(sequences)
    }
