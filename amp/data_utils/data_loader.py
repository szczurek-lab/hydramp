import random
from collections import defaultdict
from typing import Tuple, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from amp.data_utils.sequence import to_one_hot, pad


class AMPDataManager:

    def __init__(
            self,
            positive_filepath: str,
            negative_filepath: str,
            min_len: int,
            max_len: int,
    ):
        self.positive_data = pd.read_csv(positive_filepath)
        self.negative_data = pd.read_csv(negative_filepath)

        self.min_len = min_len
        self.max_len = max_len

    def _filter_by_length(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = (df['Sequence'].str.len() >= self.min_len) & (df['Sequence'].str.len() <= self.max_len)
        return df.loc[mask]

    def _filter_data(self):
        return self._filter_by_length(self.positive_data), self.negative_data

    @staticmethod
    def _get_probs(peptide_lengths: List[int]) -> Dict[int, float]:
        probs = defaultdict(lambda: 1)
        for length in peptide_lengths:
            probs[length] += 1
        return {k: round(v / len(peptide_lengths), 4) for k, v in probs.items()}

    @staticmethod
    def _draw_subsequences(df, new_lengths):
        random.seed(44)
        new_lengths.sort(reverse=True)
        df = df.sort_values(by="Sequence length", ascending=False)

        d = []
        for row, new_length in zip(df.itertuples(), new_lengths):
            seq = row[2]
            curr_length = row[3]
            if new_length > curr_length:
                new_seq = seq
            elif new_length == curr_length:
                new_seq = seq
            else:
                begin = random.randrange(0, int(curr_length) - new_length)
                new_seq = seq[begin:begin + new_length]
            d.append(
                {
                    'Name': row[1],
                    'Sequence': new_seq,
                }
            )
        new_df = pd.DataFrame(d)
        return new_df

    def _equalize_data(self, positive_data: pd.DataFrame, negative_data: pd.DataFrame, balanced_classes: bool = True):
        positive_seq = positive_data['Sequence'].tolist()
        positive_lengths = [len(seq) for seq in positive_seq]

        negative_seq = negative_data['Sequence'].tolist()
        negative_lengths = [len(seq) for seq in negative_seq]
        negative_data.loc[:, "Sequence length"] = negative_lengths

        probs = self._get_probs(positive_lengths)
        k = len(positive_lengths) if balanced_classes else len(negative_lengths)

        new_negative_lengths = random.choices(list(probs.keys()), probs.values(), k=k)
        negative_data_distributed = self._draw_subsequences(self.negative_data, new_negative_lengths)
        return positive_data, negative_data_distributed

    def plot_distributions(self, equalize: bool = True):
        if equalize:
            pos_dataset, neg_dataset = self.get_data()
        else:
            pos_dataset, neg_dataset = self.positive_data, self.negative_data
        sns.set(color_codes=True)
        # TODO: figure out where this functionality should be. Goal is to plot distribution before and after baladancing
        positive_seq = pos_dataset['Sequence'].tolist()
        positive_lengths = [len(seq) for seq in positive_seq]

        negative_seq = neg_dataset['Sequence'].tolist()
        negative_lengths = [len(seq) for seq in negative_seq]

        fig, (ax2, ax3) = plt.subplots(figsize=(12, 6), ncols=2)
        sns.distplot(positive_lengths, ax=ax2)
        sns.distplot(negative_lengths, ax=ax3)
        ax2.set_title("Positive")
        ax3.set_title("Negative")

        plt.show()

    def _join_datasets(self, pos_dataset: pd.DataFrame, neg_dataset: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        pos_dataset.loc[:, 'Label'] = 1
        neg_dataset.loc[:, 'Label'] = 0
        merged = pd.concat([pos_dataset, neg_dataset])
        x = np.asarray(merged['Sequence'].tolist())
        y = np.asarray(merged['Label'].tolist())
        x = pad(to_one_hot(x))
        return x, y

    def get_data(self, balanced: bool = True):
        pos_dataset, neg_dataset = self._filter_data()
        return self._equalize_data(pos_dataset, neg_dataset, balanced_classes=balanced)

    def get_merged_data(self, balanced: bool = True):
        pos_dataset, neg_dataset = self.get_data(balanced=balanced)
        return self._join_datasets(pos_dataset, neg_dataset)
