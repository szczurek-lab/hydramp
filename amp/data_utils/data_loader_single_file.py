import random
from collections import defaultdict
from typing import Tuple, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from amp.data_utils.sequence import to_one_hot, pad


class AMPDataManagerSingleFile:

    def __init__(
            self,
            filepath: str,
            min_len: int,
            max_len: int,
    ):
        self.data = pd.read_csv(filepath)

        self.min_len = min_len
        self.max_len = max_len

    def _filter_by_length(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = (df['Sequence'].str.len() >= self.min_len) & (df['Sequence'].str.len() <= self.max_len)
        return df.loc[mask]

    def _filter_data(self):
        return self._filter_by_length(self.data)

    def get_data(self) -> Tuple[np.ndarray, int]:
        """ returns
        - encoded sequences
        - num of filtered out sequences"""
        dataset = self._filter_data()
        x = np.asarray(dataset['Sequence'].tolist())
        return pad(to_one_hot(x)), len(self.data) - len(dataset)
