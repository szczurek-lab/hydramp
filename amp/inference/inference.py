import itertools
from functools import reduce
from typing import Any, Callable, Dict, Iterable, List, Literal, Tuple, Union

import joblib
import numpy as np
import numpy.ma as ma
import pandas as pd
from keras import layers, models
from sklearn.decomposition import PCA
from tqdm import tqdm

from amp.config import LATENT_DIM
from amp.data_utils import sequence as du_sequence
from amp.inference.filtering import get_filtering_mask
from amp.utils.basic_model_serializer import load_master_model_components
from amp.utils.generate_peptides import translate_peptide
from amp.utils.phys_chem_propterties import calculate_physchem_prop
from amp.utils.seed import set_seed


def _get_comb_iterator(means: List[float], stds: List[float]) -> Iterable:
    iterator = itertools.product(means, stds)
    next(iterator)
    return iterator


def _unroll_to_batch(a: np.array, batch_size: int, combinations: int, attempts: int) -> np.array:
    return a.reshape(batch_size, combinations, attempts, -1)


def _apply_along_axis_improved(func: Callable[[np.ndarray, int], np.ndarray], axis: int, arr: np.array, ):
    return np.array([func(v, i) for i, v in enumerate(np.rollaxis(arr, axis))])


def _dispose_into_bucket(intersection: np.ndarray,
                         prototype_sequences: List[str],
                         generated_sequences: np.ndarray,
                         generated_amp: np.ndarray,
                         generated_mic: np.ndarray,
                         attempts: int,
                         block_size: int) -> List[Dict[str, np.ndarray]]:
    """
    Takes block of generated peptides that corresponds to a single original (input) peptide and filter out based on
     intersection, uniquness
    @param intersection:
    @param generated_sequences:
    @param generated_amp:
    @param generated_mic:
    @param attempts:
    @param block_size:
    @return:
    """
    bucket_indices = np.arange(0, (attempts + 1) * block_size, attempts)
    disposed_generated_sequences = []
    for origin_seq, (left_index, right_index) in zip(prototype_sequences, zip(bucket_indices, bucket_indices[1:])):
        # in case of low temperature it might be the case that an analouge will be actually a peptide we start from
        intersection[left_index:right_index] &= (generated_sequences[left_index:right_index] != origin_seq)
        current_bucket_indices = intersection[left_index:right_index]
        current_bucket_sequences = generated_sequences[left_index:right_index][current_bucket_indices].tolist()
        if not current_bucket_sequences:
            disposed_generated_sequences.append(None)
            continue
        current_amps = generated_amp[left_index:right_index][current_bucket_indices]
        current_mic = generated_mic[left_index:right_index][current_bucket_indices]

        current_bucket_sequences, indices = np.unique(current_bucket_sequences, return_index=True)
        current_amps = current_amps[indices]
        current_mic = current_mic[indices]

        bucket_data = {
            'sequence': current_bucket_sequences,
            'amp': current_amps.tolist(),
            'mic': current_mic.tolist()
        }
        bucket_data.update(calculate_physchem_prop(current_bucket_sequences))
        disposed_generated_sequences.append(bucket_data)
    return disposed_generated_sequences


def slice_blocks(flat_arrays: Tuple[np.ndarray, ...], block_size: int) -> Tuple[np.ndarray, ...]:
    """
    Changes ordering of sequence from [a, b, c, a, b, c, ...] to [a, a, b, b, c, c, ...]
    @param flat_arrays: arrays to process
    @param block_size: number of arrays before stacking
    @return: rearranged arrays into blocks
    """
    return tuple([x.reshape(-1, block_size).T.flatten() for x in flat_arrays])


class HydrAMPGenerator:
    def __init__(self, model_path: str, decomposer_path: str, softmax=False):
        components = load_master_model_components(model_path, return_master=True, softmax=softmax)
        self.model_path = model_path
        self.decomposer_path = decomposer_path
        self._encoder, self._decoder, self._amp_classifier, self._mic_classifier, self.master = components
        self._latent_decomposer: PCA = joblib.load(decomposer_path)
        self._sigma_model = self.get_sigma_model()

    def get_sigma_model(self):
        inputs = layers.Input(shape=(25,))
        z_mean, z_sigma, z = self.master.encoder.output_tensor(inputs)
        return models.Model(inputs, [z_mean, z_sigma, z])

    def get_sigma(self, x):
        _, z_sigma, _ = self._sigma_model.predict(x)
        return np.exp(z_sigma / 2)

    @staticmethod
    def _transpose_sequential_results(res: Dict[str, np.array]):
        transposed_results = {}
        properties = list(res.keys())
        properties.remove('sequence')
        for index, sequence in enumerate(res['sequence']):
            seq_properties = {}
            for prop in properties:
                seq_properties[prop] = res[prop][index]
            transposed_results[sequence] = seq_properties
        return transposed_results

    @staticmethod
    def _encapsulate_sequential_results(res: List[Dict[str, np.array]]):
        transposed_results = []
        for item in res:
            if item is None:
                transposed_results.append(None)
                continue
            item_generated_sequences = []
            properties = list(item.keys())
            for index, sequence in enumerate(item['sequence']):
                seq_properties = {}
                for prop in properties:
                    seq_properties[prop] = item[prop][index]
                item_generated_sequences.append(seq_properties)
            transposed_results.append(item_generated_sequences)
        return transposed_results

    @staticmethod
    def select_peptides(peptides, amp, mic, n_attempts: int = 64, target_positive: bool = True):
        amp = amp.reshape(n_attempts, -1)
        mic = mic.reshape(n_attempts, -1)
        if target_positive:
            mask_amp = amp < 0.8  # ignore those below 0.8
            combined = ma.masked_where(mask_amp, amp)
            good = combined.argmax(axis=0)

        else:
            mask_amp = amp > 0.2  # ignore those above 0.2
            combined = ma.masked_where(mask_amp, amp)
            good = combined.argmin(axis=0)
        peptides = np.array(peptides).reshape(n_attempts, -1).T

        selective_index = list(range(peptides.shape[0]))
        good_peptides = peptides[selective_index, good]
        good_amp = amp.T[selective_index, good]
        good_mic = mic.T[selective_index, good]
        return good_peptides, good_amp, good_mic

    def unconstrained_generation(self,
                                 mode: Literal["amp", "nonamp"] = 'amp',
                                 n_target: int = 100,
                                 seed: int = None,
                                 filter_out: bool = True,
                                 properties: bool = True,
                                 n_attempts: int = 64,
                                 **kwargs) -> Union[List[Dict[str, Any]], List[str]]:
        """
        Sample new peptides from latent space
        @param mode: "amp" or "nonamp"
        @param n_target: how many sequences must  be returned
        @param seed: parameter for reproduction (only with respect to sampled z, final results might differ between
        runs for models with Gumbel-Softmax)
        @param filter_out: if True, generated peptides are
        @param properties: if True, each sequence  is a dictionary with additional properties
        @param n_attempts: how many times a single latent vector is decoded - for normal Softmax models it should be set
        to 1 as every decoding call returns the same peptide.
        @param kwargs: additional boolean arguments for filtering. This include
        - filter_positive_clusters
        - filter_repetitive_clusters
        - filter_cysteins
        - filter_known_amps
        @return: list of peptides (aa sequences) OR
                 list of dicts , each dict encloses a single peptide with its properties
        """
        mode = mode.lower().strip()
        assert mode in ['amp', 'nonamp'], "Generation mode not recognised"
        mode = (mode == 'amp')

        set_seed(seed)
        amp_input = mic_input = 1 if mode else 0
        min_amp, max_amp = min_mic, max_mic = (0.8, 1.0) if mode else (0.0, 0.2)

        accepted_sequences, accepted_amp, accepted_mic = [], [], []
        pbar = tqdm(total=n_target)
        while len(accepted_sequences) < n_target:

            z = np.random.normal(size=(n_target, LATENT_DIM))
            z = self._latent_decomposer.inverse_transform(z)
            z = np.vstack([z] * n_attempts)
            c_amp = np.repeat(amp_input, z.shape[0]).reshape(-1, 1)
            c_mic = np.repeat(mic_input, z.shape[0]).reshape(-1, 1)

            z_cond = np.hstack([z, c_amp, c_mic])
            candidate = self._decoder.predict(z_cond, verbose=1, batch_size=100000)
            candidate_index_decoded = candidate.argmax(axis=2)
            generated_sequences = [translate_peptide(pep) for pep in candidate_index_decoded]
            generated_amp = self._amp_classifier.predict(candidate_index_decoded).flatten()
            generated_mic = self._mic_classifier.predict(candidate_index_decoded).flatten()

            if not filter_out:
                generated_sequences = np.array(generated_sequences).reshape(n_attempts, -1)
                generated_amp = generated_amp.reshape(n_attempts, -1)
                generated_mic = generated_mic.reshape(n_attempts, -1)
                if mode:  # if mode is amp
                    best = generated_amp.argmax(axis=0)
                else:
                    best = generated_amp.argmin(axis=0)
                selective_index = list(range(generated_sequences.shape[1]))

                accepted_sequences = generated_sequences.T[selective_index, best]
                accepted_amp = generated_amp.T[selective_index, best]
                accepted_mic = generated_mic.T[selective_index, best]
                break

            generated_sequences, generated_amp, generated_mic = self.select_peptides(generated_sequences,
                                                                                     generated_amp,
                                                                                     generated_mic,
                                                                                     n_attempts=n_attempts,
                                                                                     target_positive=mode)

            phych_chem_mask = get_filtering_mask(sequences=generated_sequences, filtering_options=kwargs)
            phych_chem_mask_idx = np.arange(0, len(generated_sequences))
            phych_chem_mask = phych_chem_mask_idx if type(phych_chem_mask) is bool \
                else phych_chem_mask_idx[phych_chem_mask]
            amp_good_indices = np.argwhere((generated_amp >= min_amp) & (generated_amp <= max_amp)).flatten()
            mic_good_indices = np.argwhere((generated_mic >= min_mic) & (generated_mic <= max_mic)).flatten()
            _, is_unique_indices = np.unique(generated_sequences, return_index=True)

            intersection = reduce(np.intersect1d,
                                  (amp_good_indices, mic_good_indices, is_unique_indices, phych_chem_mask))

            if len(intersection) == 0:
                continue

            properly_generated_sequences = pd.DataFrame.from_dict({
                'sequence': generated_sequences[intersection]
            }).set_index(intersection)

            accepted_sequences.extend(properly_generated_sequences['sequence'].tolist())
            accepted_amp.extend(generated_amp[properly_generated_sequences.index].tolist())
            accepted_mic.extend(generated_mic[properly_generated_sequences.index].tolist())
            pbar.update(min(n_target, len(properly_generated_sequences)))
        pbar.close()
        if not properties:
            return list(
                accepted_sequences[:n_target])  # discard surplus sequences to avoid problems with unexpected length

        generated_data = {
            'sequence': np.array(accepted_sequences),
            'amp': np.array(accepted_amp),
            'mic': np.array(accepted_mic)
        }
        sorting_criteria = 'mic' if filter_out else 'amp'
        generated_data = pd.DataFrame.from_dict(generated_data).nlargest(n_target, sorting_criteria).to_dict(
            orient='list')

        generated_data.update(calculate_physchem_prop(generated_data['sequence']))

        return self._encapsulate_sequential_results([generated_data])[0]

    def analogue_generation(self, sequences: List[str], seed: int,
                            filtering_criteria: Literal['improvement', 'discovery'] = 'improvement',
                            n_attempts: int = 100, temp: float = 5.0, **kwargs) -> Dict[Any, Dict[str, Any]]:
        """
        Generates new peptides based on input sequences
        @param sequences: peptides that form a template for further processing
        @param filtering_criteria: 'improvement' if generated peptides should be strictly better than input sequences
        'discovery' if generated sequences should be good enough but not strictly better
        @param n_attempts: how many times a single latent vector is decoded - for normal Softmax models it should be set
        to 1 as every decoding call returns the same peptide.
        @param temp: creativity parameter. Controls latent vector sigma scaling
        @param seed:
        @param kwargs:additional boolean arguments for filtering. This include
        - filter_positive_clusters
        - filter_repetitive_clusters or filter_hydrophobic_clusters
        - filter_cysteins
        - filter_known_amps

        @return: dict, each key corresponds to a single input sequence.
        """
        set_seed(seed)
        filtering_criteria = filtering_criteria.strip().lower()
        assert filtering_criteria == 'improvement' or filtering_criteria == 'discovery', \
            "Unrecognised filtering constraint"

        block_size = len(sequences)
        padded_sequences = du_sequence.pad(du_sequence.to_one_hot(sequences))
        sigmas = self.get_sigma(padded_sequences)
        amp_org = self._amp_classifier.predict(padded_sequences, verbose=1, batch_size=80000)
        mic_org = self._mic_classifier.predict(padded_sequences, verbose=1, batch_size=80000)
        padded_sequences = np.vstack([padded_sequences] * n_attempts).reshape(-1, 25)
        z = self._encoder.predict(padded_sequences, verbose=1, batch_size=80000)
        amp_stacked = np.vstack([amp_org] * n_attempts)
        mic_stacked = np.vstack([mic_org] * n_attempts)

        noise = np.random.normal(loc=0, scale=temp * np.vstack([sigmas] * n_attempts), size=z.shape)
        encoded = z + noise

        amp_condition = mic_condition = np.ones((len(padded_sequences), 1))

        conditioned = np.hstack([
            encoded,
            amp_condition,
            mic_condition,
        ])
        decoded = self._decoder.predict(conditioned, verbose=1, batch_size=80000)
        new_peptides = np.argmax(decoded, axis=2)

        new_amp = self._amp_classifier.predict(new_peptides, verbose=1, batch_size=80000)
        new_mic = self._mic_classifier.predict(new_peptides, verbose=1, batch_size=80000)

        if filtering_criteria == 'improvement':
            better = new_amp > amp_stacked.reshape(-1, 1)
            better = better & (new_mic > mic_stacked.reshape(-1, 1))
        else:
            better = new_amp >= 0.8
            better = better & (new_mic > 0.5)

        better = better.flatten()

        new_peptides = np.array([translate_peptide(x) for x in new_peptides])
        new_peptides, new_amp, new_mic, better = slice_blocks((new_peptides, new_amp, new_mic, better), block_size)
        mask = get_filtering_mask(sequences=new_peptides, filtering_options=kwargs)
        mask &= better
        filtered_peptides = _dispose_into_bucket(better, sequences, new_peptides, new_amp, new_mic, n_attempts,
                                                 block_size)
        filtered_peptides = self._encapsulate_sequential_results(filtered_peptides)
        generation_result = {
            'sequence': sequences,
            'amp': amp_org.flatten().tolist(),
            'mic': mic_org.flatten().tolist(),
            'generated_sequences': filtered_peptides

        }
        generation_result.update(calculate_physchem_prop(sequences))

        return self._transpose_sequential_results(generation_result)
