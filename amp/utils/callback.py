import csv
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from amp.config import KL_ANNEALRATE, MAX_KL, MAX_LENGTH, MAX_TEMPERATURE, MIN_KL, MIN_TEMPERATURE, TAU_ANNEALRATE
from amp.data_utils import sequence
from amp.models.model import Model
from amp.utils.basic_model_serializer import BasicModelSerializer
from keras import backend
from keras.callbacks import Callback


class VAECallback(Callback):
    def __init__(
            self,
            encoder,
            decoder,
            amp_classifier,
            mic_classifier,
            kl_annealrate: float = KL_ANNEALRATE,
            max_kl: float = MAX_KL,
            kl_weight=backend.variable(MIN_KL, name="kl_weight"),
            tau=backend.variable(MAX_TEMPERATURE, name="temperature "),
            tau_annealrate=TAU_ANNEALRATE,
            min_tau=MIN_TEMPERATURE,
            max_length: int = MAX_LENGTH,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.amp_classifier = amp_classifier
        self.mic_classifier = mic_classifier
        self.kl_annealrate = kl_annealrate
        self.max_kl = max_kl
        self.kl_weight = kl_weight
        self.tau = tau
        self.tau_annealrate = tau_annealrate
        self.min_tau = min_tau
        self.positive_callback_sample = sequence.pad(sequence.to_one_hot(['GFKDLLKGAAKALVKTVLF'])).reshape(1,
                                                                                                           max_length)
        self.negative_callback_sample = sequence.pad(sequence.to_one_hot(['FPSELANMKNALGFFHIGEIF'])).reshape(1,
                                                                                                             max_length)
        self.positive_mic = self.mic_classifier.predict(self.positive_callback_sample.reshape(1, 25))
        self.negative_mic = self.mic_classifier.predict(self.negative_callback_sample.reshape(1, 25))

    def on_epoch_end(self, epoch, logs={}):
        alphabet = list('ACDEFGHIKLMNPQRSTVWY')
        new_kl = np.min([backend.get_value(self.kl_weight) * np.exp(self.kl_annealrate * epoch), self.max_kl])
        backend.set_value(self.kl_weight, new_kl)

        pos_encoded_sample = self.encoder.predict(self.positive_callback_sample)
        neg_encoded_sample = self.encoder.predict(self.negative_callback_sample)
        pos_sample = np.concatenate([pos_encoded_sample, np.array([[1]]), np.array(self.positive_mic)], axis=1)
        neg_sample = np.concatenate([neg_encoded_sample, np.array([[0]]), np.array(self.negative_mic)], axis=1)
        pos_prediction = self.decoder.predict(pos_sample)
        neg_prediction = self.decoder.predict(neg_sample)
        pos_peptide = ''.join([alphabet[el - 1] if el != 0 else "'" for el in pos_prediction[0].argmax(axis=1)])
        neg_peptide = ''.join([alphabet[el - 1] if el != 0 else "'" for el in neg_prediction[0].argmax(axis=1)])
        pos_class_prob = self.amp_classifier.predict(np.array([pos_prediction[0].argmax(axis=1)]))
        neg_class_prob = self.amp_classifier.predict(np.array([neg_prediction[0].argmax(axis=1)]))
        pos_mic_pred = self.mic_classifier.predict(np.array([pos_prediction[0].argmax(axis=1)]))
        neg_mic_pred = self.mic_classifier.predict(np.array([neg_prediction[0].argmax(axis=1)]))

        new_tau = np.max([backend.get_value(self.tau) * np.exp(- self.tau_annealrate * epoch), self.min_tau])
        backend.set_value(self.tau, new_tau)

        print(
            f'Original positive: GGAGHVPEYFVGIGTPISFYG, \n'
            f'generated: {pos_peptide}, \n'
            f'AMP probability: {pos_class_prob[0][0]}, \n',
            f'MIC prediction: {pos_mic_pred[0][0]}. \n'
        )
        print(
            f'Original negative: FPSELANMKNALGFFHIGEIF, \n'
            f'generated: {neg_peptide}, \n'
            f'AMP probability: {neg_class_prob[0][0]}, \n'
            f'MIC prediction: {neg_mic_pred[0][0]}. \n'
        )

        print("Current KL weight is " + str(backend.get_value(self.kl_weight)))
        print("Current temperature is " + str(backend.get_value(self.tau)))


class SaveModelCallback(Callback):

    def __init__(self, model: Model, model_save_path, name):
        """
        model : amp.models.Model instance (that corresponds to trained keras.Model instance)
        model_save_path: location for model root directory
        name: name of the model (experiment), this is also the name of model root directory
        """
        super().__init__()
        self.model_st = model
        self.model_save_path = model_save_path
        self.name = name
        self.root_dir = os.path.join(model_save_path, name)
        self.serializer = BasicModelSerializer()
        os.path.join(model_save_path, self.name)
        Path(self.root_dir).mkdir(parents=True, exist_ok=True)

        self.metrics_file_path = os.path.join(self.root_dir, "metrics.csv")
        self.metrics_initialized = False
        self.metric_order = None

    def _initialize_metrics_doc(self, metrics_names: List[str]):
        col_names = ["epoch_no"] + metrics_names
        with open(self.metrics_file_path, 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(col_names)
        self.metric_order = metrics_names
        self.metrics_initialized = True

    def _save_metrics(self, logs: Dict[str, Any], epoch_no: int):
        unboxed_metrics = [logs[m_name] for m_name in self.metric_order]
        row = [epoch_no] + unboxed_metrics
        with open(self.metrics_file_path, 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def on_epoch_end(self, epoch, logs=None):
        if not self.metrics_initialized:
            self._initialize_metrics_doc(list(logs.keys()))
        save_path = os.path.join(self.model_save_path, self.name, str(epoch))
        self.serializer.save_model(self.model_st, save_path)
        self._save_metrics(logs, epoch)
