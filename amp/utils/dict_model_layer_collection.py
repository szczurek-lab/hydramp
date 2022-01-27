from typing import Dict
import json
import os

from keras import layers
import numpy as np

from amp.models import model


class DictModelLayerCollection(model.ModelLayerCollection):

    LAYER_CONFIG_NAME = 'layer.json'
    WEIGHT_NAME = '_weight.npy'

    def __init__(self, dict_to_layer: Dict[str, layers.Layer]):
        self._dict_to_layer = dict_to_layer

    def __add__(self, layers_with_names: Dict[str, layers.Layer]):
        for layer_name, layer in layers_with_names.items():
            self._dict_to_layer[layer_name] = layer
        return self

    def __getitem__(self, item: str) -> layers.Layer:
        return self._dict_to_layer[item]

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        for current_layer_name, current_layer in self._dict_to_layer.items():
            current_layer_path = os.path.join(path, current_layer_name)
            self._save_single_layer(path=current_layer_path, layer=current_layer)

    def _save_single_layer(self, path: str, layer: layers.Layer):
        os.makedirs(path, exist_ok=True)
        config_path = os.path.join(path, self.LAYER_CONFIG_NAME)
        with open(config_path, 'w') as json_handle:
            config = {
                'class_name': layer.__class__.__name__,
                'config': layer.get_config(),
            }
            json.dump(config, json_handle)
            for layer_nb, weight in enumerate(layer.get_weights()):
                current_weight_path = os.path.join(path, f'{layer_nb}{self.WEIGHT_NAME}')
                np.save(current_weight_path, weight)

    @classmethod
    def load(cls, path: str) -> "DictModelLayerCollection":
        files_in_dir = os.listdir(path)
        full_paths = [os.path.join(path, file_path) for file_path in files_in_dir]
        full_subdirs = [(path, name) for path, name in zip(full_paths, files_in_dir) if os.path.isdir(path)]
        return cls(
            dict_to_layer={
                name: cls._load_layer_from_path(path) for path, name in full_subdirs
            }
        )

    @classmethod
    def _load_layer_from_path(cls, path):
        config_dict_path = os.path.join(path, cls.LAYER_CONFIG_NAME)
        with open(config_dict_path, 'r') as json_handle:
            config = json.load(json_handle)
        layer = layers.deserialize(
            config=config,
        )
        weight_paths = sorted([
            os.path.join(path, weight_name)
            for weight_name in os.listdir(path) if weight_name.endswith(cls.WEIGHT_NAME)
        ])
        weights = [np.load(weight_path) for weight_path in weight_paths]
        layers.Layer.loaded_weights = None
        layer.loaded_weights = weights
        return layer
