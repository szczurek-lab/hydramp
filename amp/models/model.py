from typing import Dict

from keras import layers


class ModelLayerCollection:

    def __getitem__(self, item: str) -> layers.Layer:
        raise NotImplementedError

    def __add__(self, layers_with_names: Dict[str, layers.Layer]):
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "ModelLayerCollection":
        raise NotImplementedError


class Model:

    def get_config_dict(self) -> Dict:
        raise NotImplementedError

    def get_layers_with_names(self) -> Dict[str, layers.Layer]:
        raise NotImplementedError

    @classmethod
    def from_config_dict_and_layer_collection(
            cls,
            config_dict: Dict,
            layer_collection: ModelLayerCollection,
    ) -> "Model":
        raise NotImplementedError

    @staticmethod
    def call_layer_on_input(layer: layers.Layer, input_: layers.Layer):
        if hasattr(layer, 'loaded_weights') and layer.loaded_weights:
            result = layer(input_)
            layer.set_weights(layer.loaded_weights)
            return result
        return layer(input_)


class ModelSerializer:

    def save_model(self, model: Model, path: str):
        raise NotImplementedError

    def load_model(self, path: str) -> Model:
        raise NotImplementedError
