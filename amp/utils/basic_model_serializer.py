import json
import os
from typing import Tuple

from keras import layers, models, activations, Model

from amp.config import LATENT_DIM, MAX_LENGTH
from amp.models import model as amp_model
from amp.models import model_garden
from amp.models.master.master import MasterAMPTrainer
from amp.utils import dict_model_layer_collection


class BasicModelSerializer(amp_model.ModelSerializer):
    MODEL_CONFIG_NAME = 'model_config.json'
    LAYER_DIR_NAME = 'layers'

    def save_model(self, model: amp_model.Model, path: str):
        os.makedirs(path, exist_ok=True)
        model_config_path = os.path.join(path, self.MODEL_CONFIG_NAME)
        with open(model_config_path, 'w') as json_handle:
            json.dump(model.get_config_dict(), json_handle)
        model_layer_collection = dict_model_layer_collection.DictModelLayerCollection(
            dict_to_layer=model.get_layers_with_names(),
        )
        layer_saving_path = os.path.join(path, self.LAYER_DIR_NAME)
        model_layer_collection.save(layer_saving_path)

    def load_model(self, path: str) -> amp_model.Model:
        model_config_path = os.path.join(path, self.MODEL_CONFIG_NAME)
        with open(model_config_path, 'r') as json_handle:
            model_config = json.load(json_handle)
        model_class = model_garden.MODEL_GAREDN[model_config['type']]
        model_layer_collection_path = os.path.join(path, self.LAYER_DIR_NAME)
        return model_class.from_config_dict_and_layer_collection(
            config_dict=model_config,
            layer_collection=dict_model_layer_collection.DictModelLayerCollection.load(
                path=model_layer_collection_path,
            )
        )


def load_master_model_components(model_path: str, return_master=False, softmax=False) -> Tuple[models.Model, ...]:
    serializer = BasicModelSerializer()
    amp_master: MasterAMPTrainer = serializer.load_model(path=model_path)

    input_to_encoder = layers.Input(shape=(MAX_LENGTH,))
    encoder_model = amp_master.encoder(input_to_encoder)
    input_to_decoder = layers.Input(shape=(LATENT_DIM + 2,))
    decoder_model = amp_master.decoder(input_to_decoder)
    amp_classifier = amp_master.amp_classifier(input_to_encoder)
    mic_classifier = amp_master.mic_classifier(input_to_encoder)

    if softmax:
        new_act = layers.TimeDistributed(layers.Activation(activations.softmax),
                                         name='decoder_time_distribute_activation')
        decoder_model.layers.pop()
        x = new_act(decoder_model.layers[-1].output)
        decoder_model = Model(input=decoder_model.input, output=[x])

    if return_master:
        return encoder_model, decoder_model, amp_classifier, mic_classifier, amp_master
    return encoder_model, decoder_model, amp_classifier, mic_classifier
