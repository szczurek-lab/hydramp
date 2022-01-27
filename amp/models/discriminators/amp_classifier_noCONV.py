from typing import Any
from typing import Dict
from typing import Optional

from keras import layers
from keras import models
from keras import optimizers

from amp.models.discriminators import discriminator
from amp.models import model


class NoConvAMPClassifier(discriminator.Discriminator):
    """
    The discriminator part of cVAE.
    """

    def __init__(
            self,
            embedding: layers.Embedding,
            lstm1: layers.Layer,
            lstm2: layers.Layer,
            dense_output: layers.Layer,
            input_shape: tuple,
            name: str = 'NoConvAMPClassifier',
    ):
        self.embedding = embedding
        self.lstm1 = lstm1
        self.lstm2 = lstm2
        self.dense_output = dense_output
        self.name = name
        self.input_shape = input_shape
        self.dense_emb = layers.Dense(self.embedding.output_dim, use_bias=False)

    def output_tensor(self, input_=None):

        if input_ is None:
            x = layers.Input(shape=self.input_shape)
        else:
            x = input_

        emb = self.call_layer_on_input(self.embedding, x)
        lstm1 = self.call_layer_on_input(self.lstm1, emb)
        pool = self.call_layer_on_input(layers.MaxPooling1D(pool_size=5), lstm1)
        lstm2 = self.call_layer_on_input(self.lstm2, pool)
        return self.call_layer_on_input(self.dense_output, lstm2)

    def output_tensor_with_dense_input(self, input_: Optional[Any]):
        if input_ is None:
            x = layers.Input(shape=(self.input_shape[0], 21))
        else:
            x = input_

        emb = self.call_layer_on_input(self.dense_emb, x)
        try:
            self.dense_emb.set_weights(self.embedding.get_weights())
        except ValueError:
            if hasattr(self.embedding, 'loaded_weights') and self.embedding.loaded_weights:
                self.dense_emb.set_weights(self.embedding.loaded_weights)

        self.dense_emb.trainable = self.embedding.trainable

        lstm1 = self.call_layer_on_input(self.lstm1, emb)
        pool = self.call_layer_on_input(layers.MaxPooling1D(pool_size=5), lstm1)
        lstm2 = self.call_layer_on_input(self.lstm2, pool)
        return self.call_layer_on_input(self.dense_output, lstm2)

    def __call__(self, input_=None):
        x = input_ if input_ is not None else layers.Input(shape=(self.input_shape[0],))
        model = models.Model(x, self.output_tensor(x))
        adam = optimizers.Adam(lr=1e-3)

        model.compile(
            loss='binary_crossentropy',
            metrics=['accuracy'],
            optimizer=adam
        )

        return model

    def freeze_layers(self):
        self.embedding.trainable = False
        self.lstm1.trainable = False
        self.lstm2.trainable = False
        self.dense_emb.trainable = False
        self.dense_output.trainable = False

    def unfreeze_layers(self):
        self.embedding.trainable = True
        self.lstm1.trainable = True
        self.lstm2.trainable = True
        self.dense_emb.trainable = True
        self.dense_output.trainable = True

    def get_config_dict(self) -> Dict:
        return {
            'type': type(self).__name__,
            'name': self.name,
            'input_shape': self.input_shape,
        }

    def get_layers_with_names(self) -> Dict[str, layers.Layer]:
        return {
            f'{self.name}_embedding': self.embedding,
            f'{self.name}_lstm1': self.lstm1,
            f'{self.name}_lstm2': self.lstm2,
            f'{self.name}_dense_output': self.dense_output,
        }

    @classmethod
    def from_config_dict_and_layer_collection(
            cls,
            config_dict: Dict,
            layer_collection: model.ModelLayerCollection,
    ) -> "NoConvAMPClassifier":
        return cls(
            name=config_dict['name'],
            embedding=layer_collection[config_dict['name'] + '_embedding'],
            lstm1=layer_collection[config_dict['name'] + '_lstm1'],
            lstm2=layer_collection[config_dict['name'] + '_lstm2'],
            dense_output=layer_collection[config_dict['name'] + '_dense_output'],
            input_shape=config_dict['input_shape'],
        )


class NoConvAMPClassifierFactory:

    @staticmethod
    def get_default(max_length: int) -> NoConvAMPClassifier:

        emb = layers.Embedding(
            input_dim=21,
            output_dim=128,
            input_length=max_length,
        )
        lstm1 = layers.LSTM(
            64,
            unroll=True,
            stateful=False,
            dropout=0.1,
            return_sequences=True
        )
        lstm2 = layers.LSTM(
            100,
            unroll=True,
            stateful=False,
            dropout=0.1)

        dense_output = layers.Dense(1, activation='sigmoid', name="classifier_dense")
        return NoConvAMPClassifier(
            embedding=emb,
            lstm1=lstm1,
            lstm2=lstm2,
            dense_output=dense_output,
            input_shape=(max_length, 21)
        )
