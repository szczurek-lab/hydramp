from typing import Any
from typing import Dict
from typing import Optional

from keras import layers
from keras import models
from keras import backend

from amp.models.encoders import encoder
from amp.models import model


class AMPEncoder(encoder.Encoder):

    def __init__(
            self,
            embedding: layers.Embedding,
            hidden: layers.Layer,
            hidden2: layers.Layer,
            dense_z_mean: layers.Layer,
            dense_z_sigma: layers.Layer,
            input_shape: tuple,
            latent_dim: int,
            hidden_dim: int,
            name: str = 'AMPExpandedEncoder',
    ):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.input_shape = input_shape

        self.embedding = embedding
        self.hidden = hidden
        self.hidden2 = hidden2
        self.z_mean = dense_z_mean
        self.z_sigma = dense_z_sigma
        self.name = name
        self.dense_emb = layers.Dense(self.embedding.output_dim, use_bias=False, name="encoder_dense_emb")

    def output_tensor(self, input_=None):
        emb = self.call_layer_on_input(self.embedding, input_)
        hidden = self.call_layer_on_input(self.hidden, emb)
        hidden2 = self.call_layer_on_input(self.hidden2, hidden)
        z_mean = self.call_layer_on_input(self.z_mean, hidden2)
        z_sigma = self.call_layer_on_input(self.z_sigma, hidden2)
        z = self.call_layer_on_input(layers.Lambda(self.sampling, output_shape=(self.latent_dim,)), ([z_mean, z_sigma]))

        return z_mean, z_sigma, z

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
        hidden = self.call_layer_on_input(self.hidden, emb)
        hidden2 = self.call_layer_on_input(self.hidden2, hidden)
        z_mean = self.call_layer_on_input(self.z_mean, hidden2)
        z_sigma = self.call_layer_on_input(self.z_sigma, hidden2)
        z = self.call_layer_on_input(layers.Lambda(self.sampling, output_shape=(self.latent_dim,)), ([z_mean, z_sigma]))

        return z_mean, z_sigma, z

    def __call__(self, input_=None):
        x = input_ if input_ is not None else layers.Input(shape=(self.input_shape[0],))
        z_mean, z_sigma, z = self.output_tensor(x)
        model = models.Model(x, z_mean)
        return model

    def sampling(self, input_: Optional[Any] = None):
        z_mean, z_sigma = input_
        epsilon = backend.random_normal(shape=(self.latent_dim,),
                                        mean=0., stddev=1.)
        return z_mean + backend.exp(z_sigma / 2) * epsilon

    def get_config_dict(self) -> Dict:
        return {
            'type': type(self).__name__,
            'name': self.name,
            'latent_dim': self.latent_dim,
            'hidden_dim': self.hidden_dim,
            'input_shape': self.input_shape
        }

    def get_layers_with_names(self) -> Dict[str, layers.Layer]:
        return {

            f'{self.name}_embedding': self.embedding,
            f'{self.name}_hidden': self.hidden,
            f'{self.name}_hidden2': self.hidden2,
            f'{self.name}_dense_z_mean': self.z_mean,
            f'{self.name}_dense_z_sigma': self.z_sigma,
        }

    @classmethod
    def from_config_dict_and_layer_collection(
            cls,
            config_dict: Dict,
            layer_collection: model.ModelLayerCollection,
    ) -> "AMPEncoder":
        return cls(
            name=config_dict['name'],
            embedding=layer_collection[config_dict['name'] + '_embedding'],
            hidden=layer_collection[config_dict['name'] + '_hidden'],
            hidden2=layer_collection[config_dict['name'] + '_hidden2'],
            hidden_dim=config_dict['hidden_dim'],
            latent_dim=config_dict['latent_dim'],
            input_shape=config_dict['input_shape'],
            dense_z_mean=layer_collection[config_dict['name'] + '_dense_z_mean'],
            dense_z_sigma=layer_collection[config_dict['name'] + '_dense_z_sigma'],
        )


class AMPEncoderFactory:

    @staticmethod
    def get_default(
            hidden_dim: int,
            latent_dim: int,
            max_length: int,
    ) -> AMPEncoder:
        emb = layers.Embedding(
            input_dim=21,
            output_dim=100,
            input_length=max_length,
            mask_zero=False,
            name="encoder_embedding"
        )
        hidden = layers.Bidirectional(
            layers.GRU(
                hidden_dim,
                return_sequences=True,
            ),
            name="encoder_hidden_bidirectional_1"
        )
        hidden2 = layers.Bidirectional(
            layers.GRU(
                hidden_dim,
                return_sequences=False,
            ),
            name="encoder_hidden_bidirectional_2"
        )

        dense_z_mean = layers.Dense(latent_dim, name="encoder_dense_z_mean")
        dense_z_sigma = layers.Dense(latent_dim, name="encoder_dense_z_sigma")
        return AMPEncoder(
            embedding=emb,
            hidden=hidden,
            hidden2=hidden2,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            input_shape=(max_length, 21),
            dense_z_mean=dense_z_mean,
            dense_z_sigma=dense_z_sigma,
        )
