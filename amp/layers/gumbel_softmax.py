from keras import layers
from tensorflow_probability import distributions
import tensorflow as tf


class GumbelSoftmax(layers.Layer):

    def __init__(
            self,
            temperature: tf.Variable,
            **kwargs,
    ):
        self.temperature = temperature
        super(GumbelSoftmax, self).__init__(**kwargs)

    def build(self, input_shape):
        super(GumbelSoftmax, self).build(input_shape)

    def call(self, x):
        sampler = distributions.RelaxedOneHotCategorical(
            logits=x,
            temperature=self.temperature,
        )
        return sampler.sample()

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_weights(self):
        return self.temperature.eval()
