from keras import backend
from keras import layers


class AutoregressiveGRU(layers.Layer):

    def __init__(
            self,
            output_dim: int,
            output_len: int,
            recurrent: layers.Recurrent,
            **kwargs,
    ):
        self.output_dim = output_dim
        self.output_len = output_len
        self.initial_state = None
        self.recurrent = recurrent
        super(AutoregressiveGRU, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AutoregressiveGRU, self).build(input_shape)

    def call(self, x):
        outputs = []
        current_output = backend.zeros_like(backend.repeat(x, 1))
        current_state = x
        for _ in range(self.output_len):
            current_output, current_state = self.recurrent(
                current_output,
                initial_state=current_state,
            )
            if hasattr(self.recurrent, 'loaded_weights') and self.recurrent.loaded_weights is not None:
                self.recurrent.set_weights(self.recurrent.loaded_weights)
            outputs.append(current_output)
        result = layers.concatenate(outputs, axis=1)
        result = backend.reshape(result, (-1, self.output_len, self.output_dim))
        return result

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_len, self.output_dim

    @classmethod
    def build_for_gru(
            cls,
            output_dim: int,
            output_len: int,
            **kwargs,
    ):
        return cls(
            output_dim=output_dim,
            output_len=output_len,
            recurrent=layers.GRU(
                output_dim,
                return_sequences=True,
                return_state=True,
            ),
            **kwargs,
        )

    @classmethod
    def build_for_lstm(
            cls,
            output_dim: int,
            output_len: int,
            **kwargs,
    ):
        return cls(
            output_dim=output_dim,
            output_len=output_len,
            recurrent=layers.LSTM(
                output_dim,
                return_sequences=True,
                return_state=True,
            ),
            **kwargs,
        )
