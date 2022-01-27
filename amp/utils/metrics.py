from keras import backend
from keras import metrics
import tensorflow as tf


def kl_loss(z_mean, z_sigma):
    return - 0.5 * backend.sum(
        1 + z_sigma - backend.square(z_mean) - backend.exp(z_sigma),
        axis=-1
    )


def sparse_categorical_accuracy(y_true, y_pred):
    return backend.mean(backend.equal(y_true, backend.cast(backend.argmax(y_pred, axis=-1), backend.floatx())))


def reconstruction_loss(y_true, y_pred):
    return backend.mean(metrics.sparse_categorical_crossentropy(y_true, y_pred), axis=-1)


def get_generation_acc(threshold: float = 0.5):
    def metric_(y_true, y_pred):
        exceeds_zero_threshold = tf.math.greater(y_pred[:, :, 0], threshold)
        exceeds_zero_threshold = tf.cast(exceeds_zero_threshold, 'float32')
        exceeds_threshold_flag = tf.math.greater(
            tf.cumsum(exceeds_zero_threshold, axis=1), 0
        )
        amino_tp = tf.cast(tf.equal(tf.cast(tf.argmax(y_pred[:, :, 1:], axis=-1) + 1, 'float32'), y_true), 'float32')
        empty_tp = tf.cast(tf.equal(y_true, 0), 'float32')
        amino_tp = tf.math.reduce_sum(tf.where(
            exceeds_threshold_flag,
            tf.zeros_like(amino_tp, dtype='float32'),
            amino_tp,
        ), axis=-1)
        empty_tp = tf.math.reduce_sum(tf.where(
            exceeds_threshold_flag,
            empty_tp,
            tf.zeros_like(empty_tp, dtype='float32'),
        ), axis=-1)
        empty_entries_sum = tf.reduce_sum(
            tf.cast(exceeds_threshold_flag, dtype='float32'), axis=-1)
        non_empty_entries_sum = tf.reduce_sum(1 - tf.cast(
            exceeds_threshold_flag, dtype='float32'), axis=-1)
        amino_acc = tf.where(
            non_empty_entries_sum > 0,
            amino_tp / non_empty_entries_sum,
            tf.zeros_like(amino_tp, dtype='float32'),
        )
        empty_acc = tf.where(
            empty_entries_sum > 0,
            empty_tp / empty_entries_sum,
            tf.ones_like(empty_tp, dtype='float32')
        )
        return amino_acc, empty_acc
    return metric_
