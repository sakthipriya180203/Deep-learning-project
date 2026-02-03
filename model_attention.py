import tensorflow as tf
from tensorflow.keras import layers

class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1), initializer="random_normal")
        self.b = self.add_weight(shape=(1,), initializer="zeros")

    def call(self, inputs):
        score = tf.matmul(inputs, self.W) + self.b
        weights = tf.nn.softmax(score, axis=1)
        context = weights * inputs
        return tf.reduce_sum(context, axis=1)
