import tensorflow as tf
from tensorflow.keras import layers, Model

class AttentionLayer(layers.Layer):
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1],1))
        self.b = self.add_weight(shape=(input_shape[1],1))


    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x,self.W)+self.b)
        a = tf.nn.softmax(e, axis=1)
        context = tf.reduce_sum(x*a, axis=1)
        return context, a


def build_attention(window=20, features=3, units=64, lr=0.001):


    inp = layers.Input(shape=(window,features))

    x = layers.LSTM(units, return_sequences=True)(inp)

    context, weights = AttentionLayer()(x)

    out = layers.Dense(1)(context)

    model = Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")

    weight_model = Model(inp, weights)

    return model, weight_model
