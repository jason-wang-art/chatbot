import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense


class BahdanauAttention(Layer):
    def __init__(self, units, name='attention_layer', **kwargs):
        super(BahdanauAttention, self).__init__(name=name, **kwargs)
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    # define params
    # def build(self, input_shape):
    #     super().build(input_shape)

    # tensorflow2.0 自定义层需要实现get_config 否则会报
    # NotImplementedError: Layers with arguments in `__init__` must override `get_config`.
    # 将它们转为字典键值并且返回使用
    def get_config(self):
        # config = {"units": self.units}
        base_config = super(BahdanauAttention, self).get_config()
        return dict(list(base_config.items()))

    def call(self, query, values):
        # hidden shape == (batch_size, decoder_units)
        # hidden_with_time_axis shape == (batch_size, 1, decoder_units)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)
        # score shape == (batch_size, encoder_seq_len, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, encoder_seq_len, decoder_units)
        score = self.V(tf.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, encoder_seq_len, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, encoder_units)
        # values shape (batch_size, encoder_seq_len, encoder_units)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
