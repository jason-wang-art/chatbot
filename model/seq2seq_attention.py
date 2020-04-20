import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, Embedding, Bidirectional, Dense, Concatenate, Reshape
from model.attention import BahdanauAttention


def encoder_model(encoder_input, vocab_size, embedding_dim, units):
    # shape: (batch_size, inputs.shape[1], embedding_dim)
    output = Embedding(vocab_size, embedding_dim)(encoder_input)
    # shape: (batch_size, inputs.shape[1], encoder_units), (batch_size, encoder_units)
    output, state = GRU(units, return_sequences=True,
                        return_state=True, recurrent_initializer='glorot_uniform')(output)
    model = Model(encoder_input, [output, state])
    return model


def decoder_model(decoder_input, hidden, encoder_outputs, vocab_size, embedding_dim, units):
    """
    :param input_tensor: shape (batch_size, decoder_seq_len)
    :param hidden: (batch_size, decoder_units)
    :param encoder_outputs: (batch_size, encoder_seq_len, encoder_units)
    :param vocab_size:
    :param embedding_dim:
    :param units:
    :return:
    """
    # shape (batch_size, 1, embedding_dim) decoder 每次输入只有一个词, 因为它是从生成的词预测下一个词 一直重复
    output = Embedding(vocab_size, embedding_dim)(decoder_input)
    # shape (batch_size, encoder_units), (batch_size, encoder_seq_len, 1)
    context_vector, attention_weights = BahdanauAttention(units, trainable=True)([hidden, encoder_outputs])
    # shape (batch_size, 1, embedding_dim + encoder_units)
    # 有两种实现方式
    # 1 将context_vector和输入进行拼接 送入net
    # 2 将context_vector和gru输出进行拼接 送入net
    # 这里是实现的第1个
    output = Concatenate(axis=-1)([tf.expand_dims(context_vector, 1), output])
    # shape (batch_size, 1, decoder_units), (batch_size, decoder_units)
    output, state = GRU(units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')(output)
    # (batch_size * 1, decoder_units)
    output = Reshape(target_shape=(-1, units))(output)
    # shape (batch_size, vocab_size)
    output = Dense(vocab_size)(output)
    model = Model(inputs=[decoder_input, hidden, encoder_outputs], outputs=[output, state])
    return model


def inference(process, encoder, decoder, sentence):
    inputs = process.val_process(sentence)
    inputs = tf.convert_to_tensor(inputs)
    result = ''
    hidden = [tf.zeros((1, 256))]
    encoding_out, encoding_hidden = encoder(inputs, hidden)
    decoding_hidden = encoding_hidden
    # decoder的第一个输入是开始符
    decoding_input = tf.expand_dims([process.a_tokenizer.word_index['<eos>']], 0)
    for i in range(1000):
        predictions, decoding_hidden = decoder([decoding_input, decoding_hidden, encoding_out])
        predictions = tf.squeeze(predictions)
        predicted_id = tf.argmax(predictions).numpy()
        # print(predictions, len(predictions))
        # 碰到结束符 break
        if process.a_tokenizer.index_word[predicted_id] == '<sos>' or i > 20:
            return result, sentence
        result += process.a_tokenizer.index_word[predicted_id] + ' '
        decoding_input = tf.expand_dims([predicted_id], 0)
    return result, sentence

# from tensorflow.keras.layers import Input
# #
# inputs = Input((23,))
# encoder = encoder_model(inputs, 1000, 56, 256)
# # encoder.summary()
# print(encoder.output[0].shape, encoder.output[1].shape)
# inputs2 = Input((1,))
# input3 = Input((256, ))
# input4 = Input((23, 256))
# decoder = decoder_model(inputs2, input3, input4, 1000, 56, 256)
# decoder.summary()
