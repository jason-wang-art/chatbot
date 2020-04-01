import tensorflow as tf
from pre_process import PreProcess
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import time
import os
from model.seq2seq_attention import encoder_model, decoder_model
from tensorflow.keras.layers import Input

encoder_weights_path = 'models/encoder.h5'
decoder_weights_path = 'models/decoder.h5'

process = PreProcess('./data/qingyun.tsv', samples_num=3000)
samples_num = process.length

# define params
batch_size = 64
embedding_dim = 50
units = 256
steps_per_epoch = samples_num // batch_size

encoder_input = Input((process.q_lenght,))
encoder = encoder_model(encoder_input, process.q_vocab_size, embedding_dim, units)

decoder_input, hidden_input, encoder_output_input = Input((1,)), Input((units,)), Input((process.q_lenght, units))
decoder = decoder_model(decoder_input, hidden_input, encoder_output_input, process.a_vocab_size, embedding_dim, units)

if os.path.exists(encoder_weights_path):
    encoder.load_weights(encoder_weights_path)

if os.path.exists(decoder_weights_path):
    decoder.load_weights(decoder_weights_path)

data_generate = process.generate_data(batch_size)
# 学习率衰减来做
optimizer = Adam(lr=1e-3)

loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


# 转成图方法 加快运算
# @tf.function
def train_step(inp, targ):
    loss = 0
    with tf.GradientTape() as tape:
        encoding_output, encoding_hidden = encoder(inp)
        decoding_hidden = encoding_hidden
        # shape (batch_size, 1)
        decoding_input = tf.expand_dims([process.a_tokenizer.word_index['<eos>']] * batch_size, 1)

        # feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, decoding_hidden = decoder([decoding_input, decoding_hidden, encoding_output])
            loss += loss_function(targ[:, t], predictions)
            decoding_input = tf.expand_dims(targ[:, t], 1)
    batch_loss = (loss / int(targ.shape[0]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss


epochs = 100
epoch_batch_num = samples_num // batch_size

# model weights path
model_weights_path = './models'
if not os.path.exists(model_weights_path):
    os.mkdir(model_weights_path)

for epoch in range(epochs):
    total_loss = 0
    start = time.time()

    for step in range(epoch_batch_num):
        inp, targ = next(data_generate)
        batch_loss = train_step(inp, targ)
        total_loss += batch_loss
        # if step % 100 == 0:
        print('Epoch {} step {} Loss {:.4f}'.format(epoch + 1, step + 1, batch_loss.numpy()))
    if (epoch + 1) % 5 == 0:
        encoder.save('./models/encoder_%02d.h5' % ((epoch + 1) / 5))
        decoder.save('./models/decoder_%02d.h5' % ((epoch + 1) / 5))
    print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
