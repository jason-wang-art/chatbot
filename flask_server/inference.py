from model.seq2seq_attention import encoder_model, decoder_model, inference
from pre_process import PreProcess
from tensorflow.keras.layers import Input

process = PreProcess('../data/qingyun.tsv')

encoder_weights_path = '../models/encoder.h5'
decoder_weights_path = '../models/decoder.h5'

# define params
embedding_dim = 50
units = 256


def get_response(sentence):
    # get model
    encoder_input = Input((process.q_lenght,))
    encoder = encoder_model(encoder_input, process.q_vocab_size, embedding_dim, units)

    decoder_input, hidden_input, encoder_output_input = Input((1,)), Input((units,)), Input((process.q_lenght, units))
    decoder = decoder_model(decoder_input, hidden_input, encoder_output_input, process.a_vocab_size, embedding_dim,
                            units)
    # load model weights
    encoder.load_weights(encoder_weights_path)
    decoder.load_weights(decoder_weights_path)
    result, sentence = inference(process, encoder, decoder, sentence)
    return result.replace(' ', ''), sentence
