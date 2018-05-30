import tensorflow as tf
from utils import TextConverter
from seq2seq import Seq2Seq
import os
from IPython import embed

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('num_steps', 10, 'length of one input seq')
tf.flags.DEFINE_integer('lstm_size', 512, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', True, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_string('converter_name', 'default', 'model/name/converter.pkl')
tf.flags.DEFINE_string('checkpoint_path', './model/default/', 'checkpoint path')
tf.flags.DEFINE_integer('max_length', 10, 'max length to generate')
tf.flags.DEFINE_boolean('bidirectional', True, 'whether to use bidirectional')
tf.flags.DEFINE_boolean('beam_search', False, 'whether to use beam search')
tf.flags.DEFINE_integer('beam_width', 3, 'size for beam search')


def main(_):
    converter = TextConverter(filename=FLAGS.converter_name +'_converter.pkl')
    if os.path.isdir(FLAGS.checkpoint_path):
        FLAGS.checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)

    model = Seq2Seq('sample', converter.vocab_size,
                    lstm_size=FLAGS.lstm_size,
                    num_steps=FLAGS.num_steps,
                    num_layers=FLAGS.num_layers,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size,
                    bidirectional=FLAGS.bidirectional,
                    beam_search=FLAGS.beam_search,
                    beam_width=FLAGS.beam_width)

    model.load(FLAGS.checkpoint_path)

    max_len = FLAGS.num_steps
    while True:
        inp = input('Input (Q to quit): ')
        if inp == 'Q':
            break
        else:
            inp = converter.sentence_to_idxs(inp)
            if (len(inp) > max_len):
                inp = inp[:max_len]
            else:
                inp = inp + [0 for i in range(max_len - len(inp))]
            if FLAGS.beam_search == True:
                decoder_outputs = model.sample(inp)
                predicted_ids = decoder_outputs.predicted_ids[0]
                parent_ids = decoder_outputs.parent_ids[0]
                sentences = converter.beam_to_sentences(predicted_ids, parent_ids)
                for i, s in enumerate(sentences):
                    print('Output %d: %s' % (i, s))
            else:
                sample_id = model.sample(inp)
                output = converter.idxs_to_words(sample_id[0])
                print('Output: %s' % output)
            print('--------------------')


if __name__ == '__main__':
    tf.app.run()
