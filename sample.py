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
                    bidirectional=True)

    model.load(FLAGS.checkpoint_path)

    max_len = FLAGS.num_steps
    while True:
        inp = input('input a sentence (Q to quit): ')
        if inp == 'Q':
            break
        else:
            inp = converter.sentence_to_idxs(inp)
            if (len(inp) > max_len):
                inp = inp[:max_len]
            else:
                inp = inp + [0 for i in range(max_len - len(inp))]
            sample_id = model.sample(inp)
            output = converter.idxs_to_words(sample_id[0])
            print('output: %s' % output)
            print('--------------------')


if __name__ == '__main__':
    tf.app.run()
