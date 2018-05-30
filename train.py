import tensorflow as tf
from utils import batch_generator, TextConverter
from seq2seq import Seq2Seq
from dataprepare import loaddata
import os
import codecs


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('name', 'default', 'name of the model')
tf.flags.DEFINE_integer('batch_size', 64, 'number of seqs in one batch')
tf.flags.DEFINE_integer('num_steps', 10, 'length of one input seq')
tf.flags.DEFINE_integer('max_steps', 10, 'max length of one output seq')
tf.flags.DEFINE_integer('lstm_size', 512, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', True, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
tf.flags.DEFINE_float('train_keep_prob', 0.8, 'dropout rate during training')
tf.flags.DEFINE_integer('max_iters', 10000, 'max iters to train')
tf.flags.DEFINE_integer('save_every_n', 500, 'save the model every n steps')
tf.flags.DEFINE_integer('log_every_n', 50, 'log to the screen every n steps')
tf.flags.DEFINE_integer('max_vocab', 3500, 'max char number')
tf.flags.DEFINE_boolean('bidirectional', True, 'whether to use bidirectional')

path = './cornell movie-dialogs corpus/'

def main(_):
    model_path = os.path.join('model', FLAGS.name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)

    train_data, train_lang = loaddata(path, FLAGS.num_steps)
    vocab_size = train_lang.vocab_size

    converter = TextConverter(lang=train_lang, max_vocab=FLAGS.max_vocab)
    converter.save_lang(filename= FLAGS.name +'_converter.pkl')

    g = batch_generator(train_data, FLAGS.batch_size, FLAGS.max_steps)
    
    model = Seq2Seq('train', vocab_size,
                    batch_size=FLAGS.batch_size,
                    num_steps=FLAGS.num_steps,
                    max_steps=FLAGS.max_steps,
                    lstm_size=FLAGS.lstm_size,
                    num_layers=FLAGS.num_layers,
                    learning_rate=FLAGS.learning_rate,
                    train_keep_prob=FLAGS.train_keep_prob,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size,
                    max_iters=FLAGS.max_iters,
                    bidirectional=FLAGS.bidirectional,
                    beam_search=False
                    )
    model.train(g,
                converter,
                FLAGS.max_steps,
                model_path,
                FLAGS.save_every_n,
                FLAGS.log_every_n,
                )


if __name__ == '__main__':
    tf.app.run()
