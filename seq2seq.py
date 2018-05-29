# coding: utf-8
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os


class Seq2Seq:
	def __init__(self, mode, vocab_size, batch_size=64, num_steps=20,
				 max_steps=20, lstm_size=128, num_layers=2, learning_rate=0.001,
				 grad_clip=5, train_keep_prob=0.5, use_embedding=False,
				 embedding_size=128, max_iters=10000, bidirectional=False):
		if mode == 'sample':
			batch_size = 1
		# else:
		# 	batch_size, num_steps = batch_size, num_steps

		self.mode = mode
		self.vocab_size = vocab_size
		self.batch_size = batch_size
		self.num_steps = num_steps
		self.max_steps = max_steps
		self.max_iters = max_iters
		self.lstm_size = lstm_size
		self.num_layers = num_layers
		self.learning_rate = learning_rate
		self.grad_clip = grad_clip
		self.train_keep_prob = train_keep_prob
		self.use_embedding = use_embedding
		self.embedding_size = embedding_size
		self.bidirectional = bidirectional

		tf.reset_default_graph()
		self.build_inputs()
		self.build_encoder()
		self.build_decoder()
		self.build_loss()
		self.build_optimizer()
		self.saver = tf.train.Saver()

	def build_inputs(self):
		with tf.name_scope('inputs'):
			self.inputs = tf.placeholder(tf.int32, shape=(
				self.batch_size, self.num_steps), name='inputs')
			self.targets1 = tf.placeholder(tf.int32, shape=(
				self.batch_size, None), name='targets1')
			self.targets2 = tf.placeholder(tf.int32, shape=(
				self.batch_size, None), name='targets2')
			self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
			self.x_seq_lengths = tf.placeholder(tf.int32, shape=(self.batch_size), name='x_seq_lengths')
			self.y_seq_lengths = tf.placeholder(tf.int32, shape=(self.batch_size), name='y_seq_lengths')

			self.embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size])
			if self.use_embedding:
				# embedding
				self.encoder_inputs = tf.nn.embedding_lookup(self.embedding, self.inputs)
				self.decoder_inputs = tf.nn.embedding_lookup(self.embedding, self.targets1)
			else:
				# one hot
				self.encoder_inputs = tf.one_hot(self.inputs, self.vocab_size)
				self.decoder_inputs = tf.one_hot(self.targets1, self.vocab_size)

	def get_a_cell(self, lstm_size, keep_prob):
		lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
		drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
		return drop

	def build_encoder(self):
		with tf.name_scope('encoder'):
			if not self.bidirectional:
				# basic lstm RNN
				encoder_cell = tf.nn.rnn_cell.MultiRNNCell(
					[self.get_a_cell(self.lstm_size, self.keep_prob) for _ in range(self.num_layers)]
				)

				self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(encoder_cell, self.encoder_inputs, dtype=tf.float32)
			else:
				# bidirectional RNN
				encoder_cell_fw = tf.nn.rnn_cell.MultiRNNCell(
					[self.get_a_cell(self.lstm_size, self.keep_prob) for _ in range(self.num_layers)]
				)
				encoder_cell_bw = tf.nn.rnn_cell.MultiRNNCell(
					[self.get_a_cell(self.lstm_size, self.keep_prob) for _ in range(self.num_layers)]
				)

				encoder_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(encoder_cell_fw,
																		encoder_cell_bw,
																		self.encoder_inputs,
																		sequence_length=self.x_seq_lengths,
																		dtype=tf.float32)

				self.encoder_outputs = tf.concat(encoder_outputs, 2)
				encoder_states = []
				for i in range(self.num_layers):
					if isinstance(encoder_state[0][i], tf.contrib.rnn.LSTMStateTuple):
						encoder_state_c = encoder_state[0][i].c + encoder_state[0][i].c
						encoder_state_h = encoder_state[0][i].h + encoder_state[0][i].h
						tmp_encoder_state = tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)
					elif isinstance(encoder_state[0][i], tf.Tensor):
						tmp_encoder_state = encoder_state[0][i] + encoder_state[0][i]
					encoder_states.append(tmp_encoder_state)
				self.encoder_state = tuple(encoder_states)

	def build_decoder(self):
		with tf.name_scope('decoder'):
			decoder_cell = tf.nn.rnn_cell.MultiRNNCell(
				[self.get_a_cell(self.lstm_size, self.keep_prob) for _ in range(self.num_layers)]
			)

			projection_layer = tf.layers.Dense(self.vocab_size, use_bias=False)

			attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.lstm_size, self.encoder_outputs)
			decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size=self.lstm_size)
			decoder_initial_state = decoder_cell.zero_state(self.batch_size, tf.float32).clone(cell_state=self.encoder_state)

			if self.mode == 'train':
				helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_inputs, self.y_seq_lengths)
			elif self.mode == 'sample':
				helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
					self.embedding,
					start_tokens=tf.fill([self.batch_size], 1), 
					end_token=2
				)
			
			decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, decoder_initial_state, projection_layer)

			self.decoder_outputs, self.decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
															# output_time_major=False,
															maximum_iterations=self.max_steps)

			self.logits = self.decoder_outputs.rnn_output
			self.sample_id = self.decoder_outputs.sample_id

	def build_loss(self):
		with tf.name_scope('loss'):
			crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.targets2)
			target_weights = tf.to_float(tf.sign(self.targets2))
			self.loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(tf.reduce_sum(self.y_seq_lengths))
			# mask = tf.sequence_mask(self.seq_lengths, self.max_steps, dtype=tf.float32)
			# self.loss = tf.contrib.seq2seq.sequence_loss(self.logits, self.targets2, mask)

	def build_optimizer(self):
		# clipping gradients
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
		train_op = tf.train.AdamOptimizer(self.learning_rate)
		self.optimizer = train_op.apply_gradients(zip(grads, tvars))

	def train(self, batch_generator, converter, max_steps, save_path, save_every_n, log_every_n):
		self.session = tf.Session()
		with self.session as sess:
			sess.run(tf.global_variables_initializer())
			# Train network
			n_iter = 0
			for x, y1, y2, seq_lengths in batch_generator:
				n_iter += 1
				start = time.time()
				feed = {self.inputs: x,
						self.targets1: y1,
						self.targets2: y2,
						self.keep_prob: self.train_keep_prob,
						self.x_seq_lengths: seq_lengths[0],
						self.y_seq_lengths: seq_lengths[1]}
				sample_id, batch_loss, decoder_state, _ = sess.run([self.sample_id,
																	self.loss,
																	self.decoder_state,
																	self.optimizer],
																	feed_dict=feed)

				end = time.time()
				# control the print lines
				if n_iter % log_every_n == 0:
					print('iter: {}/{}... '.format(n_iter, self.max_iters),
						  'loss: {:.4f}... '.format(batch_loss),
						  '{:.4f} sec/batch'.format((end - start)))
					print('input:')
					print(converter.idxs_to_words(x[0]))
					print('output:')
					print(converter.idxs_to_words(sample_id[0]))
					print('--------------------')
				if (n_iter % save_every_n == 0):
					self.saver.save(sess, os.path.join(save_path, 'model'), global_step=n_iter)
				if n_iter >= self.max_iters:
					break
			self.saver.save(sess, os.path.join(save_path, 'model'), global_step=n_iter)

	def sample(self, x):
		sess = self.session
		# new_state = sess.run(self.initial_state)
		feed = {self.inputs: [x],
				self.keep_prob: 1.,
				# self.initial_state: new_state,
				self.x_seq_lengths: [len(x)]
				}
		sample_id, new_state = sess.run([self.sample_id, self.decoder_state],
									feed_dict=feed)

		return sample_id

	def load(self, checkpoint):
		self.session = tf.Session()
		self.saver.restore(self.session, checkpoint)
		print('Restored from: {}'.format(checkpoint))
