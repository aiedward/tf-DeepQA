import numpy as np
import copy
import time
import tensorflow as tf
import pickle

def get_batch(batch, max_len):
	batch_x = []
	batch_y1 = []
	batch_y2 = []
	x_seq_lengths = []
	y_seq_lengths = []

	x_max_steps = max_len
	# x_max_steps = 0
	# for d in batch:
	# 	if len(d[1]) > x_max_steps:
	# 		x_max_steps = len(d[1])
	# if max_len < x_max_steps:
	# 	x_max_steps = max_len

	# y_max_steps = max_len
	y_max_steps = 0
	for d in batch:
		if len(d[1]) > y_max_steps:
			y_max_steps = len(d[1])
	if max_len < y_max_steps:
		y_max_steps = max_len

	for d in batch:
		x = d[0]
		y = d[1]

		x_len = len(x)
		x_seq_lengths.append(x_len)
		if x_len >= x_max_steps:
			batch_x.append(x[:x_max_steps])
		else:
			batch_x.append(x + [0 for i in range(x_max_steps - x_len)])

		y_len = len(y)
		if y_len >= y_max_steps:
			batch_y1.append([1] + y[:y_max_steps - 1])
			batch_y2.append(y[:y_max_steps - 1] + [2])
			y_seq_lengths.append(y_max_steps)
		else:
			batch_y1.append([1] + y + [0 for i in range(y_max_steps - y_len - 1)])
			batch_y2.append(y + [2] + [0 for i in range(y_max_steps - y_len - 1)])
			y_seq_lengths.append(y_len + 1)

		seq_lengths = [x_seq_lengths, y_seq_lengths]

	return batch_x, batch_y1, batch_y2, seq_lengths

def batch_generator(data, batch_size, max_steps):
	data = copy.copy(data)
	n_batches = int(len(data) / batch_size)
	data = data[:batch_size * n_batches]
	epoch = 1
	while True:
		print('----------')
		print('Epoch %d' % epoch)
		print('----------')
		np.random.shuffle(data)
		for i in range(n_batches):
			batch_x, batch_y1, batch_y2, seq_lengths = get_batch(data[i*batch_size:(i+1)*batch_size], max_steps)
			x = batch_x
			y1 = batch_y1
			y2 = batch_y2
			yield x, y1, y2, seq_lengths
		epoch += 1

class TextConverter(object):
	def __init__(self, lang=None, max_vocab=10000, filename=None):
		if filename is not None:
			with open('./converter/' + filename, 'rb') as f:
				self.lang = pickle.load(f)
		else:
			self.lang = lang
		self.vocab_size = self.lang.vocab_size

	def idxs_to_words(self, idx_sentence):
		s = ''
		for idx in idx_sentence:
			w = self.lang.index2word[idx]
			if w == '<EOS>':
				# s += '. '
				# s += w
				break
			else:
				s += w + ' '
		return s.capitalize()

	def sentence_to_idxs(self, sentence):
		idxs = []
		words = sentence.split(' ')
		for word in words:
			word = word.lower().replace('.', '').replace(',', '').replace('!', '').replace('?', '')
			try:
				idx = self.lang.word2index[word]
			except KeyError:
				idx = self.lang.word2index['<UNK>']
			idxs.append(idx)
		return idxs

	def save_lang(self, filename):
		with open('./converter/' + filename, 'wb') as f:
			pickle.dump(self.lang, f)

	def beam_to_sentences(self, predicted_ids, parent_ids):
		depth = len(predicted_ids)
		width = len(predicted_ids[0])
		sentences = [[] for _ in range(width)]
		for i in range(width):
			sentences[i].append(predicted_ids[depth - 1][i])
			tmp_idx = i
			for d in range(depth - 1):
				sentences[i].append(predicted_ids[depth - 2 - d][parent_ids[depth - 1 - d][tmp_idx]])
				tmp_idx = parent_ids[depth - 1 - d][tmp_idx]

		s = []
		for idxs in sentences:
			s.append(self.idxs_to_words(list(reversed(idxs))))

		return s

