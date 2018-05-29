from preprocess import readfiles

class Lang:
	def __init__(self, name):
		self.name = name
		self.word2index = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
		self.word2count = {'<PAD>': 0, '<SOS>': 0, '<EOS>': 0, '<UNK>': 0}
		self.index2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
		# self.out2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
		self.vocab_size = len(self.word2index)
		# self.out_vocab_size = len(self.out2word)

	def addword(self, word):
		word = word_clean(word)
		if word not in self.word2index:
			self.word2index[word] = self.vocab_size
			self.word2count[word] = 1
			self.index2word[self.vocab_size] = word
			self.vocab_size += 1
		else:
			self.word2count[word] += 1

def word_clean(word):
	cleaned_word = word.lower().replace('.', '').replace(',', '').replace('!', '').replace('?', '')
	return cleaned_word

def createLang(lines, lang_name):
	lang = Lang(lang_name)
	for line_id, line in lines.items():
		words = line.split(' ')
		for word in words:
			lang.addword(word)

	# print('Lang %s is created.' % lang_name)
	# print('total %d words in the lang.' % lang.vocab_size)

	for w, c in lang.word2count.items():
		if c < 3 and w not in ['<SOS>', '<EOS>', '<PAD>', '<UNK>']:
			lang.index2word.pop(lang.word2index[w])
			lang.word2index.pop(w)
			lang.word2count[w] = 0
			lang.word2count['<UNK>'] += 1
			lang.vocab_size -= 1

	new_w2idx = dict()
	new_idx2w = dict()
	count = 0
	for w, idx in lang.word2index.items():
		new_idx2w[count] = w
		new_w2idx[w] = count
		count += 1

	lang.word2index = new_w2idx
	lang.index2word = new_idx2w

	print('Lang %s is created.' % lang_name)
	print('total %d words in the lang.' % lang.vocab_size)
	# print('total %d output choices in the lang.' % lang.out_vocab_size)
	print('-------------------')

	return lang

def lines2index(lines_data, lang):
	def findword2index(lang, word):
		try:
			return lang.word2index[word]
		except KeyError:
			return lang.word2index['<UNK>']

	indexed_lines = dict()
	for line_id, line in lines_data.items():
		words = line.split()
		indexed_line = []
		for word in words:
			word = word_clean(word)
			index = findword2index(lang, word)
			indexed_line.append(index)
		indexed_lines[line_id] = indexed_line

	print('total %d indexed lines.' % len(indexed_lines))

	return indexed_lines

def createTrainData(conversations, indexed_lines, max_len):
	train_data = []
	for con in conversations:
		x_id = con[0]
		y_id = con[1]
		x = indexed_lines[x_id]
		y = indexed_lines[y_id]
		if 3 in y or len(x) > max_len or len(y) > max_len:
			continue
		else:
			train_data.append((x, y))

	print('total %d train pairs.' % len(train_data))
	print('-------------------')

	return train_data

def loaddata(path, max_len=10):
	conversations, movie_lines, w2v = readfiles(path)
	lang = createLang(movie_lines, 'movie')
	indexed_lines = lines2index(movie_lines, lang)
	data = createTrainData(conversations, indexed_lines, max_len)

	return data, lang, w2v
