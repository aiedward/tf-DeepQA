import codecs

def readfiles(path):
	# movie conversations
	conversations = []
	with codecs.open(path + 'movie_conversations.txt', 'r', encoding='utf-8', errors='ignore') as f:
		lines = f.readlines()
		for line in lines:
			tmp = line.split(' +++$+++ ')
			con = tmp[3].replace('[', '').replace(']', '').replace('\'', '').replace('\n', '').split(', ')
			for i in range(len(con) - 1):
				conversations.append((con[i], con[i + 1]))

	print('total %d conversation pairs.' % len(conversations))

	# movie lines
	movie_lines = dict()
	with codecs.open(path + 'movie_lines.txt', 'r', encoding='utf-8', errors='ignore') as f:
		lines = f.readlines()
		for line in lines:
			tmp = line.split(' +++$+++ ')
			movie_lines[tmp[0]] = tmp[4]

	print('total %d movie lines.' % len(movie_lines))
	print('-------------------')

	# pre-trained w2v
	w2v = dict()
	with codecs.open('./glove.6B.300d.txt', 'r', encoding='utf8', errors='ignore') as f:
		for line in f:
			tmp = line.split()
			w2v[tmp[0]] = [float(i) for i in tmp[1:]]

	print('total %d word2vec lines.' % len(w2v))
	print('-------------------')

	return conversations, movie_lines, w2v


