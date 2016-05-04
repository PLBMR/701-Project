import cPickle
from gensim.models.word2vec import Word2Vec


model = Word2Vec.load_word2vec_format('../../data/GoogleNews-vectors-negative300.bin', 
                                             binary=True)

all_words = {}

def tree_to_seq(tree):

	try:
		words = tree.get_words().split()

	except:
		words = [tree.word]


	seq = []

	for word in words:
		vec = []
		if word in model:
			vec = model[word]
		else:
			seq.append([0. for i in range(300)])
			vec = [0.0001 for i in range(300)]

		seq.append(vec)
		all_words[word] = vec

	# TODO: comment and fix this hack..
	print words
	return words

def get_data():

	all_trees = cPickle.load(open("../../data/lstmTrees.pkl"))

	test_trees = all_trees[('Liberal', 'Conservative')]

	libSeq = []
	repSeq = []

	for t1, t2 in test_trees:
		libSeq.append(tree_to_seq(t1))
		repSeq.append(tree_to_seq(t2))

	return libSeq, repSeq, all_words

