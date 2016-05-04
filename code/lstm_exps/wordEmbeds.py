import cPickle
from gensim.models.word2vec import Word2Vec

model = Word2Vec.load_word2vec_format('../../data/GoogleNews-vectors-negative300.bin', 
                                             binary=True)

def tree_to_seq(tree):
	try:
		words = tree.get_words()

	except:
		words = [tree.word]

	seq = []

	for word in words:
		if word in model:
			seq.append(model[word])
		else:
			seq.append([0. for i in range(300)])

	return seq

def get_vectors():

	all_trees = cPickle.load(open("../../data/lstmTrees.pkl"))

	test_trees = all_trees[('Liberal', 'Conservative')]

	libSeq = []
	repSeq = []

	for t1, t2 in test_trees:
		libSeq.append(tree_to_seq(t1))
		repSeq.append(tree_to_seq(t2))

	return libSeq, repSeq

