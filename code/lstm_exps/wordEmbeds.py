import cPickle
from gensim.models.word2vec import Word2Vec


model = Word2Vec.load_word2vec_format('../../data/GoogleNews-vectors-negative300.bin', 
                                             binary=True)

all_words = {}

presidents = ['obama', 'clinton', 'biden', 'edwards', 'richardson', 'gingrich'
                'bachmann', 'cain', 'giuliani', 'huckabee', 'huntsman', 'mccain',
                 'paul', 'pawlenty', 'perry', 'romney', 'santorum', 'thompson']

def tree_to_seq(tree):

	try:
		words = tree.get_words().split()

	except:
		words = [tree.word]


	seq = []

	for word in words:
		if word in presidents:
			continue
		vec = []
		if word in model:
			vec = model[word]
		else:
			# seq.append([0. for i in range(300)])
			vec = [0.0001 for i in range(300)]

		# seq.append(vec)
		all_words[word] = vec

	# TODO: comment and fix this hack..
	return words

def get_data():

	all_trees = cPickle.load(open("../../data/lstmTrees.pkl"))


	libSeq = []
	repSeq = []

	test_trees = all_trees[('Liberal', 'Conservative')]
	for t1, t2 in test_trees:
		libSeq.append(tree_to_seq(t1))
		repSeq.append(tree_to_seq(t2))

	test_trees = all_trees[('Conservative', 'Liberal')]
	for t1, t2 in test_trees:
		repSeq.append(tree_to_seq(t1))
		libSeq.append(tree_to_seq(t2))

	test_trees = all_trees[('Liberal', 'Liberal')]
	for t1, t2 in test_trees:
		libSeq.append(tree_to_seq(t1))
		libSeq.append(tree_to_seq(t2))

	test_trees = all_trees[('Conservative', 'Conservative')]
	for t1, t2 in test_trees:
		repSeq.append(tree_to_seq(t1))
		repSeq.append(tree_to_seq(t2))

	return libSeq, repSeq, all_words

def get_data2():

	demTrees, repTrees, neutral = cPickle.load(open("../../data/full_ibc/ibcData.pkl"))

	demWords = []

	for tree in demTrees:
		demWords.append(tree_to_seq(tree))

	repWords = []
	for tree in repTrees:
		repWords.append(tree_to_seq(tree))

	return demWords, repWords, all_words

def get_data3():

	demTrees, repTrees, neutral = cPickle.load(open("../../baselines/PSC.pkl"))


	demWords = []

	for tree in demTrees:
		demWords.append(tree_to_seq(tree))

	repWords = []
	for tree in repTrees:
		repWords.append(tree_to_seq(tree))

	return demWords, repWords, all_words