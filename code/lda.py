import sys
import argparse
import numpy as np
import preprocessing
import itertools

class lda():

	def __init__(self, alpha, beta, nr_topics):
		"""
		Initialize values as global variables
		alpha: scalar for dirichlet distribution
		beta: scalar for dirichlet distribution
		nr_topics: scalar for number of desired topics
		"""
		self.alpha = alpha
		self.beta = beta
		self.nr_topics = nr_topics
		# Preprocess data
		prep = preprocessing.preprocessing(dump_files=False, load_files=False, dump_clean=False, load_clean=False)
		# Get lyrics
		dataset = prep.get_dataset()

		# Count unknowns:
		artists_unknown = [item['artist'] for item in dataset].count('unknown')
		title_unknown = [item['title'] for item in dataset].count('unknown')
		genre_unknown = [item['genre'] for item in dataset].count('unknown')
		subgenre_unknown = [item['subgenres'] for item in dataset].count(['unknown'])
		print "total unknown: artist: %i, title: %i, genre: %i, subgenre: %i" %(artists_unknown, title_unknown, genre_unknown, subgenre_unknown)

		# Get all genre and subgenres
		all_genres = prep.get_information_dictionary('genre', 'title').keys()
		print all_genres
		sys.exit()
		all_subgenres_nested = [item[3] for item in lyric_info]
		chain = itertools.chain(*all_subgenres_nested )
		#print set(list(chain))
		
		# Create vocabulary
		self.total_vocab = prep.get_vocabulary(lyrics)
		self._initialize_counts(lyrics)

	def _initialize_counts(self,lyrics):
		""" Initialize the counts of all words in documents """
		print "Count words.."
		# Initialize matrix for occurance words in documents [N x V]
		nr_lyrics = len(lyrics)
		vocab = self.total_vocab.keys()
		self.doc_word = np.zeros((nr_lyrics, len(vocab)))

		# Loop over documents:
		for i in range(0, nr_lyrics):
			# Loop over words in doc:
			for j in range(0, len(lyrics[i])): 
				word = lyrics[i][j]
				# Update word count in total vocabulary
				self.total_vocab[word] = self.total_vocab.get(word,0) + 1
				wordindex = vocab.index(word)
				self.doc_word[i][wordindex] += 1

		print self.doc_word[1][:]


	def start_lda(self):
		""" """
		# TODO: just put some functions here which are needed in lda
		# Get topic mixture distribution
		theta = self.dirichlet(self.alpha)
		# Pick a topic
		topic = self.sample_multinomial(theta)


	def sample_multinomial(self, distribution):
		"""
		Take multinomial distribution given specific distribution.
		Return index of chosen item
		"""
		return np.random.multinomial(1,distribution).argmax()

	def dirichlet(self, alpha):
		""" Sample from dirichlet distribution given dirichlet parameter
		Returns array of size nr_topics)
		"""
		return np.random.mtrand.dirichlet([alpha] * self.nr_topics)




if __name__ == "__main__":

	#Command line arguments
	parser = argparse.ArgumentParser(description="Run LDA")
	parser.add_argument('-a', metavar='Specify value for alpha', type=float)
	parser.add_argument('-b', metavar='Specify value for beta', type=float)
	parser.add_argument('-topics', metavar='Specify number of topics.', type=int)
	args = parser.parse_args()

	alpha = 0.01
	beta = 0.01
	nr_topics = 10

	if(vars(args)['a'] is not None):
		alpha = vars(args)['a']
	if(vars(args)['b'] is not None):
		beta = vars(args)['b']
	if(vars(args)['topics'] is not None):
		nr_topics = vars(args)['topics']

	lda = lda(alpha, beta, nr_topics)

	
