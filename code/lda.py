import sys
import argparse
import numpy as np
import preprocessing
import itertools
import random

class lda():
	"""
	Global variables
	- alpha: 		scalar for dirichlet distribution
	- beta: 		scalar for dirichlet distribution
	- nr_topics: 		scalar for number of desired topics
	- all_genres: 		scalar for total number of genres encountered
	- vocab 		list of vocabulary words used for indices
	- genre_list		list of genres used for indices

	Matrixes:
	- doc_word:		count of times word occurs in document
	- words_topics:		count of times word belongs to a topic
	- topics_genres:	count of times topic belongs to a genre (and indirectly the words)
	"""

	def __init__(self, alpha, beta, nr_topics):
		""" Initialize
		"""
		self.alpha = alpha
		self.beta = beta
		self.nr_topics = nr_topics
		# Preprocess data
		prep = preprocessing.preprocessing(dump_files=False, load_files=False, dump_clean=False, load_clean=False)
		# Get lyrics
		self.dataset = prep.get_dataset()

		# Count unknowns:
		artists_unknown = [item['artist'] for item in self.dataset].count('unknown')
		title_unknown = [item['title'] for item in self.dataset].count('unknown')
		genre_unknown = [item['genre'] for item in self.dataset].count('unknown')
		subgenre_unknown = [item['subgenres'] for item in self.dataset].count(['unknown'])
		print "total unknown: artist: %i, title: %i, genre: %i, subgenre: %i" %(artists_unknown, title_unknown, genre_unknown, subgenre_unknown)

		# Get all genre and subgenres
		all_genres = prep.get_information_dictionary('genre', 'title').keys()
		self.all_genres =  all_genres
		#all_subgenres_nested = [item[3] for item in lyric_info]
		#chain = itertools.chain(*all_subgenres_nested )
		#print set(list(chain))
		
		# Create vocabulary
		self.total_vocab = prep.get_vocabulary()
		self._initialize_counts()

	def _initialize_counts(self):
		""" Initialize the counts of all words in documents """
		print "Count words.."
		# Initialize matrix for occurance words in documents [N x V]
		nr_lyrics = len(self.dataset)
		self.vocab = self.total_vocab.keys()
		self.doc_word = np.zeros((nr_lyrics, len(vocab)))
		
		# Save count for words assigned to a topic
		nr_genres = len(self.all_genres)
		self.words_topics = np.zeros((len(vocab), self.nr_topics))
		self.topics_genres = np.zeros((self.nr_topics, nr_genres))
		self.genre_list = self.all_genres


		# Initialize all counts
		# Loop over documents:
		for i in range(0, nr_lyrics):
			genre_i = self.dataset[i]['genre']
			genre_index = genre_list.index(genre_i)
			#print "Genre of %i : %s (index: %i)" %(i, genre_i, genre_index)
			# Loop over words in doc:
			cleaned_lyrics =  self.dataset[i]['cleaned_lyrics']
			for j in range(0, len(cleaned_lyrics)): 
				word = cleaned_lyrics[j]
				# Update word count in total vocabulary
				self.total_vocab[word] = self.total_vocab.get(word,0) + 1
				wordindex = vocab.index(word)
				self.doc_word[i][wordindex] += 1

				# CHoose random topic
				k = random.randint(0,self.nr_topics-1)
				# Update lists
				self.words_topics[wordindex][k] +=1
				self.topics_genres[k][genre_index] += 1


	def start_lda(self):
		""" """
		# TODO: just put some functions here which are needed in lda
		# Get topic mixture distribution
		theta = self.dirichlet(self.alpha)
		# Pick a topic
		topic = self.sample_multinomial(theta)


		nr_lyrics = len(self.dataset)
		# Loop through all documents
		for i in range(0, nr_lyrics):
			genre_i = self.dataset[i]['genre']
			genre_index = genre_list.index(genre_i)
			cleaned_lyrics = self.dataset[i]['cleaned_lyrics']
			# Loop through all words
			for j in range(0, len(cleaned_lyrics)): 
				word = cleaned_lyrics[j]
				# Get index of vocabulary
				wordindex = vocab.index(word)
				p_zij = self.probability_topic(word_index, genre_index)
				k = sample_multinomial(p_zij)

				# UPDATE ALL
				print "a"

	def probability_topic(self, word_index, genre_index):
		p_zij = np.random.rand(self.nr_topics)
		return p_zij

	def sample_multinomial(self, distribution):
		"""
		Take multinomial distribution given specific distribution.
		Return index of chosen item
		"""
		return np.random.multinomial(1,distribution).argmax()

	def dirichlet(self, alpha):
		""" Sample from dirichlet distribution given dirichlet parameter
		Returns array of size nr_topics
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

	
