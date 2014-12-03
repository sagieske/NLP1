import sys
import argparse
import numpy as np
import preprocessing
import itertools
import random
import sklearn

class lda():
	"""
	Global variables
	- alpha: 		scalar for dirichlet distribution
	- beta: 		scalar for dirichlet distribution
	- nr_topics: 		scalar for number of desired topics
	- all_genres: 		scalar for total number of genres encountered
	Matrixes:
	- doc_word:		count of times word occurs in document
	- words_topics:		count of times word belongs to a topic
	- topics_genres:	count of times topic belongs to a genre (and indirectly the words)
	Dictionary
	- topics		dictionary with tuple (doc, wordposition), also (i,j), as key and value is topic that is assigned
	- vocab 		dictionary with vocabulary words as keys and scalar as index used for the matrices
	- genre_list		dictionary with genres as keys and scalar as index used for the matrices
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
		
		# Create vocabulary
		self.total_vocab = prep.get_vocabulary()
		# Initialization of matrices and dictionaries 
		self._initialize_lists()
		# Initialize counts for matrices
		self._initialize_counts()

	def _initialize_lists(self):
		"""	
		Initialize all matrices and dictionaries.
		Dictionaries vocab and genre_list are initialized with its key (word or genre) and as value a index created by counter.
		This is because dictionary lookups are faster than array .index() function
		"""
		# Initialize matrix for occurance words in documents [N x V]
		nr_lyrics = len(self.dataset)
		nr_vocab = len(self.total_vocab.keys())
		#self.vocab = self.total_vocab.keys()
		self.doc_word = np.zeros((nr_lyrics, nr_vocab),  dtype=int)
		
		# Save count for words assigned to a topic
		nr_genres = len(self.all_genres)
		self.words_topics = np.zeros((nr_vocab, self.nr_topics),  dtype=int) + beta
		self.topics_genres = np.zeros((self.nr_topics, nr_genres),  dtype=int) + alpha
		#self.genre_list = self.all_genres

		self.topics = {}
		self.vocab = {}
		self.genre_list = {}

		# initialize indices fro matrices
		vocab_index_counter= 0
		for item in self.total_vocab.keys():
			self.vocab[item] = vocab_index_counter
			vocab_index_counter += 1

		genre_index_counter = 0
		for genre in self.all_genres:
			self.genre_list[genre] = genre_index_counter
			genre_index_counter += 1


	def _initialize_counts(self):
		"""
		Initialize the counts of all words in documents
		Loop through all documents and get its genre.
		Loop through all words in document. Then choose random topic to initialize values in matrices
		"""

		print "Initialize counts.."
		# Get sizes
		nr_lyrics = len(self.dataset)
		nr_genres = len(self.all_genres)

		# Initialize all counts
		# Loop over documents:
		for i in range(0, nr_lyrics):
			# Get index of document and associated genre
			genre_index = self.genre_list[self.dataset[i]['genre']]
			# Get cleaned lyrics of item
			cleaned_lyrics =  self.dataset[i]['cleaned_lyrics']
			# Loop over words in doc:
			for j in range(0, len(cleaned_lyrics)): 
				word = cleaned_lyrics[j]
				# Update word count in total vocabulary
				self.total_vocab[word] = self.total_vocab.get(word,0) + 1
				wordindex = self.vocab[word]
				self.doc_word[i][wordindex] += 1

				# Choose random topic
				k = random.randint(0,self.nr_topics-1)

				# Update matrices
				self.words_topics[wordindex][k] +=1
				self.topics_genres[k][genre_index] += 1

				# Set topic of ij to k
				self.topics[(i,j)] = k


	def start_lda(self, N):
		""" """
		# TODO: just put some functions here which are needed in lda
		# Get topic mixture distribution
		theta = self.dirichlet(self.alpha)
		# Pick a topic
		topic = self.sample_multinomial(theta)
		print "start LDA!"

		nr_lyrics = len(self.dataset)
		# Do gibbs sampling N times for all items
		for iteration in range(0,N):
			# Loop through all documents
			for i in range(0, nr_lyrics):
				# get genre (and corresponding index)
				genre_index = self.genre_list[self.dataset[i]['genre']]
				cleaned_lyrics = self.dataset[i]['cleaned_lyrics']
				# Loop through all words
				for j in range(0, len(cleaned_lyrics)): 
					# Get word (and corresponding index)
					word = cleaned_lyrics[j]
					word_index = self.vocab[word]
					position = (i,j)

					current_topic = self.topics[position]

					# Get topic probability distribution
					p_zij = self.probability_topic(current_topic,word_index, genre_index)
					# Get topic index using topic distribution
					k = self.sample_multinomial(p_zij)

					# update matrices
					self.update(current_topic, position, word_index, genre_index, k)

	def update(self, previous_topic_index, position, word_index, genre_index, topic_index):
		"""
		Update values in matrices using indices. 
		Get previous topic that was assigned to word at that position.
		Subtract 1 from matrices
		"""
		# Subtract in matrices the old topic index
		self.words_topics[word_index][previous_topic_index] -= 1
		self.topics_genres[previous_topic_index][genre_index] -= 1

		# Add in matrices the new topic index
		self.words_topics[word_index][topic_index] +=1
		self.topics_genres[topic_index][genre_index] += 1

		# Update topic assignment
		self.topics[position] = topic_index
		#print "Updated word %i from topic %i to topic %i" %(word_index, previous_topic_index, topic_index)



	def probability_topic(self, current_topic, word_index, genre_index):
		"""
		Calculate probabilities of topics for word_ij and return array (sums to 1)
		"""
		# For each topic k:
			# ((beta + count_words_topic) / (aantal woorden * beta + topic count)) * ((alpha + count_genre_topic) / (aantal topics * alpha + count_genre))

		p_zij = np.zeros(self.nr_topics)

		for i in range(0, self.nr_topics):
			a = self.beta + self.count_words_topic(current_topic, word_index, i)
			b = len(self.vocab) * self.beta + self.count_topic(current_topic, i)
			c = self.alpha + self.count_genre_topic(current_topic, genre_index, i)
			d = self.nr_topics * self.alpha + self.count_genre(current_topic, genre_index)

			result = (a/float(b)) * (c/float(d))

			p_zij[i] = result

		# Normalize array to sum up to 1
		total = np.sum( p_zij,axis=0)
		p_zij = np.divide(p_zij, total)

		return p_zij

	def sample_multinomial(self, distribution):
		"""
		Take multinomial distribution given specific distribution.
		Return index of chosen item
		"""
		# numpy function cries when array does not sum up to one
		# (can occur due to float rounding errors) so catch
		try:
			return np.random.multinomial(1,distribution).argmax()

		except:
			newlist = []
			if sum(distribution[:-1]) >= 1.0:
				extra = 1.0000000000000000000000001 - sum(distribution[:-1])
				for index in range(0,len(distribution)):
					newlist.append(distribution[index] - extra/float(len(distribution)))
				print "TEST"
			else:
				newlist = distribution
			if sum(newlist[:-1]) > 1.0:
				print "STILLLL"
				# Set default list
				newlist = [1/float(len(distribution))] * len(distribution)
			return np.random.multinomial(1,newlist).argmax()


	def dirichlet(self, alpha):
		""" Sample from dirichlet distribution given dirichlet parameter
		Returns array of size nr_topics
		"""
		return np.random.mtrand.dirichlet([alpha] * self.nr_topics)

	def count_words_topic(self, current_topic, wordindex, topic):
		""" Count the number of times this similar word(wordindex) is associated with topic, excluding the word at given position"""
		# If word has same topic, remove 1 from count
		if current_topic == topic:
			return self.words_topics[wordindex, topic] - 1
		else:
			return self.words_topics[wordindex, topic] 

		

	def count_topic(self, current_topic, topic):
		""" Count the total number of words associated with topic, excluding word at given position"""
		# Excluded word is associated with this topic, so subtract 1
		if current_topic == topic:
			return sum(self.words_topics[:, topic]) -1
		# Current word is not associated with this topic
		else:
			return sum(self.words_topics[:, topic])

	def count_genre_topic(self, current_topic, genre_index, topic):
		""" Count the number of times this specific topic is associated with the genre, excluding the word at given position"""
		# Topic of excluding word is associated with this topic, so subtract 1
		if current_topic == topic:
			return self.topics_genres[topic, genre_index] -1 
		# Topic of excluding word is not associated with this topic
		else:
			return self.topics_genres[topic, genre_index] 

	def count_genre(self, current_topic, genre_index):
		""" Count the total number of topics associated with genre, excluding topic of word at given position"""
		# Topic of excluding word is associated with this genre, so subtract 1
		if self.topics_genres[current_topic, genre_index] > 0:		
			return sum(self.topics_genres[:, genre_index]) -1
		# Topic of excluding word is not associated with this genre
		else: 
			return sum(self.topics_genres[:, genre_index])



if __name__ == "__main__":

	#Command line arguments
	parser = argparse.ArgumentParser(description="Run LDA")
	parser.add_argument('-a', metavar='Specify value for alpha', type=float)
	parser.add_argument('-b', metavar='Specify value for beta', type=float)
	parser.add_argument('-topics', metavar='Specify number of topics.', type=int)
	args = parser.parse_args()

	alpha = 0.1
	beta = 0.4
	nr_topics = 10

	if(vars(args)['a'] is not None):
		alpha = vars(args)['a']
	if(vars(args)['b'] is not None):
		beta = vars(args)['b']
	if(vars(args)['topics'] is not None):
		nr_topics = vars(args)['topics']

	lda = lda(alpha, beta, nr_topics)
	lda.start_lda(1)

	
