import sys
import argparse
import numpy as np
import preprocessing
import itertools
import random
import sklearn
import ast
import time

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import re




# try:
#     import cPickle as pickle
# except:
import pickle

class lda():
	"""
	Instance variables
	- alpha: 			scalar for dirichlet distribution
	- beta: 			scalar for dirichlet distribution
	- nr_topics: 		scalar for number of desired topics
	- all_genres: 		scalar for total number of genres encountered
	Array
	- genre_count		array of total count of assignments per genre
	- topic_count 		array of total count of word assignments per topic
	Matrixes:
	- doc_word:			count of times word occurs in document
	- words_topics:		count of times word belongs to a topic
	- topics_genres:	count of times topic belongs to a genre (and indirectly the words)
	Dictionary
	- topics			dictionary with tuple (doc, wordposition), also (i,j), as key and value is topic that is assigned
	- vocab 			dictionary with vocabulary words as keys and scalar as index used for the matrices
	- genre_list		dictionary with genres as keys and scalar as index used for the matrices
	- index_to_vocab	dictionary which maps index to word (reverse of vocab dictionary)
	- index_to_genre	dictionary which maps index to genre (reverse of genre_list dictionary)
	"""
	#TODO: preprocessing now only words with 1000 for testing purposes. It is so goddamn slow

	def __init__(self, alpha, beta, nr_topics, load_init=False, skip_lda=False):
		""" Initialize
		TODO: load_init is to be used for initialization from pickle load from file. NOT USED YET!
		"""
		self.alpha = alpha
		self.beta = beta
		self.nr_topics = nr_topics
		self.skiplda = skip_lda
		# Preprocess data
		prep = preprocessing.preprocessing(dump_files=False, load_files=True, dump_clean=False, load_clean=True)
		# Get lyrics
		self.dataset = prep.get_dataset()
		# Use smaller dataset add [:set]
		print "total nr of lyrics:", len(self.dataset)

		# Count unknowns:
		artists_unknown = [item['artist'] for item in self.dataset].count('unknown')
		title_unknown = [item['title'] for item in self.dataset].count('unknown')
		genre_unknown = [item['genre'] for item in self.dataset].count('unknown')
		subgenre_unknown = [item['subgenres'] for item in self.dataset].count(['unknown'])
		print "total unknown: artist: %i, title: %i, genre: %i, subgenre: %i" %(artists_unknown, title_unknown, genre_unknown, subgenre_unknown)

		# Get all genre and subgenres
		all_genres = prep.get_information_dictionary('genre', 'title').keys()
		self.all_genres =  all_genres

		self.genre_count = np.zeros(len(all_genres), dtype=int)
		self.topic_count = np.zeros(nr_topics, dtype=int)

		# Initialization of matrices and dictionaries 
		self._initialize_lists(load=load_init)
		# Initialize counts for matrices
		self._initialize_counts(load=True)

	def _initialize_lists(self, load=False):
		"""	
		Initialize all matrices and dictionaries.
		Dictionaries vocab and genre_list are initialized with its key (word or genre) and as value a index created by counter.
		This is because dictionary lookups are faster than array .index() function
		"""
		# Get all words
		sublist_words = [lyriclist['cleaned_lyrics'] for lyriclist in self.dataset]
		# Get set for vocabulary
		all_words = list(set([item for sublist in sublist_words for item in sublist]))

		# Create indices for words and viceversa
		self.index_to_vocab =  dict(enumerate(all_words))
		self.vocab = dict((y,x) for x,y in self.index_to_vocab.iteritems())

		# Create indices for genres and viceversa
		self.index_to_genre = dict(enumerate(self.all_genres)) 
		self.genre_list = dict((y,x) for x,y in self.index_to_genre.iteritems())

		# Initialize topic dictionary
		self.topics = {}

		# Save count for words assigned to a topic
		nr_genres = len(self.all_genres)
		nr_lyrics = len(self.dataset)
		nr_vocab = len(all_words)

		# sys.exit()

		# Initialize matrices
		self.words_topics = np.zeros((nr_vocab, self.nr_topics),  dtype=int)
		self.topics_genres = np.zeros((self.nr_topics, nr_genres),  dtype=int)
		# Initialize matrix for occurance words in documents [N x V]
		self.doc_word = np.zeros((nr_lyrics,len(all_words)),  dtype=int)


	def _initialize_counts(self, load=False):
		"""
		Initialize the counts of all words in documents
		Loop through all documents and get its genre.
		Loop through all words in document. Then choose random topic to initialize values in matrices
		"""

		if self.skiplda:
			self.load_data('inputlda')
		elif load:
			self.load_data('init_data')
		else:

			print "Initialize counts.."
			# Get sizes
			nr_lyrics = len(self.dataset)
			nr_genres = len(self.all_genres)

			

			# Initialize all counts
			# Loop over documents:
			for i in range(0, nr_lyrics):
				# Get index of document and associated genre
				genre_index = self.genre_list[self.dataset[i]['genre']]
				self.genre_count[genre_index] += 1
				# Get cleaned lyrics of item
				cleaned_lyrics =  self.dataset[i]['cleaned_lyrics']
				# Loop over words in doc:
				for j in range(0, len(cleaned_lyrics)): 
					word = cleaned_lyrics[j]
					# Update word count in total vocabulary
					wordindex = self.vocab[word]
					self.doc_word[i][wordindex] += 1

					# Choose random topic
					k = random.randint(0,self.nr_topics-1)
					self.topic_count[k] += 1

					# Update matrices
					self.words_topics[wordindex][k] +=1
					self.topics_genres[k][genre_index] += 1

					# Set topic of ij to k
					self.topics[(i,j)] = k

			self.dump_data('init_data')


	def start_lda(self, N, topwords, toptopics, filename):
		""" """
		# TODO: just put some functions here which are needed in lda
		# Get topic mixture distribution
		#theta = self.dirichlet(self.alpha)
		# Pick a topic
		#topic = self.sample_multinomial(theta)

		print "start LDA!"
		start = time.time()

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
				if i % 100 == 0 and i != 0:
					print "- lyrics done: %i" %(i)
			print "done iteration %i (stopwatch: %s)" %(iteration, str(time.time()-start))
			self.print_to_file(N, topwords, toptopics, filename, iteration)

		

		# prints initialization
		if N == 0:
			iteration = N
			#self.print_to_file(N, topwords, toptopics, filename, iteration)

		self.dump_data("iter" + str(N) + "_a" + str(self.alpha) + "_b" + str(self.beta) + "_topics" + str(self.nr_topics))

		#self.dump_data('done_data')		

	def update(self, previous_topic_index, position, word_index, genre_index, topic_index):
		"""
		Update values in matrices using indices. 
		Get previous topic that was assigned to word at that position.
		Subtract 1 from matrices
		"""
		# Subtract in matrices the old topic index
		self.words_topics[word_index][previous_topic_index] -= 1
		self.topics_genres[previous_topic_index][genre_index] -= 1
		self.topic_count[previous_topic_index] -= 1

		# Add in matrices the new topic index
		self.words_topics[word_index][topic_index] +=1
		self.topics_genres[topic_index][genre_index] += 1
		self.topic_count[topic_index] += 1

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


	def dump_data(self, filename):
		to_dump = {}
		to_dump['topics'] = self.topics
		to_dump['alpha'] = self.alpha
		to_dump['beta'] = self.beta
		to_dump['nr_topics'] = self.nr_topics
		to_dump['words_topics'] = self.words_topics
		to_dump['topics_genres'] = self.topics_genres
		print "Dump information to dumpfile: ", filename
		pickle.dump(to_dump, open(filename,"wb"))

	def load_data(self, filename):
		try:
			print "Load data from file: ", (filename)
			with open(filename,'r') as f:
				dumped = pickle.load(f)
				self.topics = dumped['topics']
				self.alpha = dumped['alpha']
				self.beta = dumped['beta']
				self.nr_topics = dumped['nr_topics']
				self.words_topics = dumped['words_topics']
				self.topics_genres = dumped['topics_genres']
		except:
			print "File %s corrupted, not found or Memory Error." %(filename)
			if self.skiplda:
				sys.exit()
			self._initialize_counts(False)



	def document_topic_distribution(self):
		""" Create array of topic distribution per document.
		Returns matrix of topic distribution per document and array of the genre corresponding to each document
		"""
		nr_lyrics = len(self.dataset)
		# Initialize matrix. Every lyric has array of len topics
		dt_dist = np.zeros((nr_lyrics, self.nr_topics))
		genres = []
		for i in range(0, nr_lyrics):
			# Get every topic assignment for every word, add to array of distributions
			for j in range(0, len(self.dataset[i]['cleaned_lyrics'])):
				k = self.topics[(i, j)]
				dt_dist[i][k] += 1
			# Normalize
			total = np.sum(dt_dist[i], axis=0)
			dt_dist[i] = np.divide(dt_dist[i], total)
			# Add current genre to list
			genres.append(self.dataset[i]['genre'])
		return genres, dt_dist


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
			#return sum(self.words_topics[:, topic]) -1
			return self.topic_count[topic] - 1
		# Current word is not associated with this topic
		else:
			# return sum(self.words_topics[:, topic])
			return self.topic_count[topic]

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
			# return sum(self.topics_genres[:, genre_index]) -1
			return self.genre_count[genre_index] - 1
		# Topic of excluding word is not associated with this genre
		else: 
			# return sum(self.topics_genres[:, genre_index])
			return self.genre_count[genre_index]


	def get_top_words_topic(self, topic_index, nr_top):
		"""
		Get # top words associated with topic. 
		"""
		# Get vector of the counts per word associated with this topic
		vector_words = self.words_topics[:, topic_index]
		# Get indices of words with highest counts
		indices_max = vector_words.argsort()[-nr_top:][::-1]
		return indices_max

	def get_from_indices(self, indices_array, dict_type):
		"""
		Return words corresponding to dictionary. Dict_type can be 'genre' or 'word' which correspond to global dictionary choice
		"""
		# Choose dictionary
		if dict_type == 'genre':
			chosen_dict = self.index_to_genre
		elif dict_type == 'words':
			chosen_dict = self.index_to_vocab
		else:
			print "wrong input dict. Now using dictionary for words"
			chosen_dict = self.index_to_vocab
		# Get all words corresponding to indices
		string_list = []
		for index in indices_array:
			string_list.append(chosen_dict[index])
		return string_list

	def get_top_genre(self, genre_index, nr_topics, nr_words):
		"""
		Get # top words associated with topic of top topics that are associated with genre.
		Returns a list of list of words. Each sublist corresponds to the top topic and its elements are the top words for a topic. 
		"""
		# Get vector of the counts per topic associated with this genre
		vector_topics = self.topics_genres[:, genre_index]
		# Get indices of topics with highest counts
		indices_max_topics = vector_topics.argsort()[-nr_topics:][::-1]

		top_topics = []
		# For every topic, get their top words and append to list
		for topic_index in indices_max_topics:
			indices_max_words = self.get_top_words_topic(topic_index, nr_words)
			top_words = self.get_from_indices(indices_max_words, 'words')
			top_topics.append(top_words)
		return indices_max_topics ,top_topics

	def print_to_file(self, runs, top_words, top_topics, filename, iteration):
		"""
		Write top words per topic and top topics per genre to file
		"""
		# Get top words per topic
		words_topic_total = []
		for i in range(0,self.nr_topics):
			max_indices = self.get_top_words_topic(i, top_words)
			words_topic = self.get_from_indices(max_indices, 'words')
			words_topic_total.append(words_topic)

		# Get top topics per genre
		genre_list_total = []
		indices_max_topics_total  = []
		for i in range(0,len(self.all_genres)):
			# Genre list is list of topics, which are represented as list of words. Indices_max_topics are the indices of topics
			indices_max_topics, genre_list = self.get_top_genre(i, top_topics, top_words)
			genre_list_total.append(genre_list)
			indices_max_topics_total.append(indices_max_topics)

	
		total_filename_topics = filename + "_" + str(iteration) + "_topics.txt"
		total_filename_genre = filename + "_" + str(iteration) + "_genre.txt"
		# Write to file for topics (if not exists open!)
		print "write to file: %s" %(filename)
		with open(total_filename_topics, 'w+') as f:
			f.write('Runs: %i, alpha: %.2f, beta: %.2f, nr topics: %i, nr genres: %i, iteration: %i, top_topics: %i, top_words: %i\n\n' \
				%(runs, self.alpha, self.beta, self.nr_topics, len(self.all_genres), iteration, top_topics, top_words) )
			f.write('TOPIC-WORD DISTRIBUTION at iteration %i/%i \n' %(iteration, runs))
			# Print list for every topic
			for i in range(0,len(words_topic_total)):
        			f.write('Topic %i\n%s\n' %(i, str(words_topic_total[i])))

		# Write to file for genres (if not exists open!)
		with open(total_filename_genre, 'w+') as f:
			f.write('Runs: %i, alpha: %.2f, beta: %.2f, nr topics: %i, nr genres: %i, iteration: %i, top_topics: %i, top_words: %i\n\n' \
				%(runs, self.alpha, self.beta, self.nr_topics, len(self.all_genres), iteration, top_topics, top_words) )
			f.write('GENRE-TOPIC DISTRIBUTION at iteration %i/%i \n' %(iteration, runs))
			# Print every genre
			for i in range(0,len(genre_list_total)):
        			f.write('\nGENRE %i (%s)\t (topics: %s)\n' %(i, self.all_genres[i], str(indices_max_topics_total[i]) ) )
				# Print every topic
				for j in range(0,len(genre_list)):
					f.write('--> Topic %i\n' %(indices_max_topics_total[i][j]))


					nr_words_topic = self.count_topic(0, indices_max_topics_total[i][j])
					to_print = "\t"
					for h in range(0, len(words_topic_total[indices_max_topics_total[i][j]])):
						curword = words_topic_total[indices_max_topics_total[i][j]][h]
						nr_word_in_topic = self.count_words_topic(0, self.vocab[curword], indices_max_topics_total[i][j])
						prob = nr_word_in_topic/float(nr_words_topic)
						to_print += "'%s' (probability: %0.5f), " %(curword, prob)

					f.write('%s\n' %(to_print))


					f.write('%s\n' %(str(words_topic_total[indices_max_topics_total[i][j]])))


	def genre_profiles(self):
		"""
		Create profiles for genres by using the topic distributions found
		"""

		# Get the topic distribution for every document and the corresponding genre
		genres, distr = lda.document_topic_distribution()
		genre_indices = []
		# Loop over all possible genres
		for genre in self.all_genres:
			print genre
			# Get indices of documents that belong to this genre
			indices = [i for i, x in enumerate(genres) if x == genre]
			genre_indices.append(indices)
			# Get topic distributions for all documents belonging to this genre
			genre_matrix =  np.array([x for i, x in enumerate(distr) if i in indices])

			# Get mean per topic
			mean_genre =np.mean(genre_matrix, axis=0)*100
			stdev_genre = np.std(genre_matrix, axis=0)*100

			print mean_genre
			print stdev_genre
			print len(mean_genre)
			print len(stdev_genre)


			fig = plt.figure()
			ax = fig.add_subplot(111)
			ax.set_title('Genre: %s' %genre)
			ind = np.arange(len( mean_genre))
			width = 0.35
			ax.bar(ind, mean_genre, width,  align='center', yerr=stdev_genre, ecolor='k')
			ax.set_ylabel('Mean')
			ax.set_xticks(ind)
			#plt.xticks(ind,('Young Male','Young Female','Elderly Male','Elderly Female'))
			#self.autolabel(mean_genre,peakval) 
			genre = re.sub('/', '-', genre)
			#if genre == 'pop/rock':
			#	genre = 'pop-rock'
			plt.savefig("%s.png" %genre)
			plt.close('all')

	def autolabel(self, bars,peakval):
	    for ii,bar in enumerate(bars):
		height = bars[ii]
		plt.text(ind[ii], height-5, '%s'% (peakval[ii]), ha='center', va='bottom')

	def load_new_document(self, document_string):
		''' load new document and create its topic profile '''
		f = open(document_string, 'r')
		all_words = []
		#Get all individual words without white space
		for line in f:
			all_words += line.strip('\n').split(' ')
		#Initialize topic distribution for the document with 0 for every topic
		document_topic_distribution = [0 for i in range(0, nr_topics)]
		#For every word
		for word in all_words:
			try:
				#Get the index of the word in the vocab matrix
				word_index = self.vocab[word.lower()]
				#Get the count distribution over topics for the word
				word_probabilities = self.words_topics[word_index]
				#Normalize the counts to get the probability distribution
				norm_list = normalize_array(word_probabilities)
				#Sample a topic from the probability distribution
				sampled_topic = self.sample_multinomial(norm_list)
				#Increase the count for this topic in the count array
				document_topic_distribution[sampled_topic] += 1
			#If the word isn't found, continue to next word
			except KeyError:
				continue
		#Return normalized topic count array i.e. distribution over topics for this document
		normalized_topic_distribution = normalize_array(document_topic_distribution)
		return normalized_topic_distribution

def normalize_array(count_list):
	''' Normalize an array of counts '''
	#Get total sum
	total_sum = sum(count_list)
	normalized_list = []
	#Divide current count by total count and append probability to array
	for count in count_list:
		normalized_list.append(float(count)/total_sum)
	#Return probability distribution
	return normalized_list




if __name__ == "__main__":

	#Command line arguments
	parser = argparse.ArgumentParser(description="Run LDA")
	parser.add_argument('-a', metavar='Specify value for alpha', type=float)
	parser.add_argument('-b', metavar='Specify value for beta', type=float)
	parser.add_argument('-topics', metavar='Specify number of topics.', type=int)
	parser.add_argument('-runs', metavar='Specify number of iterations of gibbs.', type=int)
	parser.add_argument('-toptopics', metavar='Specify number of top words shown for a topic.', type=int)
	parser.add_argument('-topwords', metavar='Specify number of top topics shown for a genre.', type=int)
	parser.add_argument('-f', metavar='Specify filename to write output to', type=str)
	parser.add_argument('-skiplda', help='Provide to skip the LDA and use data from file "inputlda"', action="store_true")
	#metavar='Provide to skip the LDA and use data from file "inputlda"', 
	args = parser.parse_args()

	# TODO: chosing alpha/beta: http://psiexp.ss.uci.edu/research/papers/sciencetopics.pdf
	"""With scientific documents, a large value of beta would lead the
	model to find a relatively small number of topics, perhaps at the
	level of scientific disciplines, whereas smaller values of beta will
	produce more topics that address specific areas of research.
	"""
	nr_topics = 100
	alpha = 0.1
	beta = 0.1
	nr_runs = 1
	top_words = 50
	top_topics = 5
	filename = ''
	skiplda = False

	if(vars(args)['a'] is not None):
		alpha = vars(args)['a']
	if(vars(args)['b'] is not None):
		beta = vars(args)['b']
	if(vars(args)['topics'] is not None):
		nr_topics = vars(args)['topics']
	if(vars(args)['runs'] is not None):
		nr_runs = vars(args)['runs']
	if(vars(args)['toptopics'] is not None):
		top_topics = vars(args)['toptopics']
	if(vars(args)['topwords'] is not None):
		top_words = vars(args)['topwords']
	if(vars(args)['f'] is not None):
		filename = vars(args)['f']
	if(args.skiplda):
		skiplda = vars(args)['skiplda']

	# template for filename
	if filename is '':
		filename = 'output/topics%i_a%.2f_b%.2f_runs%i' %(nr_topics, alpha, beta, nr_runs)

	print "Info:\n- %i runs with: %i topics, alpha: %f, beta: %f\n- number of top words shown for a topic: %i\n- number of top topics shown for a genre: %i\n" %(nr_runs, nr_topics, alpha, beta, top_words, top_topics)

	lda = lda(alpha, beta, nr_topics, load_init=True, skip_lda=skiplda)

	if not skiplda:
		lda.start_lda(nr_runs, top_words, top_topics, filename)

	#lda.start_lda(nr_runs, top_words, top_topics, filename)
	
	#lda.genre_profiles()

	#Test load_new_document function with a new document (example call)
	topic_distribution_krallice = lda.load_new_document('new_docs/krallica_litanyofregrets.txt')

