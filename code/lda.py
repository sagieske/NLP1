#!/usr/bin/env python -W ignore::DeprecationWarning
import sys
import argparse
import numpy as np
import preprocessing
import itertools
import random
import sklearn
import ast
import time
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn import cross_validation
import operator

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import re
import copy

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
	- orig_lda		boolean. If True then also run original lda
	- skip_lda		boolean. If True skips lda
	- fold			scalar to indicate the fold for k-fold cross-validation
	Array
	- genre_count		array of total count of assignments per genre
	- topic_count 		array of total count of word assignments per topic
	- doc_word_count_orig_lda	array of total count of words in document
	- labels_dataset	labels (=genre) for each item in training set
	- labels_testset	labels (=genre) for each item in test set
	Matrixes:
	- doc_word:		count of times word occurs in document
	- words_topics:		count of times word belongs to a topic
	- topics_genres:	count of times topic belongs to a genre (and indirectly the words)
	Dictionary
	- topics		dictionary with tuple (doc, wordposition), also (i,j), as key and value is topic that is assigned
	- topics_orig_lda	USE FOR ORIGINAL LDA: dictionary with tuple (doc, wordposition), also (i,j), as key and value is topic that is assigned
	- vocab 		dictionary with vocabulary words as keys and scalar as index used for the matrices
	- genre_list		dictionary with genres as keys and scalar as index used for the matrices
	- index_to_vocab	dictionary which maps index to word (reverse of vocab dictionary)
	- index_to_genre	dictionary which maps index to genre (reverse of genre_list dictionary)
	"""

	INPUTLDA = 'inputlda'
	#INPUTLDA_ORIG = 'inputlda_orig'
	INIT_DATA = 'init_data'
	#INIT_DATA_ORIG = 'init_data_orig'

	def __init__(self, alpha, beta, nr_topics, skip_lda=False, orig_lda=False):
		""" Initialize
		TODO: load_init is to be used for initialization from pickle load from file. NOT USED YET!
		"""
		
		self.alpha = alpha
		self.beta = beta
		self.nr_topics = nr_topics
		self.skiplda = skip_lda
		self.orig_lda = orig_lda
		self.fold = 0
		# Preprocess data
		prep = preprocessing.preprocessing(dump_files=False, load_files=True, dump_clean=False, load_clean=True)
		# Get lyrics
		self.total_dataset = prep.get_dataset()[:500]
		# Use smaller dataset add [:set]
		print "total nr of lyrics:", len(self.total_dataset)

		labels = []
		#labels_subgenre = []
		label_count = {}
		# Get all labels of dataset
		for item in self.total_dataset:
			labels.append(item['genre'])
			label_count[item['genre']] = label_count.get(item['genre'], 0) +1
			#subgenres = item['subgenres']
			#labels_subgenre.append([i for i in subgenres if i is not 'unknown'])

		
		# Set instance variable to list of set of all labels
		self.all_genres =  list(set(labels))
			

		# Count unknowns:
		#artists_unknown = [item['artist'] for item in self.total_dataset].count('unknown')
		#title_unknown = [item['title'] for item in self.total_dataset].count('unknown')
		#genre_unknown = [item['genre'] for item in self.total_dataset].count('unknown')
		#subgenre_unknown = [item['subgenres'] for item in self.total_dataset].count(['unknown'])
		#print "total unknown: artist: %i, title: %i, genre: %i, subgenre: %i" %(artists_unknown, title_unknown, genre_unknown, subgenre_unknown)

		""" UNCOMMENT TO CREATE NEW FOLD INDICES
		# Get kfold training and test indices (folds: 10 so that it uses 90% for training data)
		# Stratified 10-fold cross-validation
		skf = cross_validation.StratifiedKFold(labels, n_folds=10)
		train_indices_folds = []
		test_indices_folds = []
		for train_index, test_index in skf:
			train_indices_folds.append(train_index)
			test_indices_folds.append(test_index)
		pickle.dump((train_indices_folds, test_indices_folds), open('train_test_indices_stratified',"wb"))
		"""

		# OR LOAD FROM PICKLE FILE:
		self.train_indices_folds, self.test_indices_folds = pickle.load(open('train_test_indices_stratified',"r"))

		# Create the training and test set. Both are set as instance variables
		self.create_train_test_set(0)

		# Initialize counts
		self.genre_count = np.zeros(len(self.all_genres), dtype=int)
		self.topic_count = np.zeros(nr_topics, dtype=int)

		# Counts for original LDA
		if self.orig_lda:
			self.topic_count_orig_lda = np.zeros(nr_topics, dtype=int)
			self.doc_word_count_orig_lda = np.zeros(len(self.dataset), dtype=int)

		# Initialization of matrices and dictionaries 
		self._initialize_lists()
		# Initialize counts for matrices
		# LOAD COUNTS, set to true
		self._initialize_counts(load=True)


	def reset_to_next_fold(self, fold):
		"""
		Create train and test set for fold number. Reset all lists and counts
		"""
		# Set fold variable
		self.fold=fold
		# Reset train and test set for new fold
		self.create_train_test_set(fold)
		# Initialize counts
		self.genre_count = np.zeros(len(self.all_genres), dtype=int)
		self.topic_count = np.zeros(nr_topics, dtype=int)

		# Counts for original LDA
		if self.orig_lda:
			self.topic_count_orig_lda = np.zeros(nr_topics, dtype=int)
			self.doc_word_count_orig_lda = np.zeros(len(self.dataset), dtype=int)

		# Initialization of matrices and dictionaries 
		self._initialize_lists()
		# Initialize counts for matrices
		# LOAD COUNTS, set to true
		self._initialize_counts(load=False)



	def create_train_test_set(self, fold):
		"""
		Create train and test set for fold number. Variables are now set for instance
		Dataset = training set, testset = test set
		"""
		# Reset instance variables
		self.dataset = []
		self.labels_dataset = []
		self.testset = []
		self.labels_testset = []
		# Get indices for training data for this fold
		training_indices = self.train_indices_folds[fold]
		# TODO ONLY USES 500 for now
		# create dataset and training set using indices
		for index in range(0,len(self.total_dataset))[:500]:
			if index in training_indices:
				self.dataset.append(self.total_dataset[index])
				self.labels_dataset.append(self.total_dataset[index]['genre'])
			# if it's not training data, it's test data
			else:
				self.testset.append(self.total_dataset[index])
				self.labels_testset.append(self.total_dataset[index]['genre'])



	def _initialize_lists(self):
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

		# Initialize matrices
		self.words_topics = np.zeros((nr_vocab, self.nr_topics),  dtype=int)
		self.topics_genres = np.zeros((self.nr_topics, nr_genres),  dtype=int)


		# ORIGINAL LDA
		if self.orig_lda:
			# Initialize topic dictionary
			self.topics_orig_lda = {}	
			# Initialize doc and topic occurance
			self.words_topics_orig_lda = np.zeros((nr_vocab, self.nr_topics),  dtype=int)
			self.topics_doc_orig_lda = np.zeros((self.nr_topics, nr_lyrics),  dtype=int)

		# Initialize matrix for occurance words in documents [N x V]
		self.doc_word = np.zeros((nr_lyrics,len(all_words)),  dtype=int)


	def _initialize_counts(self, load=False):
		"""
		Initialize the counts of all words in documents
		Loop through all documents and get its genre.
		Loop through all words in document. Then choose random topic to initialize values in matrices
		"""

		# Loads are dependent on whether you also use original lda. self.fold indicates in which fold this occurs
		if self.skiplda:
			if self.orig_lda:
				self.load_data(self.INPUTLDA)
			else:
				self.load_data(self.INPUTLDA)
		elif load:
			if self.orig_lda:
				self.load_data(self.INIT_DATA+str(self.fold))
			else:
				self.load_data(self.INIT_DATA+str(self.fold))
		else:

			print "Initialize counts.."
			# Get sizes
			nr_lyrics = len(self.dataset)
			nr_genres = len(self.all_genres)

			
			# Initialize all counts
			# Loop over documents:
			for i in range(0, nr_lyrics):
				# Get index of document and associated genre
				genre_index = self.genre_list[self.labels_dataset[i]]
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
					# Set topic of ij to k
					self.topics[(i,j)] = k
				
					# Update count
					self.topic_count[k] += 1

					# Update matrices
					self.words_topics[wordindex][k] +=1
					self.topics_genres[k][genre_index] += 1

					# Initialization needed or original LDA
					if self.orig_lda:
						self.topics_orig_lda[(i,j)] = k		
						self.words_topics_orig_lda[wordindex][k] +=1
						self.topics_doc_orig_lda[k][i] += 1
						self.topic_count_orig_lda[k] += 1

				# Set nr of words for doc needed or original LDA
				if self.orig_lda:
					self.doc_word_count_orig_lda[i] = len(cleaned_lyrics)

			self.dump_data(self.INIT_DATA+str(self.fold))




	def start_gibbs(self, N, topwords, toptopics, filename):
		"""
		Runs Gibbs sampling for LDA on N iterations. Topwords and toptopics are needed for number of words/topics needed
		for representation of topic/genre. Filename is file to which output (representation) is printed
		"""

		print "start Gibbs sampling! Also compute original LDA:", self.orig_lda
		start = time.time()

		nr_lyrics = len(self.dataset)
		# Do gibbs sampling N times for all items
		for iteration in range(0,N):
			# Loop through all documents
			for i in range(0, nr_lyrics):
				# get genre (and corresponding index)
				genre_index = self.genre_list[self.labels_dataset[i]]
				cleaned_lyrics = self.dataset[i]['cleaned_lyrics']
				# Loop through all words
				for j in range(0, len(cleaned_lyrics)): 
					# Get word (and corresponding index)
					word = cleaned_lyrics[j]
					word_index = self.vocab[word]
					position = (i,j)

					# Get current topic to which this word is assigned
					current_topic = self.topics[position]

					# Get topic probability distribution
					p_zij = self.probability_topic(current_topic,word_index, genre_index)
					# Get topic index using topic distribution
					k = self.sample_multinomial(p_zij)
					# update matrices
					self.update(current_topic, position, word_index, genre_index, k)

					# ORIGINAL LDA:
					if self.orig_lda:
						# Get current topic to which this word is assigned
						current_topic_orig_lda = self.topics_orig_lda[position]
						# Get topic probability distribution
						p_zij_orig_lda = self.probability_topic_orig_lda(current_topic_orig_lda,word_index, i)
						# Get topic index using topic distribution
						k_orig_lda = self.sample_multinomial(p_zij_orig_lda)
						self.update_orig_lda(current_topic_orig_lda, position, word_index, k_orig_lda)

				if i % 500 == 0 and i != 0:
					print "- lyrics done: %i" %(i)

			print "done iteration %i (stopwatch: %s)" %(iteration, str(time.time()-start))
			# TODO, LDA??

			if iteration % 5 == 0 and iteration > 5:
				filename = "iter" + str(iteration) + "_a" + str(self.alpha) + "_b" + str(self.beta) + "_topics" \
					+ str(self.nr_topics) + "_fold" + str(self.fold) 
				self.dump_data(filename)
				self.print_to_file(N, topwords, toptopics, filename, iteration)
				self.print_to_file_lda(N, topwords, toptopics, filename, iteration)
		

		# prints initialization
		if N == 0:
			iteration = N
			#self.print_to_file(N, topwords, toptopics, filename, iteration)

		filename = "iter" + str(iteration) + "_a" + str(self.alpha) + "_b" + str(self.beta) + "_topics" \
				+ str(self.nr_topics) + "_fold" + str(self.fold) 
		self.dump_data(filename)

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


	def update_orig_lda(self, previous_topic_index, position, word_index, topic_index):
		""" 
		USED FOR ORIGINAL LDA
		Update values in matrices using indices. 
		Get previous topic that was assigned to word at that position.
		Subtract 1 from matrices
		"""
		doc_index = position[0]
		# Subtract in matrices the old topic index
		self.words_topics_orig_lda[word_index][previous_topic_index] -= 1
		self.topic_count_orig_lda[previous_topic_index] -= 1
		self.topics_doc_orig_lda[previous_topic_index][doc_index] -= 1

		# Add in matrices the new topic index
		self.words_topics_orig_lda[word_index][topic_index] +=1
		self.topic_count_orig_lda[topic_index] += 1
		self.topics_doc_orig_lda[topic_index][doc_index] += 1

		# Update topic assignment
		self.topics_orig_lda[position] = topic_index
		#print "Updated word %i from topic %i to topic %i" %(word_index, previous_topic_index, topic_index)



	def probability_topic(self, current_topic, word_index, genre_index):
		"""
		Calculate probabilities of topics for word_ij and return array (sums to 1)
		Used for extended LDA using genres
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



	def probability_topic_orig_lda(self, current_topic, word_index, doc_index):
		"""
		Calculate probabilities of topics for word_ij and return array (sums to 1)
		Used for the original LDA
		"""
		# For each topic k:
			# ((beta + count_words_topic) / (aantal woorden * beta + topic count)) 
			#  * ((alpha + count_topic_doc) / (aantal topics * alpha + doc_word_count))

		p_zij = np.zeros(self.nr_topics)

		# lda_matriced=True indicates to use lda_matrices in this function
		for i in range(0, self.nr_topics):
			a = self.beta + self.count_words_topic(current_topic, word_index, i, lda_matrices=True)
			b = len(self.vocab) * self.beta + self.count_topic(current_topic, i, lda_matrices=True)
			c = self.alpha + self.count_topic_doc(current_topic, doc_index, i)
			d = self.nr_topics * self.alpha + (self.doc_word_count_orig_lda[doc_index] - 1)	# nr of words in doc - current word. is not dependent on topic

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
		""" Dump data to corresponding file """
		to_dump = {}
		to_dump['topics'] = self.topics
		to_dump['alpha'] = self.alpha
		to_dump['beta'] = self.beta
		to_dump['nr_topics'] = self.nr_topics
		to_dump['words_topics'] = self.words_topics
		to_dump['topics_genres'] = self.topics_genres
		print "Dump information to dumpfile: ", filename
		pickle.dump(to_dump, open(filename,"wb"))

		# Original LDA implemented, extra addition to dump
		if self.orig_lda:
			to_dump_orig_lda = {}
			to_dump_orig_lda['topics_org_lda'] = self.topics_orig_lda
			to_dump_orig_lda['words_topics_orig_lda'] = self.words_topics_orig_lda
			to_dump_orig_lda['topics_doc'] = self.topics_doc_orig_lda
			orig_lda_filename = filename+"_orig"
			print "Dump LDA original information to dumpfile: ", orig_lda_filename
			pickle.dump(to_dump_orig_lda, open(orig_lda_filename ,"wb"))



	def load_data(self, filename):
		""""
		Loads instance variables from file. 
		If except is caught, do initialize_counts again
		"""
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
		except MemoryError:
			print "Memory Error for file %s." %(filename)
			if self.skiplda:
				sys.exit()
			self._initialize_counts(False)
		# Print out all other exceptions than Memory error. 
		except Exception,e: 
			print "Error: %s for file %s." %(str(e), filename)
			if self.skiplda:
				sys.exit()
			self._initialize_counts(False)

		# Load data for original LDA 
		if self.orig_lda:
			try:
				filename_orig_lda = filename+"_orig"
				print "Load original LDA data from file: ", (filename_orig_lda)
				with open(filename_orig_lda,'r') as f:
					dumped_orig_lda = pickle.load(f)
				self.topics_orig_lda = dumped_orig_lda['topics_org_lda']
				self.words_topics_orig_lda = dumped_orig_lda['words_topics_orig_lda']
				self.topics_doc_orig_lda = dumped_orig_lda['topics_doc'] 
			except MemoryError:
				print "Memory Error for file in orig%s." %(filename)
				if self.skiplda:
					sys.exit()




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
			genres.append(self.labels_dataset[i])
		return genres, dt_dist

	def document_topic_distribution_orig_lda(self):
		""" Create array of topic distribution per document.
		Returns matrix of topic distribution per document and array of the genre corresponding to each document
		"""
		nr_lyrics = len(self.dataset)
		# Initialize matrix. Every lyric has array of len topics
		dt_dist = np.zeros((nr_lyrics, self.nr_topics))
		genres = []
		for i in range(0, nr_lyrics):
			# Get all topic assignments for this document
			counts = self.topics_doc_orig_lda[:,i]
			# Normalize
			total = np.sum(dt_dist[i], axis=0)
			dt_dist[i] = np.divide(dt_dist[i], total)
			# Add current genre to list
			genres.append(self.labels_dataset[i])
		return genres, dt_dist


	def dirichlet(self, alpha):
		""" Sample from dirichlet distribution given dirichlet parameter
		Returns array of size nr_topics
		"""
		return np.random.mtrand.dirichlet([alpha] * self.nr_topics)

	def count_words_topic(self, current_topic, wordindex, topic, lda_matrices=False):
		""" Count the number of times this similar word(wordindex) is associated with topic, excluding the word at given position
		NOTE: lda_matrices indicates whether to use matrices used for original lda!"""
		# If word has same topic, remove 1 from count
		if current_topic == topic:
			if lda_matrices:
				return self.words_topics_orig_lda[wordindex, topic] - 1
			else:
				return self.words_topics[wordindex, topic] - 1
		else:
			if lda_matrices:
				return self.words_topics_orig_lda[wordindex, topic] 	
			else:
				return self.words_topics[wordindex, topic] 	

	def count_topic(self, current_topic, topic, lda_matrices=False):
		""" Count the total number of words associated with topic, excluding word at given position
		NOTE: lda_matrices indicates whether to use matrices used for original lda!"""
		# Excluded word is associated with this topic, so subtract 1
		if current_topic == topic:
			if lda_matrices:
				return self.topic_count_orig_lda[topic] - 1
			else:
				return self.topic_count[topic] - 1
		# Current word is not associated with this topic
		else:
			if lda_matrices:
				return self.topic_count_orig_lda[topic]
			else:
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

	def count_topic_doc(self, current_topic, doc_index, topic):
		""" Count the number of times this specific topic is associated with the genre, excluding the word at given position
		Used for original LDA """
		# Topic of excluding word is associated with this topic, so subtract 1
		if current_topic == topic:
			return self.topics_doc_orig_lda[topic, doc_index] -1 
		# Topic of excluding word is not associated with this topic
		else:
			return self.topics_doc_orig_lda[topic, doc_index] 



	def get_top_words_topic(self, topic_index, nr_top, lda_matrices=False):
		"""
		Get # top words associated with topic. 
		NOTE: lda_matrices indicates whether to use matrices used for original lda!
		"""
		# Get vector of the counts per word associated with this topic
		if lda_matrices:
			vector_words = self.words_topics_orig_lda[:, topic_index]
		else:
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
				genre_list = genre_list_total[i]
				# Print every topic
				for j in range(0,len(genre_list)):
					f.write('--> Topic %i\n' %(indices_max_topics_total[i][j]))

					nr_words_topic = self.count_topic(0, indices_max_topics_total[i][j])
					to_print = "\t"
					for h in range(0, len(words_topic_total[indices_max_topics_total[i][j]])):
						curword = words_topic_total[indices_max_topics_total[i][j]][h]
						nr_word_in_topic = self.count_words_topic(0, self.vocab[curword], indices_max_topics_total[i][j])
						prob = nr_word_in_topic/float(nr_words_topic)
						to_print += "'%s' (prob: %0.5f), " %(curword, prob)

					f.write('%s\n' %(to_print))


					f.write('%s\n' %(str(words_topic_total[indices_max_topics_total[i][j]])))

	def print_to_file_lda(self, runs, top_words,top_topics,  filename, iteration):
		"""
		Write top words per topic and top topics per genre to file
		"""
		# Get top words per topic
		words_topic_total = []
		for i in range(0,self.nr_topics):
			max_indices = self.get_top_words_topic(i, top_words, lda_matrices=True)
			words_topic = self.get_from_indices(max_indices, 'words')
			words_topic_total.append(words_topic)

		# Get top topics per genre:
		genre_list_total = []
		indices_max_topics_total  = []
		# Get all lyrics corresponding to genre:
		for genre in self.all_genres:
			# get all indices that belong to this genre
			indices = [i for i, x in enumerate(self.labels_dataset) if x == genre] 
			count_per_genre = np.zeros(nr_topics, dtype=int)
			# Count number of times topic is added to document that belongs to this genre
			for i in indices:
				count_per_genre += self.topics_doc_orig_lda[:,i]
			# Get indices of topics with highest counts
			indices_max_topics = count_per_genre.argsort()[-nr_topics:][::-1][:top_topics]
			genre_list = []
			# For every topic get the words
			for i in range(0, top_topics):
				topic_index = indices_max_topics[i]
				genre_list.append(words_topic_total[topic_index])
			genre_list_total.append(genre_list)
			indices_max_topics_total.append(indices_max_topics)
	
		total_filename_topics = filename + "_" + str(iteration) + "_topics_origlda.txt"
		# Write to file for topics (if not exists open!)
		print "write to file: %s" %(filename)
		with open(total_filename_topics, 'w+') as f:
			f.write('Runs: %i, alpha: %.2f, beta: %.2f, nr topics: %i, nr genres: %i, iteration: %i, top_topics: %i, top_words: %i\n\n' \
				%(runs, self.alpha, self.beta, self.nr_topics, len(self.all_genres), iteration, top_topics, top_words) )
			f.write('TOPIC-WORD DISTRIBUTION at iteration %i/%i \n' %(iteration, runs))
			# Print list for every topic
			for i in range(0,len(words_topic_total)):
        			f.write('Topic %i\n%s\n' %(i, str(words_topic_total[i])))


		total_filename_genre = filename + "_" + str(iteration) + "_genre_origlda.txt"
		# Write to file for genres (if not exists open!)
		with open(total_filename_genre, 'w+') as f:
			f.write('Runs: %i, alpha: %.2f, beta: %.2f, nr topics: %i, nr genres: %i, iteration: %i, top_topics: %i, top_words: %i\n\n' \
				%(runs, self.alpha, self.beta, self.nr_topics, len(self.all_genres), iteration, top_topics, top_words) )
			f.write('GENRE-TOPIC DISTRIBUTION at iteration %i/%i \n' %(iteration, runs))
			# Print every genre
			for i in range(0,len(genre_list_total)):
        			f.write('\nGENRE %i (%s)\t (topics: %s)\n' %(i, self.all_genres[i], str(indices_max_topics_total[i]) ) )
				genre_list = genre_list_total[i]
				# Print every topic
				for j in range(0,len(genre_list)):
					f.write('--> Topic %i\n' %(indices_max_topics_total[i][j]))


					nr_words_topic = self.count_topic(0, indices_max_topics_total[i][j])
					to_print = "\t"
					# Calculate probabilities for every word
					for h in range(0, len(words_topic_total[indices_max_topics_total[i][j]])):
						curword = words_topic_total[indices_max_topics_total[i][j]][h]
						nr_word_in_topic = self.count_words_topic(0, self.vocab[curword], indices_max_topics_total[i][j], lda_matrices=True)
						prob = nr_word_in_topic/float(nr_words_topic)
						to_print += "'%s' (prob: %0.5f), " %(curword, prob)

					# Write top words with probabilities and in seperate list
					f.write('%s\n' %(to_print))
					f.write('%s\n' %(str(words_topic_total[indices_max_topics_total[i][j]])))


	def genre_profiles(self, orig_lda=False):
		"""
		Create profiles for genres by using the topic distributions found
		"""

		# Get the topic distribution for every document and the corresponding genre
		if orig_lda:
			genres, distr = lda.document_topic_distribution_orig_lda()
			extension_filename = "_orig.png"
		else:
			genres, distr = lda.document_topic_distribution()
			extension_filename = ".png"
		genre_indices = []
		# Loop over all possible genres
		for genre in self.all_genres:

			print genre
			# Get indices of documents that belong to this genre
			indices = [i for i, x in enumerate(genres) if x == genre]
			# If no genres are found skip!
			if not indices:
				print "nothing found for genre: %s" %(genre)
				continue
			genre_indices.append(indices)
			# Get topic distributions for all documents belonging to this genre
			genre_matrix =  np.array([x for i, x in enumerate(distr) if i in indices])

			matplotlib.rcParams.update({'font.size': 6})

			# Get mean per topic
			mean_genre =np.mean(genre_matrix*100, axis=0)
			stdev_genre = np.std(genre_matrix*100, axis=0)
			#print stdev_genre

			# Plot bar chart? Not really nice
			fig = plt.figure(figsize=(8, 6))
			ax = fig.add_subplot(111)
			ax.set_title('Genre: %s' %genre)
			ind = np.arange(len( mean_genre))
			width = 0.35
			ax.bar(ind, mean_genre, width,  align='center', yerr=stdev_genre, ecolor='k')
			ax.set_ylabel('Mean in percentage')
			ax.set_xticks(ind)
			ax.set_xlabel('Topic number')
			genre = re.sub('/', '-', genre)

			plt.savefig("%s" %(genre+extension_filename))
			plt.close('all')


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

	def get_new_document_dist(self, document):
		''' load new document and create its topic profile '''
		#Initialize topic distribution for the document with 0 for every topic
		document_topic_distribution = [0 for i in range(0, nr_topics)]
		#For every word
		for word in document:
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

	def classify(self):
		print "Gathering training set information..."
		train_genre_list, distribution_train_matrix = self.document_topic_distribution()

		test_genre_list = []
		number_testing = len(self.testset)
		distribution_test_matrix = np.zeros((number_testing, self.nr_topics))

		for doc_index in range(0, number_testing):
			genre = self.testset[doc_index]['genre']
			test_genre_list.append(genre)
			distribution_test_matrix[doc_index] = self.get_new_document_dist(self.testset[doc_index]['cleaned_lyrics'])

		print "Training classifier..."
		classifier = svm.SVC(probability=False, kernel='rbf', C=1.0, gamma=0.75)
		classifier.fit(distribution_train_matrix, train_genre_list)


		print "Testing classifier..."
		predicted_genres = classifier.predict(distribution_test_matrix)
		actual_predictions = classifier.predict(distribution_test_matrix)
		print "Predicted genres: ", predicted_genres
		right = 0
		wrong = 0
		for test_point in range(0, number_testing):
			if(actual_predictions[test_point] == test_genre_list[test_point]):
				right +=1 
			else:
				wrong +=1
		print "Correct: ", right
		print "Incorrect: ", wrong
		score = classifier.score(distribution_test_matrix, test_genre_list)
		print "Score: ", score

	def generate_song(self, length, genre):
		genres, distr = lda.document_topic_distribution()
		genre_indices = []
		print distr
		# Loop over all possible genres
		print genre
		# Get indices of documents that belong to this genre
		for index in range(0,len(distr)):
			print distr[index]
			if genres[index] == genre:
				print "YES"
				genre_indices.append(index)
		print genre_indices
		# Get topic distributions for all documents belonging to this genre
		genre_matrix =  np.array([x for i, x in enumerate(distr) if i in genre_indices])

		print genre_matrix
		for document in genre_matrix:
			normalized = normalize_array(document)
			sampled_topic = self.sample_multinomial(normalized)
			word_list = words_topic[sampled_topic]
			print "words: ", word_list






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
	parser.add_argument('-origlda', help='Provide to also compute original the LDA', action="store_true")
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
	origlda = False

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
	if(args.origlda):
		origlda = vars(args)['origlda']

	# template for filename
	if filename is '':
		filename = 'output/topics%i_a%.2f_b%.2f_runs%i' %(nr_topics, alpha, beta, nr_runs)

	print "Info:\n- %i runs with: %i topics, alpha: %f, beta: %f\n- number of top words shown for a topic: %i\n- number of top topics shown for a genre: %i\n" %(nr_runs, nr_topics, alpha, beta, top_words, top_topics)

	lda = lda(alpha, beta, nr_topics, skip_lda=skiplda, orig_lda=origlda)
	kfold = False
	folds = 10

	if not skiplda:
		lda.start_gibbs(nr_runs, top_words, top_topics, filename)
		if kfold:
			for i in range(1,10):
				print "fold %i" %(i)
				lda.reset_to_next_fold(i)
				lda.start_gibbs(nr_runs, top_words, top_topics, filename)


	##lda.start_gibbs(nr_runs, top_words, top_topics, filename)
	lda.genre_profiles(orig_lda=False)
	lda.genre_profiles(orig_lda=True)

	#Test load_new_document function with a new document (example call)
	#topic_distribution_krallice = lda.load_new_document('new_docs/krallica_litanyofregrets.txt')

	lda.classify()
	#lda.generate_song('rap', 20)

