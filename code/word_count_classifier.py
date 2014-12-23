import sys
import argparse
import numpy as np
import preprocessing
import itertools
import random
import sklearn
import time
from lda import lda
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

nr_topics = 20
alpha = 50/float(nr_topics)
beta = 0.1
nr_runs = 1
top_words = 50
top_topics = 5
lda = lda(alpha, beta, nr_topics)

# Do all 5 folds
for i in range(0, 5):
	# Set fold number
	lda.reset_to_next_fold(i)
	#Get info about documents
	dataset = lda.dataset
	#Get word counts per document
	word_counts = lda.doc_word

	# THESE ARE THE SETS NEEDED FOR THIS FOLD
	print "Building test and training set..."
	training_set = lda.dataset
	training_labels = lda.labels_dataset
	test_set = lda.testset
	test_set_labels = lda.labels_testset
	for i in range(0, len(self.testset)):
		cleaned_lyrics = self.testset[i]['cleaned_lyrics']
		for j in range(0, len(cleaned_lyrics)): 
			word = cleaned_lyrics[j]
			# Update word count in total vocabulary
			wordindex = self.vocab[word]
			self.doc_word_testset[i][wordindex] += 1
	print self.doct_word[i]
	#print "Building test and training set..."
	#number_documents = len(dataset)
	#genre_list = []
	#for doc_index in range(0, number_documents):
	#	genre = dataset[doc_index]['genre']
	#	genre_list.append(genre)

	#training_set = word_counts[:6000]	
	#test_set = word_counts[6000:]
	#training_genre = genre_list[:6000]
	#test_genre = genre_list[6000:]

	print "Training classifier..."
	#classifier = svm.SVC()
	#classifier.fit(training_set, training_genre)
	classifier = svm.SVC(probability=True)
	classifier.fit(training_set, training_labels)


	print "Testing classifier..."
	predicted_genres = classifier.predict_proba(test_set)
	print predicted_genres

	list_genres = list(set(training_labels))
	for genre_label in list_genres:
		metrics = lda.calculate_metrics( test_set_labels, predicted_genres, genre_label)
		print genre_label
		print "metrics: ", metrics

'''
	test_number = len(test_genre)
	correct = 0
	incorrect = 0
	for comparison in range(0, test_number):
		if predicted_genres[comparison] == test_genre[comparison]:
			correct +=1
		else:
			incorrect += 1
	print "Correctly classified: ", correct
	print "Incorrectly classified: ", incorrect
'''
	#Get the wordindex of word love:
	#wordindex = lda.vocab['love']

	#Get the counts of the word love in document 0:
	#docindex = 0
	#lda.doc_word[docindex,wordindex]

	#Get word corresponding to vocabulary index:
	#lda.index_to_vocab[wordindex]
