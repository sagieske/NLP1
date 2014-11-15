import re
import sys
import glob
import os
import pickle
import nltk
from nltk.corpus import stopwords

class feature_extraction:
	
	FOLDER = 'lyrics/'
	DUMPFILE = 'file_dump'
	vocabulary = []

	def __init__(self, dump=True, load=False):
		""" Get all file information. Possible to dump to a specific file or load from a specific file"""
		if load:
			try:
				print "Load information from dumpfile: ", (self.DUMPFILE)
				with open(self.DUMPFILE,'r') as f:
					loaded_files = pickle.load(f)
			except:
				print "File possibly corrupted"
		# Load all files and get song information
		else:
			os.chdir(self.FOLDER)
			loaded_files = []
			for filename in glob.glob("*.txt"):
				(information_song, lyrics) = self.load_file(filename)
				if lyrics is not None:
					info = (information_song, lyrics)
					loaded_files.append(info)
			os.chdir('../')

		# Dump information to file
		if dump:
			print "Dump information to dumpfile: ", self.DUMPFILE
			with open(self.DUMPFILE,'wb') as f:
				pickle.dump(loaded_files, f)

		counter = 0
		lyrics_info_words = []
		all_words = []
		# Clean all lyrics
		for item in loaded_files:
			counter +=1
			cleaned_wordlist = self.clean(item[-1])
			lyrics_info_words.append( item[:-1] + (cleaned_wordlist,) )
			# Add to list of all words found
			if cleaned_wordlist:
				all_words += cleaned_wordlist

		# set vocabulary dictionary
		self.vocabulary = dict.fromkeys(set(all_words),0)
		print self.vocabulary
		self.calculate_word_counts(lyrics_info_words)
		

	def calculate_word_counts(self, lyrics_info_words):
		""" Calculate words """
		# Loop over all lyrics
		for item in lyrics_info_words:
			word_list = item[-1]
			if word_list is not None:
				lyric_count = {}
				for word in word_list:
					# Update vocabulary count
					# TODO! Gives error about unseen word but should not be possible
					# self.vocabulary[word] += 1
					self.vocabulary[word] = lyric_count.get(word, 0) +1
					# Update count in lyric
					lyric_count[word] = lyric_count.get(word, 0) +1
				# Update tuple with information
				item += (lyric_count,)


			

	def clean(self, sentence_array):
		"""
		Cleans lyrics by setting everything to lower case, removing stopwords and further cleaning. 
		Returns cleaned list of words or None (when language is not english)
		"""
		# Split all sentences to words:
		word_list = []
		for sentence in sentence_array:
			words = sentence.split()
			# Use lower case
			lower_case_words = [word.lower() for word in words]
			word_list += lower_case_words

		# Check language of tex
		probable_language = self.check_language(word_list)
		if probable_language != 'english':
			return None
		# Filter out stopwords
		word_list = self.remove_stopwords(probable_language, word_list)
		cleaned_list = []
		for word in word_list:
			# Remove triple (or more) occurance of letter
			word = re.sub(r'(\w)\1+',r'\1\1', word)
			# Remove non words
			word = re.sub(r'(\W)','', word)
			cleaned_list.append(word)
		return cleaned_list


	def check_language(self, word_list):
		""" source: http://blog.alejandronolla.com/2013/05/15/detecting-text-language-with-python-and-nltk/""" 
		languages_ratios = {}
		for language in stopwords.fileids():
			stopwords_set = set(stopwords.words(language))
			words_set = set(word_list)
			# Check similarity
			common_elements = words_set.intersection(stopwords_set)
			# Save as ratio
			languages_ratios[language] = len(common_elements)

		# Get language with most similarities
		most_rated_language = max(languages_ratios, key=languages_ratios.get)
		return most_rated_language

	def remove_stopwords(self, language, word_list):
		""" Remove stopwords for given language from word list"""
		stops = stopwords.words(language)
		return [word for word in word_list if word not in stops]





	def load_file(self, filename):
		"""
		Load lyrics
		"""
		file = open(filename,'r')
		# Read file and strip newline
		datafile = [line.strip() for line in file.readlines()]
		# Empty file
		if not datafile:
			return (filename, None)
		# Get information of song and seperate lyrics
		data_description = datafile[:4]
		lyrics = datafile[7:]
		information_song = self.get_info_title(data_description)
		return (information_song, lyrics)


	def get_info_title(self, info_array):

		""" Get information from title """
		# Substitute _ for spaces and convert everything to spaces
		clean_info_array = [re.sub('_', ' ', str(x)).lower() for x in info_array]
		# Collect information
		artist = re.sub('artist: ', '', clean_info_array[0])
		title = re.sub('title: ', '', clean_info_array[1])
		genre = re.sub('genre: ', '', clean_info_array[2])
		subgenres_str = re.sub("(subgenres: |u'|')", '', clean_info_array[3])
		subgenres_array = re.split(',', subgenres_str[1:-1])
		return (artist, title, genre, subgenres_array)
		


if __name__ == "__main__":
	program = feature_extraction(dump=False, load=True)
