import re
import sys
import glob
import os
import pickle
import nltk
from nltk.corpus import stopwords

class preprocessing:
	
	FOLDER = 'lyrics/'
	DUMPFILE = 'file_dump'
	DUMPFILE_CLEAN = 'lyrics_clean_dump'
	vocabulary = []
	# TODO: In preprocessing *** are also removed. May be useful for swearing words..?

	def __init__(self, dump=True, load=False, dump_clean=True, load_clean=False):
		""" Get all file information. Possible to dump to a specific file or load from a specific file"""
		# Load all files
		loaded_files = self.load_all_files(dump=False, load=True)
		# Clean all lyrics
		lyrics = self.clean_all_files(loaded_files,dump=False, load=True)

		self.create_vocabulary(lyrics)
		"""
		# set vocabulary dictionary
		self.vocabulary = dict.fromkeys(set(all_words),0)
		print " Calculating word counts"
		all_lyrics_count = self.calculate_word_counts(lyrics_info_words)
		print all_lyrics_count[('1st lady','baby i love you')]
		"""

	def create_vocabulary(self, lyrics):
		""" Create vocabulary from lyrics. Set global variable vocabulary"""
		# Remove possible Nones due to different language of lyrics
		lyrics_no_none = [x for x in lyrics if x is not None]
		all_words = [item for sublist in lyrics_no_none for item in sublist]
		self.vocabulary = dict.fromkeys(set(all_words),0)
	
	def load_all_files(self, dump=True, load=False):
		"""
		Load all lyrics from txt files
		dump/load: booleans to pickle dump or load.
		Returns array where each item is: (information_song, lyrics)
		"""
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
			print "Loading files.."
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
		return loaded_files

	def clean_all_files(self, loaded_files, dump=True, load=False):
		"""
		Cleans all lyrics and calculates
		dump/load:	booleans to load or dump using pickle
		"""
		# Load cleaned text
		if load:
			try:
				print "Load clean lyrics from file: ", (self.DUMPFILE_CLEAN)
				with open(self.DUMPFILE_CLEAN,'r') as f:
					clean_lyrics = pickle.load(f)
			except:
				print "File possibly corrupted"
		else:
			clean_lyrics = []
			# Clean all lyrics
			print "Cleaning lyrics"
			for index in range(0,len(loaded_files)):
				# Clean lyrics
				cleaned_wordlist = self.clean(loaded_files[index][-1])
				clean_lyrics.append(cleaned_wordlist)


		# Dump information to file
		if dump:
			print "Dump cleaned lyrics to file: ", self.DUMPFILE_CLEAN
			with open(self.DUMPFILE_CLEAN,'wb') as f:
				pickle.dump(clean_lyrics, f)

		return clean_lyrics
	


	def get_vocabulary(self):
		""" Returns vocabulary dictionary"""
		return self.vocabulary
		
	def calculate_word_counts(self, lyrics_info_words):
		""" Calculate words """
		all_lyrics_count = {}
		# Loop over all lyrics
		for item in lyrics_info_words:
			word_list = item[-1]
			if word_list is not None:
				lyric_count = {}
				for word in word_list:
					# Update vocabulary count
					self.vocabulary[word] += 1
					# Update count in lyric
					lyric_count[word] = lyric_count.get(word, 0) +1
				# Update tuple with information
				item += (lyric_count,)
				# Save in dictionary under key (artist, title)
				key = (item[0][0], item[0][1])
				print key
				all_lyrics_count[key] = lyric_count
		return all_lyrics_count


			

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
		Load lyrics from file
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
	program = preprocessing(dump=False, load=True)
