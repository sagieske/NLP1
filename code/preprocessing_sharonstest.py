import re
import sys
import glob
import os
import pickle
import nltk
from nltk.corpus import stopwords
import numpy as np

class preprocessing:
	"""
	global variables:
	- loaded_files: 	array of dictionaries with keys: artist, title, genre, subgenres(array), original_lyrics (array) 
	- english_lyrics: 	array of dictionaries of all english song-items linked to their loaded_files dictionaries, with addition of cleaned_lyrics (array).
						IMPORTANT: change in a dictionary in loaded_files results in change in dictionary in english_lyrics
	- doc_word:			count of times word occurs in document
	- vocab 			dictionary with vocabulary words as keys and scalar as index used for the matrices
	"""
	
	FOLDER = 'lyrics/'
	DUMPFILE = 'file_dump2'
	DUMPFILE_CLEAN = 'lyrics_clean2'
	DUMPFILE_VOCAB = 'vocab_count'

	# TODO: In preprocessing *** are also removed. May be useful for swearing words..?

	def __init__(self, dump_files=True, load_files=False, dump_clean=True, load_clean=False, dump_vocab=True, load_vocab=False):
		""" Get all file information. Possible to dump to a specific file or load from a specific file"""
		# Load all files
		loaded_files = self.load_all_files(dump=dump_files, load=load_files)

		# Clean all lyrics
		non_english_index, self.english_lyrics = self.clean_all_files(loaded_files,dump=dump_clean, load=load_clean)

		# Create vocabulary
		dump_vocab = True
		load_vocab = False
		min_doc = 2
		self.create_vocabmatrix(self.english_lyrics, min_doc, dump_vocab=dump_vocab, load_vocab=load_vocab)

		sys.exit()


	def create_vocabmatrix(self, lyrics, min_doc, dump_vocab=True, load_vocab=False):
		""" Create vocabulary from lyrics. Set global variable vocabulary"""

		if load_vocab:
			try:
				print "Load vocabulary (and lyrics adjusted to this vocab) from file: ", (self.DUMPFILE_VOCAB)
				with open(self.DUMPFILE_VOCAB,'r') as f:
					self.doc_word, self.vocab, self.english_lyrics = pickle.load(f)
			except:
				print "File possibly corrupted"

		else:
			# Remove possible Nones due to different language of lyrics
			sublist_words = [lyriclist['cleaned_lyrics'] for lyriclist in lyrics]
			all_words = list(set([item for sublist in sublist_words for item in sublist]))

			nr_lyrics = len(lyriclist)

			# Get word count array per document: self.doc_word[doc, :]
			self.doc_word = np.zeros((nr_lyrics,len(all_words)),  dtype=int)


			# Get counts for words in documents
			for i in range(0, nr_lyrics):
				# Get cleaned lyrics of item
				cleaned_lyrics = self.english_lyrics[i]['cleaned_lyrics']
				# Loop over words in doc:
				for j in range(0, len(cleaned_lyrics)): 
					# Get word
					word = cleaned_lyrics[j]
					# Get index
					wordindex = all_words.index(word)
					# Update in count of word in document
					self.doc_word[i][wordindex] += 1

			# Count nonzeros per word
			nonzeros = (self.doc_word != 0).sum(0)

			print len(all_words)
			# Get indices of words that occur in less than min_doc documents
			remove_indices = [i for i, count in enumerate(nonzeros) if count < min_doc]
			print len(remove_indices)
			sys.exit()
			delete_words = [word for i, word in enumerate(all_words) if i in remove_indices]
			# Remove words from vocabulary array 
			new_all_words = [word for word in all_words if word not in delete_words]
			print "removed words that occur in less than %i docs: %i. Total remaining: %i" %(min_doc,len(remove_indices), len(new_all_words))

			# Remove words that occur in less than 10 documents
			self.doc_word = np.array(filter(lambda row: np.count_nonzero(row) >= min_doc, self.doc_word.T)).T

			# Clean up lyrics by removing words that occur in less than 10 documents
			for i in range(0, nr_lyrics):
				# Get cleaned lyrics of item
				cleaned_lyrics = self.english_lyrics[i]['cleaned_lyrics']
				new_cleaned_lyrics = [word for word in cleaned_lyrics if word not in delete_words]
				self.english_lyrics[i]['cleaned_lyrics'] = new_cleaned_lyrics

			# Create indices for words
			self.vocab =  dict(enumerate(new_all_words))


		# Dump information to file
		if dump_vocab:
			print "Dump vocabulary (and lyrics adjusted to this vocab) to dumpfile: ", self.DUMPFILE_VOCAB
			with open(self.DUMPFILE_VOCAB,'w+') as f:
				pickle.dump((self.doc_word, self.vocab, self.english_lyrics), f)


	
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
			print "Importing from files.."
			unknown_genres = 0
			counter = 0
			for filename in glob.glob("*.txt"):
				info_dictionary = self.load_file(filename)
				# if genre is unknown, info_dictionary is None
				if info_dictionary is None:
					print "Unknown genre for file: %s" %(filename)
					unknown_genres +=1
					continue
				else:
					# Lyrics are found
					try:
						if info_dictionary['original_lyrics'] is not None:
							loaded_files.append(info_dictionary)
						else:
							print "No lyrics found in file: %s" %(filename)
					except:
						print "Error found in file: %s" %(filename)
					counter += 1
			os.chdir('../')
			print "total unknown genres: %i" %(unknown_genres)

		# Dump information to file
		if dump:
			print "Dump information to dumpfile: ", self.DUMPFILE
			with open(self.DUMPFILE,'w+') as f:
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
					non_english, clean_lyrics = pickle.load(f)
			except:
				print "File possibly corrupted"
		else:
			clean_lyrics = []
			non_english = []

			# Load extra stopwords fie
			stopwords_file = [word for line in open('english.txt', 'r') for word in line.split()]

			# Clean all lyrics
			print "Cleaning lyrics"
			for index in range(0,len(loaded_files)):
				# Clean lyrics
				cleaned_wordlist = self.clean(loaded_files[index]['original_lyrics'], stopwords_file)
				# lyric is not english
				if cleaned_wordlist == 0:
					non_english.append(index)
				else:
					# Add cleaned lyrics to dictionary
					loaded_files[index]['cleaned_lyrics'] = cleaned_wordlist
					clean_lyrics.append(loaded_files[index])


		# Dump information to file
		if dump:
			print "Dump cleaned lyrics to file: ", self.DUMPFILE_CLEAN
			with open(self.DUMPFILE_CLEAN,'w+') as f:
				pickle.dump((non_english, clean_lyrics), f)

		return non_english, clean_lyrics
	

	def get_doc_word(self):
		""" Returns matrix of word-document occurance and word to indices dictionary that is used"""
		return self.doc_word, self.vocab

	def get_dataset(self):
		""" Returns dataset of english lyrics of all songs and their information"""
		return self.english_lyrics
		
	def get_information_dictionary(self, key, value):
		""""
		Return dictionary with keys of specified key(string) and as array of value(string)
		for example: get_information_dictionary('artist', 'title')
		"""
		# Initialize dictionary
		information_dictionary = {}
		
		# loop over all english lyrics
		for item in self.english_lyrics:
			artist_name = item[key]
			title = item[value]
			# Append to list of titles (initialize with [] if artist not yet in dict)
 			information_dictionary.setdefault(artist_name, []).append(title)
		return information_dictionary



			

	def clean(self, sentence_array, stopwords_list):
		"""
		Cleans lyrics by setting everything to lower case, removing stopwords and further cleaning. 
		Returns cleaned list of words or 0 (when language is not english)
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
			return 0
		# Filter out stopwords
		word_list = self.remove_stopwords(probable_language, word_list, stopwords_list)
		cleaned_list = []
		for word in word_list:
			# Remove triple (or more) occurance of letter
			word = re.sub(r'(\w)\1+',r'\1\1', word)
			# Remove non words
			word = re.sub(r'(\W)','', word)
			cleaned_list.append(word)
		# Remove possible empty words
		cleaned_list= filter(None, cleaned_list)

		# Again try to remove stopwords that are now possibly found after non-words are removed
		# example: (the) in lyrics will not be removed and is still in words_list, () are removed in cleaned_list 
		# and then again are removed as stopwords
		cleaned_list2 = self.remove_stopwords(probable_language, cleaned_list, stopwords_list)

		return cleaned_list2


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

	def remove_stopwords(self, language, word_list, stopwords_list):
		""" Remove stopwords for given language from word list"""
		stops = stopwords.words(language)
		all_stopwords = set(stops + stopwords_list)
		return [word for word in word_list if word not in all_stopwords]


	def load_file(self, filename):
		"""
		Load lyrics from file, 
		Return as dictionary with keys artist, title, genre, subgenres and original_lyrics
		"""
		file = open(filename,'r')
		# Read file and strip newline
		datafile = [line.strip() for line in file.readlines()]
		# Empty file
		if not datafile:
			return (filename, None)
		# Get information of song and seperate lyrics
		data_description = datafile[:4]
		information_song = self.get_info_title(data_description)
		# Delete unknown genres
		if information_song['genre'] == 'unknown':
			return None
		lyrics = datafile[7:]
		information_song['original_lyrics'] = lyrics
		return information_song


	def get_info_title(self, info_array):
		""" Get information from title """
		# Substitute _ for spaces and convert everything to spaces
		clean_info_array = [re.sub('_', ' ', str(x)).lower() for x in info_array]
		# Collect information by regex search
		match_artist = re.search(r"artist:\s(.*)", clean_info_array[0])
		match_title = re.search(r"title:\s(.*)", clean_info_array[1])
		match_genre = re.search(r"genre:\s(.*)", clean_info_array[2])
		match_subgenre = re.search(r"subgenres:\s(.*)", clean_info_array[3])

		# Set to unknown if not found
		if match_artist:
			artist =  match_artist.group(1)
		else:
			artist = 'unknown'
		if match_title:
			title = match_title.group(1)
		else:
			title = 'unknown'
		if match_genre:
			genre = match_genre.group(1)
		else:
			genre = 'unknown'
		if match_subgenre:
			subgenres_str = re.sub("(subgenres: |u'|')", '', match_subgenre.group(1))
			subgenres_array = re.split(',', subgenres_str[1:-1])
			# '' means unknown
			if len(subgenres_array) == 1 and subgenres_array[0] == '':
				subgenres_array = ['unknown']
			
		else:
			subgenres_array = ['unknown']

		info = {'artist': artist, 'title': title, 'genre': genre, 'subgenres': subgenres_array}
		return info
		


if __name__ == "__main__":
	program = preprocessing(dump_files=False, load_files=True, dump_clean=False, load_clean=True)
