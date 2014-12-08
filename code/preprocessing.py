import re
import sys
import glob
import os
# cPickle is much faster.
try:
    import cPickle as pickle
except:
    import pickle
import nltk
from nltk.corpus import stopwords

class preprocessing:
	"""
	global variables:
	- loaded_files: 	array of dictionaries with keys: artist, title, genre, subgenres(array), original_lyrics (array) 
	- english_lyrics: 	array of dictionaries of all english song-items linked to their loaded_files dictionaries, with addition of cleaned_lyrics (array).
				IMPORTANT: change in a dictionary in loaded_files results in change in dictionary in english_lyrics
	- vocabulary:		dictionary with all encountered words as keys and their counts
	"""
	
	FOLDER = 'lyrics/'
	DUMPFILE = 'file_dump2'
	DUMPFILE_CLEAN = 'lyrics_clean2'
	# TODO: In preprocessing *** are also removed. May be useful for swearing words..?

	def __init__(self, dump_files=True, load_files=False, dump_clean=True, load_clean=False):
		""" Get all file information. Possible to dump to a specific file or load from a specific file"""
		# Load all files
		loaded_files = self.load_all_files(dump=dump_files, load=load_files)
		#self.count_nltk = 0
		#self.count_all = 0
		#self.total_words = 0
		#self.total_left = 0
		# Clean all lyrics
		non_english_index, self.english_lyrics = self.clean_all_files(loaded_files,dump=dump_clean, load=load_clean)
		#print "Total words: %i, total ntlk removed:  %i, total removed (nltk and english.txt): %i. total left: %i" %(self.total_words, self.count_nltk,self.count_all, self.total_left)
		# Create vocabulary
		self.create_vocabulary(self.english_lyrics)
		print self.vocabulary

		#sys.exit()


	def create_vocabulary(self, lyrics):
		""" Create vocabulary from lyrics. Set global variable vocabulary"""
		# Remove possible Nones due to different language of lyrics
		sublist_words = [lyriclist['cleaned_lyrics'] for lyriclist in lyrics]
		all_words = [item for sublist in sublist_words for item in sublist]

		self.vocabulary = dict.fromkeys(set(all_words),(0,0))
		# Count in how many docs word occurs
		for key in self.vocabulary.keys():
			doc_counter = 0
			word_counter = 0
			for sublist in sublist_words:
				if key in sublist:
					doc_counter += 1
				word_counter += sublist.count(key)
			self.vocabulary[key] = (doc_counter, word_counter)

		return self.vocabulary
	
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
				# if genre is unknown, info_dictionary is None. Ignore these files!
				if info_dictionary is None:
					#print "Unknown genre for file: %s" %(filename)
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
	

	def get_vocabulary(self):
		""" Returns vocabulary dictionary"""
		return self.vocabulary

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



	def calculate_word_counts(self, lyrics_info_words):
		""" Calculate words """
		#TODO: is not used??!
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
				all_lyrics_count[key] = lyric_count
		return all_lyrics_count


			

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

		#self.total_words += len(word_list)

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
		last_cleaned_list = self.remove_stopwords(probable_language, cleaned_list, stopwords_list)
		#self.total_left += len(last_cleaned_list)
		return last_cleaned_list


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
		#self.count_nltk += len([word for word in word_list if word in stops])
		#self.count_all += len([word for word in word_list if word in all_stopwords])
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
		# Set lyrics for known genre
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
	program = preprocessing(dump_files=True, load_files=False, dump_clean=True, load_clean=False)
