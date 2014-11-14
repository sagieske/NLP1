import re
import sys
import glob
import os



class feature_extraction:
	
	FOLDER = 'lyrics/'

	def __init__(self):
		os.chdir(self.FOLDER)
		for filename in glob.glob("*.txt"):
			(information_song, lyrics) = self.load_file(filename)
			print information_song


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

	#program = feature_extraction('lyrics/2_pac_when_im_gone_remix.txt')
	program = feature_extraction()
