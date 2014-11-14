import re, urllib
from bs4 import BeautifulSoup
import string
import argparse
import time

artists_information = {}

def get_top_songs(artist_url, n=5):
	''' Get the urls of the top 5 most popular songs of the given artist. '''
	lyrics_url_list = []
	print artist_url
	print "Retrieving top ", n, " songs for ", artist_url.split('/')
	song_soup = BeautifulSoup(urllib.urlopen('http://www.lyricsmode.com' + artist_url).read())
	if(song_soup is not None):
		url_divs = song_soup.findAll('a', attrs={'class', 'ui-song-title'})
		counter = 0
		for link in url_divs:
			if(counter is n):
				break
			url = link.findNext('a', href=True)['href']
			if('http://www.lyricsmode.com' + url not in lyrics_url_list):
				if(artist_url in url):
					lyrics_url_list.append('http://www.lyricsmode.com' + url)
					counter += 1
	return lyrics_url_list

def get_lyrics_document(url):
	''' Parse a lyricsmode document into a text document. Also, retrieve artist genre information. Write everything to a document named artist_song.txt '''
	#Get artist name from url
	song_name = url.split('/')[6].split('.')[0]
	artist_name = url.split('/')[5]
	print "Current artist: ", artist_name
	if(not artists_information.has_key(artist_name)):
		#Get url for artist information from allmusic
		artist_info_url = BeautifulSoup(urllib.urlopen('http://allmusic.com/search/artists/' + artist_name).read()).findAll('div', attrs={'class':'name'})[0].findNext('a')['href']
		#Get the Soup for the artist info
		artist_info = BeautifulSoup(urllib.urlopen(artist_info_url).read())
		artists_information[artist_name] = artist_info
	
	print "Retrieved artist data.."
	artist_info = artists_information[artist_name]
	
	
	#Retrieve genre info
	if(artist_info is not None):
		#Open writable file
		f = open('lyrics_test/'+artist_name+"_"+song_name+'.txt', 'w')
		genre_info_tmp = artist_info.find('div', attrs={'class':'genre'})
		genre_info = 'unknown'
		if(genre_info_tmp is not None):
			genre_info = genre_info_tmp.find('a', href=True).get_text()
		print "Retrieved genre: ", genre_info
		#Retrieve subgenre info
		subgenre_info = artist_info.findAll('div', attrs={'class':'styles'})
		styles_list = []
		#Get html-free text for all subgenres
		for div in subgenre_info:
			styles = div.findAll('a', href=True)
			for style in styles:
				styles_list.append(style.get_text())
		print "Retrieved subgenres: ", str(styles_list)
		print "Writing artist info to file '", artist_name + '_' + song_name + '.txt...'
		f.write('Artist: ' + artist_name)
		f.write('\nTitle: ' + song_name)
		f.write('\nGenre: ' + genre_info)
		f.write('\nSubgenres: ' + str(styles_list))
		print "Getting cleaned lyrics.."
		text = BeautifulSoup(urllib.urlopen(url).read()).find('p', attrs={'id':"lyrics_text"})
		print "Writing lyrics to file ", artist_name + '_' + song_name + '.txt...'
		if(text is not None):
			text = text.get_text()
			f.write('\n\nSong:\n\n' + text.encode("utf-8"))
		#close file
		f.close()
	"\n\nNext song.."

def get_artists(n=99):
	''' Get top n artists for each symbol '''
	alphabet = string.lowercase
	artist_url_list = []
	requests = 0
	for letter in alphabet:
		requests += 1
		if(requests % 10 == 0):
			print "Sleeping to prevent too many requests in too little time..."
			time.sleep(2)
		print "Retrieving ", n, " artists with start symbol ", letter, "..."
		artist_counter = 0
		"Retrieving symbol information..."
		soup = BeautifulSoup(urllib.urlopen('http://www.lyricsmode.com/lyrics/' + letter))
		if(soup is not None):
			"Retrieving artists..."
			rows = soup.find('body').findAll('td')
			for cell in rows:
				if(artist_counter == n):
					break
				a = cell.find('a')
				if(a is not None and a['href'] is not '/'):
					artist_url_list.append(a['href'])
					artist_counter += 1
	artist_counter = 0
	soup = BeautifulSoup(urllib.urlopen('http://www.lyricsmode.com/lyrics/0-9').read())
	if(soup is not None):
		"Retrieving artists..."
		rows = soup.find('body').findAll('td')
		for cell in rows:
			if(artist_counter == n):
				break
			a = cell.find('a')
			if(a is not None and a['href'] is not '/'):
				artist_url_list.append(a['href'])
				artist_counter += 1
	return artist_url_list


if __name__ == "__main__":
	# get_lyrics_document('http://www.lyricsmode.com/lyrics/a/apo_hiking_society/panalangin.html')
	# return 0

	parser = argparse.ArgumentParser(description="Run crawler")
	parser.add_argument('-n', metavar='How many artists should be retrieved per letter?', type=int)
	args = parser.parse_args()

	n = 99
	if(vars(args)['n'] is not None):
		n = vars(args)['n']
	#Retrieve artist urls, can specify a number n to get less than 100 (not more)
	artists = get_artists(n)
	url_list = []
	requests = 0
	#Get urls of song lyrics
	for artist in artists:
		requests += 1
		if(requests % 10 == 0):
			print "Sleeping to prevent too many requests in too little time..."
			time.sleep(1)
		url_list += (get_top_songs(artist))
	#Create lyrics documents
	for url in url_list:
		requests += 1
		if(requests % 10 == 0):
			print "Sleeping to prevent too many requests in too little time..."
			time.sleep(1)
		get_lyrics_document(url)