import re, urllib
from bs4 import BeautifulSoup

def get_top_songs(artist):
	''' Get the urls of the top 5 most popular songs of the given artist. '''
	lyrics_url_list = []
	url_divs = BeautifulSoup(urllib.urlopen('http://www.lyricsmode.com/lyrics/' + artist[0] + '/' + artist).read()).findAll('div', attrs={'class', 'top-lyrics-number'})
	for link in url_divs:
		lyrics_url_list.append('http://www.lyricsmode.com' + link.findNext('a', href=True)['href'])
	return lyrics_url_list

def get_lyrics_document(url):
	print 'url is: ', url, '\n\n\n'
	artist_name = url.split('/')[5]
	song_name = url.split('/')[6].split('.')[0]
	print artist_name
	print song_name
	text = BeautifulSoup(urllib.urlopen(url).read()).find('p', id="lyrics_text")
	enters = text.findAll('br')
	for enter in enters:
		enter.replace_with('')
	cleaned_text = text.get_text()
	f = open('lyrics/'+artist_name+"_"+song_name+'.txt', 'w')
	f.write(cleaned_text.encode("utf-8"))
	f.close()


if __name__ == "__main__":
	#artist list, pink is written as p!nk which is written as pnk in lyricsmode
	artists = ['metallica', 'rihanna', 'pnk', 'basshunter']
	url_list = []
	for artist in artists:
		url_list += (get_top_songs(artist))
	for url in url_list:
		get_lyrics_document(url)
	
'''
class MABand:
	def __init__(self, name):
		self.soup = BeautifulSoup(urllib.urlopen('http://www.metal-archives.com/bands/' + name).read())
		self.multiple = self.soup.findAll('h1', attrs={'class':'page_title'})
		self.genre = "none"
		self.country ="none"
		self.themes = "none"
		self.location = "none"
		self.status = "none"
		self.name = "none"
		self.logo = "none"
		self.photo = "none"
		self.starting_year = "none"
		self.years_active = "none"

	def get_multiple(self):
		return self.multiple

	def get_genre(self):
		if(self.genre == "none"):
			self.genre = self.soup(text="Genre:")[0].parent.findNext('dd').contents[0]
		return self.genre

	def get_country(self):
		if(self.country == "none"):
			self.country = self.soup(text="Country of origin:")[0].parent.findNext('dd').contents[0].contents[0]
		return self.country

	def get_themes(self):
		if(self.themes == "none"):
			self.themes = self.soup(text="Lyrical themes:")[0].parent.findNext('dd').contents[0]
		return self.themes

	def get_location(self):
		if(self.location == "none"):
			self.location = self.soup(text="Location:")[0].parent.findNext('dd').contents[0]
		return self.location

	def get_status(self):
		if(self.status == "none"):
			self.status = self.soup(text="Status:")[0].parent.findNext('dd').contents[0]
		return self.status

	def get_starting_year(self):
		if(self.starting_year == "none"):
			self.starting_year = self.soup(text="Formed in:")[0].parent.findNext('dd').contents[0]
		return self.starting_year

	def get_years_active(self):
		if(self.years_active == "none"):
			self.years_active = self.soup(text="Years active:")[0].parent.findNext('dd').contents[0]
		return self.years_active

	def get_name(self):
		if(self.name == "none"):
			self.name = self.soup.findAll('h1', attrs={'class':'band_name'})[0].contents[0].contents[0]	
		return self.name

	def get_logo(self):
		if(self.logo == "none"):
			self.logo = self.soup.find('a', attrs={'id':'logo'}).contents[0]['src']
		return self.logo

	def get_photo(self):
		if(self.photo == "none"):
			self.photo = self.soup.find('a', attrs={'id':'photo'}).contents[0]['src']
		return self.photo
'''
	