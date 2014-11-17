import sys
import argparse
import numpy as np
import preprocessing

class lda():

	def __init__(self, alpha, beta, nr_topics):
		"""
		Initialize values as global variables
		alpha: scalar for dirichlet distribution
		beta: scalar for dirichlet distribution
		nr_topics: scalar for number of desired topics
		"""
		self.alpha = alpha
		self.beta = beta
		self.nr_topics = nr_topics
		prep = preprocessing.preprocessing()
		lyrics = prep.get_lyrics()
		print lyrics

	def start_lda(self):
		""" """
		# TODO: just put some functions here
		# Get topic mixture distribution
		theta = self.dirichlet(self.alpha)
		# Pick a topic
		topic = self.sample_multinomial(theta)


	def sample_multinomial(self, distribution):
		"""
		Take multinomial distribution given specific distribution.
		Return index of chosen item
		"""
		return np.random.multinomial(1,distribution).argmax()

	def dirichlet(self, alpha):
		""" Sample from dirichlet distribution given dirichlet parameter
		Returns array of size nr_topics)
		"""
		return np.random.mtrand.dirichlet([alpha] * self.nr_topics)




if __name__ == "__main__":

	#Command line arguments
	parser = argparse.ArgumentParser(description="Run LDA")
	parser.add_argument('-a', metavar='Specify value for alpha', type=float)
	parser.add_argument('-b', metavar='Specify value for beta', type=float)
	parser.add_argument('-topics', metavar='Specify number of topics.', type=int)
	args = parser.parse_args()

	alpha = 0.01
	beta = 0.01
	nr_topics = 10

	if(vars(args)['a'] is not None):
		alpha = vars(args)['a']
	if(vars(args)['b'] is not None):
		beta = vars(args)['b']
	if(vars(args)['topics'] is not None):
		nr_topics = vars(args)['topics']

	lda = lda(alpha, beta, nr_topics)

	
