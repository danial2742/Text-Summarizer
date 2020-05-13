from  bs4 import BeautifulSoup
import requests
import re
import nltk
import heapq

class Summarize(object):
	"""docstring for Summarize"""
	def __init__(self, url):
		super(Summarize, self).__init__()
		self.url = url

	def pre_proccess(self):
		# download nltk data
		nltk.download('stopwords')
		nltk.download('punkt')
		# get webpage html source
		source = requests.get(self.url)
		soup = BeautifulSoup(source.text, 'lxml')
		# extract paragraphs from text
		paragraphs = [paragraph.text for paragraph in soup.find_all('p')]
		# join paragraphs and make text
		text = '\n'.join(paragraphs).strip()
		# for remove paragraph extra description popup
		text = re.sub(r'\[\d*\]', ' ', text)
		# for remove End Of Lines
		text = re.sub(r'\s+', ' ', text)
		# clean text for tokenize
		clean_text = text.lower()
		# the string DOES NOT contain any word characters
		clean_text = re.sub(r'\W', ' ', clean_text)
		# remove any digits from text
		clean_text = re.sub(r'\d', ' ', clean_text)
		# for remove End Of Lines
		clean_text = re.sub(r'\s+', ' ', clean_text)
		# return text and clean_text
		return text, clean_text

	def run(self):
		text, clean_text = self.pre_proccess()
		sentences = nltk.sent_tokenize(text)
		stop_words = nltk.corpus.stopwords.words('english')

		word2count = {}
		for word in nltk.word_tokenize(clean_text):
		    if word not in stop_words:
		        if word not in word2count.keys():
		            word2count[word] = 1
		        else:
		            word2count[word] += 1
		# Converting counts to weights
		for key in word2count.keys():
		    word2count[key] = word2count[key]/max(word2count.values())

		#sentence scores
		sent2score = {}
		for sentence in sentences:
		    for word in nltk.word_tokenize(sentence.lower()):
		        if word in word2count.keys():
		            if len(sentence.split(' ')) < 25:
		                if sentence not in sent2score.keys():
		                    sent2score[sentence] = word2count[word]
		                else:
		                    sent2score[sentence] += word2count[word]

		# return best 5 lines
		best_sentences = heapq.nlargest(5, sent2score, key=sent2score.get)

		print('*'*25)
		for sentence in best_sentences:
			print(sentence)
		print('*'*25)

Scikit = Summarize('https://en.wikipedia.org/wiki/Scikit-learn')
Scikit.run()