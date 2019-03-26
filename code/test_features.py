import nltk
import csv
import io
from string import digits
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from collections import Counter
from decimal import *

getcontext().prec = 3
reload(sys)
sys.setdefaultencoding('utf8')

sid = SentimentIntensityAnalyzer()

sen_list_with_problem = list()
sen_list_without_problem = list()


with io.open('expertiza_new_clean_data.csv', encoding='utf8', errors='ignore') as csv_file:
   	csv_reader = csv.reader(csv_file, delimiter=',')
   	for row in csv_reader:
   		if row[2] == '1':
			sen_list_with_problem.append(''.join(x for x in row[0] if x.isalpha() or x ==' '))
		elif row[2] == '-1':
			sen_list_without_problem.append(''.join(x for x in row[0] if x.isalpha() or x ==' '))

list1 = list()
list2 = list()
vectorizer = 2
X = 2
def createTfIDFFeature(list1, list2):
	global vectorizer
	global X
	doc1 = ' '.join(list1)
	doc2 = ' '.join(list2)
	vectorizer = TfidfVectorizer()
	X = vectorizer.fit_transform([doc1, doc2])

def getIndex(word_test):
	global vectorizer
	global X
	index = 0
	for word in vectorizer.get_feature_names():
		if word == word_test:
			return index
		index += 1
	return -1

def getSenScore(sen):
	global vectorizer
	global X
	score_class1 = 0
	score_class2 = 0
	for word in sen.split(' '):
		index = getIndex(word)
		if index != -1:
			score_class1 += X[0, index]
			score_class2 += X[1, index]
	return [score_class1, score_class2]

def test_classifier():
	global vectorizer
	global X
	tot_count = 0.0
	corr_count = 0.0
	with io.open('expertiza_new_clean_data_test_set.csv', encoding='utf8', errors='ignore') as csv_file:
	   	csv_reader = csv.reader(csv_file, delimiter=',')
	   	
	   	for row in csv_reader:
	   		if row[2] != 0:
		   		sen = ''.join(x for x in row[0] if x.isalpha() or x ==' ')
		   		tot_count += 1.0
		   		res = getSenScore(sen)
		   		
		   		final_res = -1 if res[0]< res[1] else 1
		   		print final_res, row[2]
		   		if str(final_res) == row[2]:
		   			corr_count += 1.0
	print 'Final Result: ', corr_count/tot_count

createTfIDFFeature(sen_list_with_problem, sen_list_without_problem)
test_classifier()


'''
def sentimentScoreAttributeAnalysis():
	ans1 = 0
	ans2 = 0

	for i in range(0, len(sen_list_with_problem)):
		temp = sen_list_with_problem[i].split('.')
		count_temp = 0
		for sen in temp:		
			count_temp += sid.polarity_scores(sen)['neg']
		ans1 += count_temp/len(temp)
		list1.append(count_temp)
	plt.scatter([0 for i in range(0, len(list1))], list1, color = "blue", label='with problems')

	for i in range(0, len(sen_list_without_problem)):
		temp = sen_list_without_problem[i].split('.')
		count_temp = 0
		for sen in temp:		
			count_temp += sid.polarity_scores(sen)['neg']
		ans2 += count_temp/len(temp)
		list2.append(count_temp)
	plt.scatter([0.2 for i in range(0, len(list2))], list2, color = "red", label='without problems')

	print "Mean NEG value for Sen with problems: ", ans1/len(sen_list_with_problem)
	print "Mean NEG value for Sen without problems: ", ans2/len(sen_list_without_problem)

	ans1 = 0
	ans2 = 0
	for i in range(0, len(sen_list_with_problem)):
		temp = sen_list_with_problem[i].split('.')
		count_temp = 0
		for sen in temp:		
			count_temp += sid.polarity_scores(sen)['pos']
		ans1 += count_temp/len(temp)
		list1.append(count_temp)
	plt.scatter([2 for i in range(0, len(list1))], list1, color = "blue")

	for i in range(0, len(sen_list_without_problem)):
		temp = sen_list_without_problem[i].split('.')
		count_temp = 0
		for sen in temp:
			count_temp += sid.polarity_scores(sen)['pos']
		ans2 += count_temp/len(temp)
		list2.append(count_temp)
	plt.scatter([2.2 for i in range(0, len(list2))], list2, color = "red")
	plt.legend(loc='upper right')
	plt.show()

	print "Mean POS value for Sen with problems: ", ans1/len(sen_list_with_problem)
	print "Mean POS value for Sen without problems: ", ans2/len(sen_list_without_problem)

sentimentScoreAttributeAnalysis()
'''

'''
def getWordTypeCout():
	NN = []
	VB = []
	AD = []
	ADV = []
	for i in range(0, len(sen_list_with_problem)):
		sen_comment = ''.join(x for x in sen_list_with_problem[i] if x.isalpha() or x ==' ')
		tokens = nltk.word_tokenize(sen_comment.lower())
		text = nltk.Text(tokens)
		tags = nltk.pos_tag(text)
		NN_count = 0.0
		VB_count = 0.0
		AD_count= 0.0
		ADV_count = 0.0
		counts = Counter(tag for word,tag in tags)
		tot = sum(counts.values())
		for ele in counts:
			if ele == 'NN' or ele == 'NNP' or ele == 'NNS':
				NN_count += counts[ele]
			if ele == 'RB' or ele == 'RBR' or ele == 'RBS':
				ADV_count += counts[ele]
			if ele == 'VB' or ele == 'VBD' or ele == 'VBG' or ele == 'VBN' or ele == 'VBP' or ele == 'VBZ':
				VB_count += counts[ele]
			if ele == 'JJ' or ele == 'JJR' or ele == 'JJS':
				AD_count += counts[ele]
		if tot != 0:
			NN.append(NN_count/tot);VB.append(VB_count/tot);AD.append(AD_count/tot);ADV.append(ADV_count/tot)

	ls1 = [round(x, 2) for x in NN]	
	ls2= [round(x, 2) for x in VB]
	ls3= [round(x, 2) for x in AD]
	ls4= [round(x, 2) for x in ADV]

	plt.scatter([0 for i in range(0, len(ls1))], ls1, color = "blue", label='with problems')
	plt.scatter([1 for i in range(0, len(ls2))], ls2, color = "blue")
	plt.scatter([2 for i in range(0, len(ls3))], ls3, color = "blue")
	plt.scatter([3 for i in range(0, len(ls4))], ls4, color = "blue")

	NN = []
	VB = []
	AD = []
	ADV = []
	for i in range(0, len(sen_list_without_problem)):
		sen_comment = ''.join(x for x in sen_list_without_problem[i] if x.isalpha() or x ==' ')
		tokens = nltk.word_tokenize(sen_comment.lower())
		text = nltk.Text(tokens)
		tags = nltk.pos_tag(text)
		NN_count = 0.0
		VB_count = 0.0
		AD_count= 0.0
		ADV_count = 0.0
		counts = Counter(tag for word,tag in tags)
		tot = sum(counts.values())
		for ele in counts:
			if ele == 'NN' or ele == 'NNP' or ele == 'NNS':
				NN_count += counts[ele]
			if ele == 'RB' or ele == 'RBR' or ele == 'RBS':
				ADV_count += counts[ele]
			if ele == 'VB' or ele == 'VBD' or ele == 'VBG' or ele == 'VBN' or ele == 'VBP' or ele == 'VBZ':
				VB_count += counts[ele]
			if ele == 'JJ' or ele == 'JJR' or ele == 'JJS':
				AD_count += counts[ele]
		if tot != 0:
			NN.append(NN_count/tot);VB.append(VB_count/tot);AD.append(AD_count/tot);ADV.append(ADV_count/tot)

	ls1 = [round(x, 2) for x in NN]
	ls2 = [round(x, 2) for x in VB]
	ls3 = [round(x, 2) for x in AD]
	ls4 = [round(x, 2) for x in ADV]

	plt.scatter([0.1 for i in range(0, len(ls1))], ls1, color = "red", label='wihout problems')
	plt.scatter([1.1 for i in range(0, len(ls2))], ls2, color = "red")
	plt.scatter([2.1 for i in range(0, len(ls3))], ls3, color = "red")
	plt.scatter([3.1 for i in range(0, len(ls4))], ls4, color = "red")
	plt.legend(loc='upper right')
	plt.show()

getWordTypeCout()
'''