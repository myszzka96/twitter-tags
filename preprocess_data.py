import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns
import imgkit
import re
import nltk
import spacy
import string
from textblob import TextBlob
from collections import Counter
from sklearn import tree
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectFromModel
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth, association_rules
from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.data import iris_data
from mlxtend.plotting import plot_decision_regions
from mlxtend.preprocessing import TransactionEncoder
from pandas.plotting import table
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from emot.emo_unicode import UNICODE_EMO, EMOTICONS# Converting emojis to words


#df['class'].value_counts()

#diffrenciating between diffrent attributes with similar values
#data = df[['Tweet Id','Text','name','Screen Name','UTC','Created At','Favorites','Retweets','Language','Client','Tweet Type','URLs','Hasgtags','Mentions','Media Type','Media URLs']].values


def is_number(s):
	try:
		float(s)
		return True
	except ValueError:
		return False

#remove extra new line characters
f = open('to_only.csv', 'w')
with open('opinion_2.csv', 'r') as fp:
	s = fp.readline()
	while s:
		ids = str(s[1:19])
		#print(ids)
		if is_number(ids):
			f.write('\n')
			s = s.strip('\n')
			f.write(s)
		else:
			s = s.strip('\n')
			f.write(s)
		s=fp.readline()	
f.close()

# read the dataset
#dp = pd.read_csv('to_only.csv', header=None, sep='\n')
#dp.head()
#print (dp)

#data = dp[['Tweet Id','Text','name','Screen Name','UTC','Created At','Favorites','Retweets','Language','Client','Tweet Type','URLs','Hasgtags','Mentions','Media Type','Media URLs']].values

#remove all entries that are not in english
f = open('eng_only.csv', 'w')
with open('to_only.csv', 'r') as fp:
	s = fp.readline()
	f.write(s)
	s = fp.readline()
	while s:
		sts = re.split(r'\t+', s)
		#l = ""
		if sts[8] == 'en':
			#for i in sts:
			#	if i.startswith('\"') and i.endswith('\"'):
			#		i = i[1:-1]
			#		l = l + i + "\t"
			#	else:
			#		l = l+
			f.write(s)
		s=fp.readline()	
f.close()

# read the dataset
df = pd.read_csv('eng_only.csv', sep='\t')
df.head()	

#data = df[['Tweet Id','Text','name','Screen Name','UTC','Created At','Favorites','Retweets','Language','Client','Tweet Type','URLs','Hasgtags','Mentions','Media Type','Media URLs']].values

def prep(text):
	#all to lower case
	df['text_lower'] = text.str.lower()
	df['text_lower'].head()	

	#remove links
	df['text_lower'] = df['text_lower'].str.replace('http\S+','')

	#remove punctuation
	df['text_punct'] = df['text_lower'].str.replace('[^\w\s]',' ')
	df['text_punct'].head()

	#remove all single char
	df['text_punct'] = df['text_punct'].str.replace('\s+[a-zA-Z]\s+',' ')
	df['text_punct'] = df['text_punct'].str.replace('\^[a-zA-Z]\s+',' ')

	#multiple spaces to single space
	df['text_punct'] = df['text_punct'].str.replace('\s+',' ')

	#remove prefixed 'b' if bite data
	df['text_punct'] = df['text_punct'].str.replace('^b\s+','')

	#stop-word removal
	'''from nltk.corpus import stopwords
	STOPWORDS = set(stopwords.words('english'))

	def stopwords(text):
		return " ".join([word for word in str(text).split() if word not in STOPWORDS])

	df["text_stop"] = df["text_punct"].apply(stopwords)
	df["text_stop"].head()
	'''
	#common word removal
	cnt = Counter()
	for text in df["text_punct"].values:
		for word in text.split():
			cnt[word] += 1

	cnt.most_common(10)
	freq = set([w for (w, wc) in cnt.most_common(10)])
	def freqwords(text):
		return " ".join([word for word in str(text).split() if word not in freq])
	df["text_common"] = df["text_punct"].apply(freqwords)
	df["text_common"].head()

	#rare word removal
	freq = pd.Series(' '.join(df['text_common']).split()).value_counts()[-10:] # 10 rare words
	freq = list(freq.index)
	df['text_rare'] = df['text_common'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
	df['text_rare'].head()

	#spell correction
	df['text_rare'][:5].apply(lambda x: str(TextBlob(x).correct()))

	#emoji removal
	def emoji(string):
		emoji_pattern = re.compile("["
		                   #u"\U0001F600-\U0001F64F"  # emoticons
		                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
		                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
		                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
		                   u"\U00002702-\U000027B0"
		                   u"\U000024C2-\U0001F251"
		                   "]+", flags=re.UNICODE)
		return emoji_pattern.sub(r'', string)
	df['text_emo'] = df['text_rare'].apply(emoji)

	#verbalize emoji & emoticons
	def convert_emojis(text):
		for emot in UNICODE_EMO:
			text = text.replace(emot, "_".join(UNICODE_EMO[emot].replace(",","").replace(":","").split()))
			return text
	# Converting emoticons to words    
	def convert_emoticons(text):
		for emot in EMOTICONS:
			text = re.sub(u'('+emot+')', "_".join(EMOTICONS[emot].replace(",","").split()), text)
			return text
	# Passing both functions to 'text_rare'
	df['text_v'] = df['text_emo'].apply(convert_emoticons)
	df['text_v'] = df['text_emo'].apply(convert_emojis)

	pd.set_option("display.max_rows", None, "display.max_columns", None)
	df.to_csv("formated.csv")

def get_tweet_sentiment(tweet): 
	''' 
	Utility function to classify sentiment of passed tweet 
	using textblob's sentiment method 
	'''
	# create TextBlob object of passed tweet text 
	analysis = TextBlob(tweet) 
	# set sentiment 
	if analysis.sentiment.polarity > 0: 
		return 'positive'
	elif analysis.sentiment.polarity == 0: 
		return 'neutral'
	else: 
		return 'negative'

prep(df['Text'])

df['sentiment_blob'] = df['text_v'].apply(get_tweet_sentiment)
df['sentiment_blob'].head()

#create labels
labels = df["sentiment_blob"].values
labels.shape

indices = df.index.values
indices.shape

#define text to be processed
data = df['text_v'].values
data.shape

#stop words bag of words
vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
data = vectorizer.fit_transform(data).toarray()


#divide into train and test
X_train, X_test, indices_train, indices_test = train_test_split(data, indices, test_size=0.3, random_state=0)

y_train, y_test = labels[indices_train],  labels[indices_test]

#train the model - classifier
text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier.fit(X_train, y_train)

#predictions
predictions = text_classifier.predict(X_test)
pred_pr = text_classifier.predict_proba(X_test)
pred_pro = text_classifier.predict_proba(X_train)

#put into table
df1 = pd.DataFrame(pred_pr, index = indices_test)
df2 = pd.DataFrame(pred_pro, index = indices_train)
frames = [df1,df2]
res = pd.concat(frames)
res = res.sort_index()

df['sentiment_confidence'] = res.max(axis=1)

#model evaluation
print('Confiusion matrix: ')
conf_matrix = pd.crosstab(y_test, predictions, rownames=['Actual'], colnames=['Predicted'])
print(conf_matrix)
print('\nClassification report: \n')
class_r = classification_report(y_test,predictions)
print(class_r)
print("Accuracy score: " + str(accuracy_score(y_test, predictions)))

#pretty charts and graphs
plot_size = plt.rcParams["figure.figsize"] 
print(plot_size[0]) 
print(plot_size[1])

plot_size[0] = 8
plot_size[1] = 6
plt.rcParams["figure.figsize"] = plot_size 

#piechart of pos/neg/neu
df.sentiment_blob.value_counts().plot(kind='pie', autopct='%1.0f%%')
plt.savefig('sentiment.png')
plt.clf()

#average conf
sns.barplot(x='sentiment_blob', y='sentiment_confidence' , data=df)
plt.savefig('sentiment_confidence.png')
plt.clf()

#conf matrix
sns.heatmap(conf_matrix, annot=True)
plt.savefig('confiusion_matrix_sent.png')
plt.clf()

#word cloud
from wordcloud import WordCloud, STOPWORDS

def wc(data,bgcolor,title):
    plt.figure()
    wc = WordCloud(background_color = bgcolor, max_words = 1000,  max_font_size = 50)
    wc.generate(' '.join(data))
    plt.imshow(wc)
    plt.axis('off')

wc(df['text_v'], 'black', 'Common words')
plt.savefig('word_cloud_sentiment.png')
plt.clf()

