from __future__ import division, print_function, unicode_literals
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

char = ("'", '"' , "." ,',' , '!' , '?' , '(' , ')' , '/',":","*" )
mails = pd.read_csv('spam.csv', encoding = 'latin-1')
mails.head()
#print(mails)

mails = mails.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1)
mails = mails.rename(columns = {"v1":"label", "v2":"message"})

mails.head()
#print(mails)

mails.loc[:,'label'] = mails.label.map({'ham':0, 'spam':1})
mails.head()
#print(mails)

train_sms,train_label,test_sms,test_label = list(),[],list(),[]
for i in range(mails.shape[0]):
    if np.random.uniform(0, 1) < 0.75:
        train_sms.append(mails["message"][i])
        train_label.append(mails["label"][i])

    else:
        test_sms.append(mails["message"][i])
        test_label.append(mails["label"][i])

def process_message(message, lower_case = True, stem = True, stop_words = True):
    if lower_case:
        message = message.lower()
    words = word_tokenize(message)
    if stop_words:
        sw = stopwords.words('english')
        words = [word for word in words if word not in sw]
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
    return words

for i in range (0,len(train_sms)):
    words = process_message(train_sms[i])
    train_sms[i] = ''
    for w in words:
        for c in char:
            w = w.replace(c,'')
        train_sms[i] += w+' '

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(train_sms).toarray()

for i in range (0,len(test_sms)):
    words = process_message(test_sms[i])
    test_sms[i] = ''
    for w in words:
        test_sms[i] += w+' '
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_sms).toarray()
trainVocab = vectorizer.vocabulary_

vectorizer = CountVectorizer(vocabulary=trainVocab)
X_test = vectorizer.fit_transform(test_sms).toarray()

khanh = MultinomialNB()
khanh.fit(X_train,train_label)
y_predict = khanh.predict(X_test)
print('Training size = %d, accuracy = %.2f%%' % \
      (len(X_train),accuracy_score(test_label, y_predict)*100))
blong = BernoulliNB()
blong.fit(X_train,train_label)
y_predict = blong.predict(X_test)
print('Training size = %d, accuracy = %.2f%%' % \
      (len(X_train),accuracy_score(test_label, y_predict)*100))
