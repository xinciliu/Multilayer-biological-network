from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import unicodedata
import sys


class Bayes:
    @staticmethod
    def MyMultinomialNB(train_matrix='', class_labels='', test_doc=''):
        X = np.array(train_matrix)
        Y = np.array(class_labels)
        clf = MultinomialNB()
        clf.fit(X, Y)
        result = []
        for x in test_doc:
            index = clf.predict(x.reshape(1, -1))
            reslist = [0, 1]
            result.append(reslist[index[0]])
        return result
