#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scipy as sp
import sklearn as sk
import string as s
import sys

import json
from random import randrange

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC, NuSVC


# In[2]:


def train_test_split_random(dataset, data_size,split=0.7):
    train_size_counter = 0
    indices = []
    train = pd.DataFrame()
    train_size = split * data_size
    dataframe_copy = dataset
    while train_size_counter < train_size:
        train_size_counter = train_size_counter + 1
        index = randrange(data_size)
        while(check_if_element_in_list(index, indices)):
            index = randrange(data_size)
                 
        indices.append(index)
    
    for i in range(len(indices)+1):
        train = train.append(dataframe_copy.iloc[[i]])
        
    dataframe_copy = dataframe_copy.drop(labels = indices, axis=0)
    dataframe_copy = dataframe_copy.reset_index(drop=True)
        
    return train, dataframe_copy

def check_if_element_in_list(x, lista):
    if x in lista:
        return True
    else:
        return False


# In[8]:


class Transformer(object):
    
    def __init__(self, train_set, test_set):
        self.train_set = train_set
        self.test_set = test_set

    def filter_data_frame(self, data_frame, function):
        return data_frame.apply(function)

    def tokenize(self):
        def split_into_words(input_text):
            words = input_text.split()
            return words
        self.train_set = self.filter_data_frame(self.train_set, split_into_words)
        self.test_set = self.filter_data_frame(self.test_set, split_into_words)


    def transform_to_Z_case(self, z="LOWERCASE"):
        if z == "LOWERCASE":
            def to_lower_case(words):
                transformed_list = [i.lower() for i in words]
                return transformed_list
            self.train_set = self.filter_data_frame(self.train_set, to_lower_case)
            self.test_set = self.filter_data_frame(self.test_set, to_lower_case)
        else:
            def to_upper_case(words):
                transformed_list = [i.upper() for i in words]
                return transformed_list
            self.train_set = self.filter_data_frame(self.train_set, to_upper_case)
            self.test_set = self.filter_data_frame(self.test_set, to_upper_case)

    def remove_punctuation_marks(self):
        def remove_punctuations(words):
            transformed_list = []
            for i in words:
                for j in s.punctuation:
                    i = i.replace(j, '')
                transformed_list.append(i)
            return transformed_list
        self.train_set = self.filter_data_frame(self.train_set, remove_punctuations)
        self.test_set = self.filter_data_frame(self.test_set, remove_punctuations)

    def trim_text(self):
        def trim_spaces_along_word(words):
            transformed_list = [i.strip() for i in words]
            return transformed_list
        self.train_set = self.filter_data_frame(self.train_set, trim_spaces_along_word)
        self.test_set = self.filter_data_frame(self.test_set, trim_spaces_along_word)

    def remove_stopwords(self):
        def remove_stop_words(input_text):
            stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                         "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
                         'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
                         'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is',
                         'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
                         'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
                         'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
                         'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                         'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
                         'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                         'so', 'than', 'too', 'very', 's', 't', 'can','cant', 'will', 'just', 'don', "don't", "should",
                         "should've",'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
                         'didn',"didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn',
                         "isn't",'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
                         'shouldn',"shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't",
                         'whos','whose','whats', 'whatre', 'wheres', 'whered','whys']
            transformed_list = [i for i in input_text if i not in stopwords]

            return transformed_list
        self.train_set = self.filter_data_frame(self.train_set, remove_stop_words)
        self.test_set = self.filter_data_frame(self.test_set, remove_stop_words)


    def detokenize(self, dataset):
        return dataset.apply(lambda x: ''.join(i + ' ' for i in x))


    def pipe(self):
        self.tokenize()
        self.transform_to_Z_case() 
        self.remove_punctuation_marks()
        self.trim_text()
        self.remove_stopwords()
        
    def count_words(self):
        #Da se vidi koje reci se najvise upotrebljuju
        dictionary = {}

        for element in self.train_set:
            for token in element:
                if token in dictionary:
                    dictionary[token] += 1
                else:
                    dictionary[token] = 1

        dictionary = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)         
        return dictionary
        


# In[18]:


# all_data = pd.read_json('train.json')
# train_data, test_data = train_test_split_random(all_data, all_data.shape[0], split=0.7 )
# X_train, Y_train, X_test, Y_test = train_data.text, train_data.clickbait, test_data.text, test_data.clickbait

# processed_text = Transformer(X_train, X_test)
# processed_text.pipe()
    
# X_train, X_test = processed_text.train_set, processed_text.test_set
# X_train = processed_text.detokenize(X_train)
# X_test = processed_text.detokenize(X_test)
    
# #tfid vectorizer
# tf_id_vectorizer = TfidfVectorizer(sublinear_tf=True, smooth_idf=False)
# train_X =  tf_id_vectorizer.fit_transform(X_train).toarray()
# test_X =  tf_id_vectorizer.transform(X_test).toarray()
    
# linear_SVM = LinearSVC()  #C=0.325, fit_intercept=False
# linear_SVM.fit(train_X, Y_train)
# y_pred = linear_SVM.predict(test_X)


# # svc = SVC(gamma='auto')
# # svc.fit(train_X, Y_train)
# # y_pred = svc.predict(test_X)

# # nu = NuSVC(gamma='scale')
# # nu.fit(train_X, Y_train)
# # y_pred = nu.predict(test_X)

# print(accuracy_score(Y_test, y_pred))
# #print(f1_score(Y_test, y_pred))


# In[ ]:


if(len(sys.argv) != 3):
    print("Mora imati dva argumenta 'train.csv' 'test.csv'")
    exit()
else:
    train_csv = sys.argv[1]
    test_csv = sys.argv[2]  
    
    train_data = pd.read_json(train_csv)
    test_data = pd.read_json(test_csv)
    
    X_train, Y_train, X_test, Y_test = train_data.text, train_data.clickbait, test_data.text, test_data.clickbait
    
    processed_text = Transformer(X_train, X_test)
    processed_text.pipe()

    X_train, X_test = processed_text.train_set, processed_text.test_set
    X_train = processed_text.detokenize(X_train)
    X_test = processed_text.detokenize(X_test)

    #tfid vectorizer
    tf_id_vectorizer = TfidfVectorizer(sublinear_tf=True, smooth_idf=False)
    train_X =  tf_id_vectorizer.fit_transform(X_train).toarray()
    test_X =  tf_id_vectorizer.transform(X_test).toarray()

    linear_SVM = LinearSVC()  #C=0.325, fit_intercept=False
    linear_SVM.fit(train_X, Y_train)
    y_pred = linear_SVM.predict(test_X)


    # svc = SVC(gamma='auto')
    # svc.fit(train_X, Y_train)
    # y_pred = svc.predict(test_X)

    # nu = NuSVC(gamma='scale')
    # nu.fit(train_X, Y_train)
    # y_pred = nu.predict(test_X)

    print(accuracy_score(Y_test, y_pred))
    #print(f1_score(Y_test, y_pred))

