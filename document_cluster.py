print("Importing libraries...")
# import csv
# import numpy as np
import pandas as pd
# import nltk
import re
# import os
# import codecs
# import matplotlib
# from sklearn import feature_extraction
# import mpld3
import nltk

# import helper
print("Creating output file...")
# read summary file:
summary = open('dataset/summary.txt', 'r').readlines()

stopwords = nltk.corpus.stopwords.words('english')

from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")


def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


totalvocab_stemmed = []
totalvocab_tokenized = []

print("Tokenizing and stemming words...")
for i in summary:
    allwords_stemmed = tokenize_and_stem(i)  # for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed)  # extend the 'totalvocab_stemmed' list

    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

print("Creating DataFrame...")
vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index=totalvocab_stemmed)
print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
print(vocab_frame.head())

print("Vectorization text frequencies...")
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                   min_df=0.1, stop_words='english',
                                   use_idf=True, tokenizer=tokenize_only, ngram_range=(1, 2))

tfidf_matrix = tfidf_vectorizer.fit_transform(summary)
print(tfidf_matrix.shape)

terms = tfidf_vectorizer.get_feature_names()
print(terms)

from sklearn.metrics.pairwise import cosine_similarity

dist = 1 - cosine_similarity(tfidf_matrix)
print(dist)

print("Clustering data using: K-Mean clustering...")
from sklearn.cluster import KMeans
num_clusters = 8

km = KMeans(n_clusters=num_clusters)

km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

print(clusters)

records = {'cluster': clusters, 'summary': summary}
frame = pd.DataFrame(records, index=[clusters], columns=['cluster', 'summary'])
print(frame['cluster'].value_counts())

print(frame)