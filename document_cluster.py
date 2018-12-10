from __future__ import print_function
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from sklearn.metrics.pairwise import cosine_similarity

# import csv
# import numpy as np
# import os
# import codecs
# import matplotlib
# from sklearn import feature_extraction
# import mpld3
# import helper

print("Creating output file...")
# read summary file:
summary = open('dataset/summary.txt', 'r').readlines()

stopwords = nltk.corpus.stopwords.words('english')


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
vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index=totalvocab_tokenized)
print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
print('Here is the vocab_frame:')
print(vocab_frame)
print("Vectorization text frequencies...")


tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                   min_df=0.1, stop_words='english',
                                   use_idf=True, tokenizer=tokenize_only, ngram_range=(1, 2))

tfidf_matrix = tfidf_vectorizer.fit_transform(summary)
terms = tfidf_vectorizer.get_feature_names()

dist = 1 - cosine_similarity(tfidf_matrix)

print("Clustering data using: K-Mean clustering...")

num_clusters = 8

km = KMeans(n_clusters=num_clusters)

km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

records = {'cluster': clusters, 'summary': summary}
frame = pd.DataFrame(records, index=[clusters], columns=['cluster', 'summary'])
print(frame['cluster'].value_counts())

print(frame)

print("Top terms per cluster:")
print()
# sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
print(order_centroids)

for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')

    for ind in order_centroids[i, :6]:  # replace 6 with n words per cluster
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print()

