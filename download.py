import csv
import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
# import mpld3
import nltk
# select All from the dialogue box and click Download
# nltk.download()

# read the dataset
dataset = csv.reader(open('dataset/air_crashes.csv', mode='r'))
for index, row in enumerate(dataset):

        # for column_index, col_value in enumerate(row):
        #     print(column_index, col_value)
    if len(row[12]) == 0:
        print("N/A")
    else:
        print(type(row[12]))
