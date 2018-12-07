# this file takes the whole csv file and extract only the summary from the data
import csv
data_set = list(csv.reader(open('dataset/air_crashes.csv', 'r')))
summary = open('dataset/summary.txt', 'w')
for index, row in enumerate(data_set[1:]):
    if len(row[12]) != 0:
        summary.write(row[12] + "\n")

summary.close()
