# coding: utf-8
import json, string, re, sys
from pyspark import SparkContext, SparkConf

# spark-submit task1_first_attempt.py <input_file> <output_file> <stopwords> <y> <m> <n>
input_file = sys.argv[1]
output_file = sys.argv[2]
stopwords_file = sys.argv[3]
y = int(sys.argv[4])
m = int(sys.argv[5])
n = int(sys.argv[6])

# A. The total number of reviews (0.5pts)
# import json file as review_data dictionary
# path = 'hw1_data/review.json'
# review_data = [json.loads(line) for line in open(input_file)]

# create RDD for review dataset and count number of records
conf = SparkConf().setAppName('task1').setMaster('local[*]')
conf.set("spark.driver.memory", "4g")
sc = SparkContext(conf=conf)

# review_rdd = sc.parallelize(review_data, numSlices=100)
review_rdd = sc.textFile(input_file).map(lambda x: json.loads(x)).persist()
num_reviews = review_rdd.count()
print('Total number of reviews is %d' % num_reviews)  # Total number of reviews is 1151625


# B. The number of reviews in a given year, y (0.5pts)
# year = 2017  # enter year you want
# define function to get specific year rdd
def return_year_rdd_given_y(review_rdd, y):
    year_rdd = review_rdd.filter(lambda x: str(y) in x['date'])
    return year_rdd


# count number of reviews in the given year
num_year = return_year_rdd_given_y(review_rdd, y).count()
print('Total number of review in a given year %d is %d' % (y, num_year))


# C. The number of distinct users who have written the reviews (0.5pts)
distinct_user_rdd = review_rdd.map(lambda x: x['user_id']).distinct()
num_distinct_user = distinct_user_rdd.count()
print('Total number of distinct user who written review is %d' % num_distinct_user)

# D. Top m users who have the largest number of reviews and its count (0.5pts)
# m = 5
top_m_rdd = review_rdd \
    .groupBy(lambda x: x['user_id']) \
    .mapValues(len) \
    .takeOrdered(m, key=lambda x: (-x[1], x[0]))
print('Top %d users who have the largest number of reviews and its count:' % m)

# save top_m_rdd in List[List[String, int]]
top_m_list = []
for k, v in top_m_rdd:
    top_m_list.append([k, v])
print(top_m_list)

# E. Top n frequent words in the review text. The words should be in lower cases. The following punctuations
# i.e., “(”, “[”, “,”, “.”, “!”, “?”, “:”, “;”, “]”, “)”, and the given stopwords are excluded (1pts)
# n = 5
punctuations = '()[],.!?:;'
regex = re.compile('[%s]' % re.escape(punctuations))

# get stopwords from stopwords.txt
stopwords = []
with open(stopwords_file) as file:
    for line in file:
        stopwords.append(line.replace('\n', ''))


# function for cleaning text without punctuations and stopwords, and all in lowercase
def clean_text(line):
    without_punctuations = regex.sub('', line)
    without_stopwords = ' '.join([w.lower() for w in without_punctuations.split() if w.lower() not in stopwords])
    return without_stopwords


top_n_frequent_word_rdd = review_rdd \
                        .map(lambda x: x['text']) \
                        .map(clean_text) \
                        .flatMap(lambda x: x.split(' ')) \
                        .map(lambda x: (x, 1)) \
                        .groupByKey() \
                        .mapValues(len) \
                        .takeOrdered(n, key=lambda x: (-x[1], x[0]))
print('Top %d frequent words in review text:' % n)
print(top_n_frequent_word_rdd)

# store top n frequent word in a list
top_n_frequent_word_list = []
for word, ct in top_n_frequent_word_rdd:
    top_n_frequent_word_list.append(word)

# store ans in dictionary
ans_dict = {'A': num_reviews,
            'B': num_year,
            'C': num_distinct_user,
            'D': top_m_list,
            'E': top_n_frequent_word_list}
# for i, q in enumerate(list(string.ascii_uppercase[:5])):
#     ans_dict[q] = ans[i]

# output as json file
with open(output_file, 'w') as f:
    json.dump(ans_dict, f)

# for spark-submit using
# spark-submit --driver-memory 4G hw1/task1_first_attempt.py hw1_data/review.json task1_ans.json hw1_data/stopwords 2018 10 10

# Vocareum
# spark-submit task1_first_attempt.py $ASNLIB/publicdata/review.json task1_ans.json $ASNLIB/publicdata/stopwords 2018 10 10
