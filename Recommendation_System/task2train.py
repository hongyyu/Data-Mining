# coding: utf-8
from datetime import datetime
from pyspark import SparkContext, SparkConf
from collections import defaultdict, Counter
from itertools import count
from operator import add
import string
import math
import json
import re
import sys


def timer(start_time=None):
    """Counting processing time
    :parameter start_time: if None start to computing time, if not None compute total processing time
    :type start_time: None, datetime
    :return start datetime or print out total processing time
    """
    # Get starting time
    if not start_time:
        start_time = datetime.now()
        return start_time
    # Calculate running time
    elif start_time:
        sec = (datetime.now() - start_time).total_seconds() % (3600 * 60)
        print('Duration: %ss' % round(sec, 2))


def clean_text(text, is_rare_words=False):
    """Remove punctuations and stopwords
    :param text: review text for each record in the train_review.json file
    :param is_rare_words:
    :return: cleaned file
    """
    if is_rare_words:
        # Remove word if it is in the rare_words
        result = ' '.join([w.lower() for w in text.split() if w.lower() not in rare_words])
    else:
        # Catch punctuations and replace with empty string
        without_punctuations = regex.sub('', text)
        # Remove word if it is in the stopwords_list
        result = ' '.join([w.lower() for w in without_punctuations.split() if w.lower() not in stopwords_list])
    return result


def get_represent_word_docs(text_rdd):
    """Term frequency * Inverse Doc Frequency
    :param text_rdd: cleaned_rdd without punctuations, stopwords, and rare words
    :return: tf_idf vectors
    """
    N = cleaned_rdd.count()                 # Total number of docs
    ni_dict = defaultdict(int)              # Number of docs that mention term i
    # Count number of term i appear for the whole data set
    unique_word_docs = text_rdd.flatMap(lambda line: [(word, 1) for word in list(set(line[1].split()))])\
        .reduceByKey(add).collect()
    # Put corresponding term and frequency into dictionary
    for w in unique_word_docs:
        term_i, frequency = w[0], w[1]
        ni_dict[term_i] = frequency
    word2index = dict(zip(sorted(ni_dict.keys()), count(0)))

    # Define function to calculate tf-idf vector and then construct business profile using top 200 words
    def calculate_tfidf_significant_word(text, mapping):
        """Compute tf-idf vector for each document and then keep on first 200 important words
        :param text: string of text that need to be computed
        :param mapping: word dictionary with words count for each document
        :return: representative words
        """
        temp_vector = []
        if text:
            for word in text:
                tf = mapping[word] / max(mapping.values())
                idf = math.log(N / ni_dict[word], 2)
                temp_vector.append((word, tf * idf))
        represent_word = list(set(sorted(temp_vector, key=lambda x: -x[1])))[:200]
        return represent_word

    # Calculate tf_idf vector for each term for each document and build business profile for each document
    significant_word = text_rdd.map(lambda line: (line[0], line[1].split(), Counter(line[1].split())))\
        .map(lambda line: (line[0], [word[0] for word in calculate_tfidf_significant_word(line[1], line[2])]))\
        .cache()

    return significant_word, word2index


if __name__ == '__main__':
    # Start time
    start = timer()

    # Initial parameters
    train_file = 'data/train_review.json'
    model_file = 'task2.model'
    stopwords = 'data/stopwords'
    # train_file = sys.argv[1]
    # model_file = sys.argv[2]
    # stopwords = sys.argv[3]

    # Create RDD
    conf = SparkConf().setAppName('task2train').setMaster('local[*]')
    conf.set("spark.driver.memory", "10g")
    conf.set("spark.driver.maxResultSize", "4g")
    sc = SparkContext(conf=conf)
    rdd = sc.textFile(train_file).map(lambda x: json.loads(x)).cache()

    # Loading stopwords
    stopwords_list = []
    with open(stopwords) as file:
        for record in file:
            stopwords_list.append(record.replace('\n', ''))
    file.close()

    # Remove punctuations, stopwords, and rare words
    regex = re.compile('[%s]' % re.escape(string.punctuation + string.digits))

    # Cleaning text in rdd
    cleaned_rdd = rdd.map(lambda line: (line['business_id'], line['text']))\
        .groupByKey()\
        .map(lambda line: (line[0], clean_text(' '.join(line[1])))).cache()
    flat_rdd = cleaned_rdd.map(lambda line: line[1])\
        .flatMap(lambda line: [(word, 1) for word in line.split()])
    total_count = flat_rdd.count()
    rare_words = flat_rdd.reduceByKey(add)\
        .filter(lambda line: line[1]/total_count < 0.000001)\
        .map(lambda line: line[0])\
        .collect()
    rare_words = set(rare_words)
    # Remove rare word in each document
    cleaned_rdd = cleaned_rdd.map(lambda line: (line[0], clean_text(line[1], True))).cache()

    # Transform cleaned document into tf-idf vectors and get top 200 significant word for each doc
    significant_word_rdd, word2index = get_represent_word_docs(cleaned_rdd)

    # Construct business profile
    business_profile = significant_word_rdd\
        .map(lambda line: (line[0], set(word2index[word] for word in line[1]))).collect()
    # Dictionary to store business_id and corresponding text
    business_index = defaultdict(str)
    for business in business_profile:
        business_index[business[0]] = business[1]
    # Construct user profile
    user_profile = rdd.map(lambda line: (line['user_id'], line['business_id']))\
        .groupByKey()\
        .map(lambda line: (line[0], set().union(*[business_index[business_id] for business_id in line[1]])))\
        .collect()

    # Output business profile and user profile
    with open(model_file, 'w') as f:
        # Business profile
        business_dict = defaultdict(str)
        for business in business_profile:
            business_dict[business[0]] = str(business[1])
        json.dump(business_dict, f)
        f.write('\n')
        # User profile
        user_dict = defaultdict(str)
        for user in user_profile:
            user_dict[user[0]] = str(user[1])
        json.dump(user_dict, f)
    f.close()
    # Finish time
    timer(start)

# spark-submit hw3/task2train.py hw3/data/train_review.json hw3/task2.model hw3/data/stopwords