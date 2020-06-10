# coding: utf-8
from datetime import datetime
from pyspark import SparkContext, SparkConf
from collections import defaultdict
import json
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


def construct_line(line):
    """Construct dictionary which storing id and corresponding rates
    :param line: a list of id and its corresponding rates
    :return: dictionary of id:rates key-value pairs
    """
    index_dict = defaultdict(float)
    for temp in list(line):
        user_id, star = temp[0], temp[1]
        index_dict[user_id] = star
    return index_dict


def get_prediction_user(line):
    business_id, user_id = line['business_id'], line['user_id']
    numerator, denominator, counter = 0, 0, 0
    try:
        for key, value in model_dict[user_id].items():
            if train_dict[key].get(business_id) is None:
                continue
            if counter >= num_neighbor:
                break
            list_without_user = [value for key, value in train_dict[key].items() if key != business_id]
            avg = sum(list_without_user) / len(list_without_user)
            numerator += (train_dict[key].get(business_id) - avg) * value
            denominator += abs(value)
            counter += 1
        try:
            ri_bar = sum(train_dict[user_id].values()) / len(train_dict[user_id])
            prediction = ri_bar + (numerator / denominator)
        except ZeroDivisionError:
            prediction = None
    except KeyError:
        prediction = None

    return user_id, business_id, prediction


def get_prediction_item(line):
    business_id, user_id = line['business_id'], line['user_id']
    numerator, denominator, counter = 0, 0, 0
    try:
        for key, value in model_dict[business_id].items():
            if train_dict[key].get(user_id) is None:
                continue
            if counter >= num_neighbor:
                break
            numerator += train_dict[key].get(user_id) * value
            denominator += abs(value)
            counter += 1
        try:
            prediction = numerator / denominator
        except ZeroDivisionError:
            prediction = None
    except KeyError:
        prediction = None

    return user_id, business_id, prediction


if __name__ == '__main__':
    # Start time
    start = timer()

    # Initial parameters
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    model_file = sys.argv[3]
    output_file = sys.argv[4]
    cf_type = sys.argv[5]

    # Set rdd
    conf = SparkConf().setAppName('task3predict').setMaster('local[*]')
    conf.set("spark.driver.memory", "4g")
    conf.set("spark.driver.maxResultSize", "4g")
    sc = SparkContext(conf=conf)

    # get train rdd and then save items and corresponding user_id and rates
    if cf_type == 'item_based':
        train_dict = sc.textFile(train_file)\
            .map(lambda x: json.loads(x))\
            .map(lambda line: (line['business_id'], (line['user_id'], line['stars'])))\
            .groupByKey()\
            .mapValues(construct_line)\
            .collectAsMap()
    elif cf_type == 'user_based':
        train_dict = sc.textFile(train_file) \
            .map(lambda x: json.loads(x)) \
            .map(lambda line: (line['user_id'], (line['business_id'], line['stars']))) \
            .groupByKey() \
            .mapValues(construct_line) \
            .collectAsMap()

    # Get model we trained and save key-value pairs in dict
    num_neighbor = 5
    model_rdd = sc.textFile(model_file).map(lambda x: json.loads(x))
    keys = list(model_rdd.take(1)[0].keys())
    pair1, pair2 = keys[0], keys[1]
    model_dict = model_rdd\
        .flatMap(lambda line: ((line[pair1], (line[pair2], line['sim'])), (line[pair2], (line[pair1], line['sim']))))\
        .groupByKey()\
        .mapValues(lambda line: {item[0]: item[1] for item in sorted(line, key=lambda x: -x[1])})\
        .collectAsMap()

    # Get the test rdd and then predict the result with valid pairs
    test_rdd = sc.textFile(test_file).map(lambda x: json.loads(x)).cache()
    if cf_type == 'item_based':
        test_res = test_rdd\
            .map(lambda line: get_prediction_item(line))\
            .filter(lambda line: line[2] is not None).collect()
    elif cf_type == 'user_based':
        test_res = test_rdd \
            .map(lambda line: get_prediction_user(line)) \
            .filter(lambda line: line[2] is not None).collect()

    # Output as json
    with open(output_file, 'w') as f:
        for line in test_res:
            ans = {
                'user_id': line[0],
                'business_id': line[1],
                'stars': line[2]
            }
            json.dump(ans, f)
            f.write('\n')
    f.close()

    # Finish time
    timer(start)

# spark-submit task3predict.py data/train_review.json data/test_review.json task3item.model task3item.predict item_based
# spark-submit task3predict.py data/train_review.json data/test_review.json task3user.model task3user.predict user_based
