# coding: utf-8
from datetime import datetime
from pyspark import SparkContext, SparkConf
from collections import defaultdict, Counter
from operator import add
import itertools as it
import sys
import math
import json
import random


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


def random_hash_function(hash_size):
    """Randomly generate parameters for hash function and generate a empty signature matrix
    :parameter bins: length of unique user_id which is also total number of records
    :parameter num_business: length of unique business_id
    :returns parameters, signature_matrix
    """
    def get_prime_list(upper):
        primes = []
        is_prime = [True for _ in range(upper + 1)]
        for num in range(2, upper + 1):
            if is_prime[num]:
                primes.append(num)
                for i in range(num, upper + 1, num):
                    is_prime[i] = False
        return primes
    num_a = [random.randrange(sys.maxsize) for _ in range(hash_size)]      # Parameter a for hash function
    num_b = [random.randrange(sys.maxsize) for _ in range(hash_size)]      # Parameter b for hash function
    primes_list = get_prime_list(50000)

    return [(num_a[i], num_b[i], random.choice(primes_list), 26184) for i in range(hash_size)]


def get_signature_matrix(line):
    signature = [float('inf') for _ in range(num_hash_func)]
    for user in line:
        hash_list = [hash_function(user, *H[index]) for index in range(num_hash_func)]
        for i in range(num_hash_func):
            if hash_list[i] < signature[i]:
                signature[i] = hash_list[i]
    return signature


def hash_function(num_row, a, b, p, m):
    """Hash function for calculating signature matrix
    :return hash number for each row number
    """
    return ((a * num_row + b) % p) % m


def jaccard_similarity(b1, b2):
    """Jaccard similarity between two set b1 and b2
    :param b1: set of index if there is a rate for business b1
    :param b2: set of index if there is a rate for business b2
    :return: jaccard similarity of two sets
    """
    return len(b1.intersection(b2))/len(b1.union(b2))


def construct_matrix(rdd, is_user_based=None):

    def construct_line(line):
        index_dict = defaultdict(float)
        for temp in list(line):
            user_id, star = temp[0], temp[1]
            index_dict[user_id] = star
        return index_dict
    if is_user_based:
        matrix = rdd.map(lambda line: (line['user_id'], (line['business_id'], line['stars']))) \
            .groupByKey() \
            .mapValues(construct_line).cache()
    else:
        matrix = rdd.map(lambda line: (line['business_id'], (line['user_id'], line['stars']))) \
            .groupByKey() \
            .mapValues(construct_line).cache()

    return matrix


def pearson_correlation(line):
    item_one = line[0]
    item_one_index = id_set_dict[item_one]
    res_dict = defaultdict(float)
    for candidate in list(line[1]):
        item_two_index = id_set_dict[candidate]
        sum_xy, sum_xx, sum_yy = 0, 0, 0
        co_index = set(item_one_index.keys()).intersection(set(item_two_index.keys()))
        avg_one = sum(item_one_index.values())/len(item_one_index)
        avg_two = sum(item_two_index.values())/len(item_two_index)
        for index in co_index:
            sum_xy += (item_one_index[index] - avg_one) * (item_two_index[index] - avg_two)
            sum_xx += (item_one_index[index] - avg_one) ** 2
            sum_yy += (item_two_index[index] - avg_two) ** 2
        if not sum_xy or not sum_xx or not sum_yy:
            continue
        else:
            pearson_value = sum_xy / (math.sqrt(sum_xx) * math.sqrt(sum_yy))
            if pearson_value <= 0:
                continue
        res_dict[candidate] = pearson_value
    return item_one, res_dict


if __name__ == '__main__':
    # Start time
    start = timer()

    # Initial parameters
    train_file = sys.argv[1]        # Input path of train file
    model_file = sys.argv[2]        # Output path of model file
    cf_type = sys.argv[3]           # Type of collaborative filter

    # Create rdd
    conf = SparkConf().setAppName('task3train').setMaster('local[*]')
    conf.set("spark.driver.memory", "10g")
    conf.set("spark.driver.maxResultSize", "4g")
    sc = SparkContext(conf=conf)
    rdd = sc.textFile(train_file).map(lambda x: json.loads(x))

    # For item based collaborative filter recommendation system
    if cf_type == 'item_based':
        # Construct matrix with rates corresponding to their user and business id
        matrix_rdd = construct_matrix(rdd)
        # Find pairs if their number of co-rated user is at least 3
        co_rated = matrix_rdd.cartesian(matrix_rdd) \
            .filter(lambda line: line[0][0] != line[1][0] and
                                 len(set(line[0][1].keys()).intersection(set(line[1][1].keys()))) >= 3) \
            .map(lambda line: tuple(sorted((line[0][0], line[1][0])))) \
            .distinct() \
            .groupByKey()
        # Initial empty dictionary for saving business_id corresponding to set of users
        id_set_dict = matrix_rdd.collectAsMap()
        # Computing pearson correlation for each valid pair
        pearson_ans = co_rated.map(lambda line: pearson_correlation(line)) \
            .mapValues(lambda line: sorted(line.items(), key=lambda x: -x[1])) \
            .flatMap(lambda line: [(line[0], value) for value in line[1]]).collect()

    # For user based collaborative filter recommendation system
    elif cf_type == 'user_based':
        # Get user dictionary with corresponding set of business_id and rates
        id_set_dict = construct_matrix(rdd, True).collectAsMap()
        # Dictionary of business_id to its index and reversed
        business2index = rdd.map(lambda line: line['business_id']).distinct().zipWithIndex().collectAsMap()
        # Utility matrix of user
        utility_matrix = rdd.map(lambda line: (line['user_id'], business2index[line['business_id']])).groupByKey()
        # For local sensitive hashing number of rows and bands
        rows = 1
        bands = 30
        num_hash_func = rows * bands
        # Get parameters of hash function for later use
        H = random_hash_function(num_hash_func)
        # Get signature matrix for every user
        signature_matrix = utility_matrix.mapValues(get_signature_matrix).cache()
        # Dictionary of user_id and corresponding set of index
        index_matrix = defaultdict(set)
        for user in utility_matrix.mapValues(set).collect():
            index_matrix[user[0]] = user[1]
        # Locality sensitive hashing
        all_rdd = []
        for i in range(0, num_hash_func, rows):
            temp = signature_matrix.map(lambda line: (tuple(line[1][i:i+rows]), line[0])) \
                .groupByKey() \
                .filter(lambda line: len(line[1]) > 1) \
                .flatMap(lambda line: [tuple(sorted(i)) for i in it.combinations(line[1], 2)]).cache()
            all_rdd.append(temp)
        candidate = sc.union(all_rdd).distinct()
        # Get actual pairs which jaccard similarity at least 0.01 and co-rated user at least 3
        actual_pairs = candidate \
            .map(lambda line: (line, jaccard_similarity(index_matrix[line[0]], index_matrix[line[1]]))) \
            .filter(lambda line: line[1] >= 0.01 and
                                 len(index_matrix[line[0][0]].intersection(index_matrix[line[0][1]])) >= 3)\
            .map(lambda line: (line[0][0], line[0][1]))\
            .groupByKey()
        # Compute pearson correlations for valid pairs
        pearson_ans = actual_pairs.map(lambda line: pearson_correlation(line)) \
            .mapValues(lambda line: sorted(line.items(), key=lambda x: -x[1])) \
            .flatMap(lambda line: [(line[0], value) for value in line[1]]).collect()

    # Output pearson correlations for every pair of items
    with open(model_file, 'w') as f:
        for line in pearson_ans:
            if cf_type == 'item_based':
                ans = {
                    'b1': line[0],
                    'b2': line[1][0],
                    'sim': line[1][1]
                }
            elif cf_type == 'user_based':
                ans = {
                    'u1': line[0],
                    'u2': line[1][0],
                    'sim': line[1][1]
                }
            json.dump(ans, f)
            f.write('\n')
    f.close()

    # Finish time
    timer(start)

# spark-submit task3train.py data/train_review.json task3item.model item_based
# spark-submit task3train.py data/train_review.json task3user.model user_based
