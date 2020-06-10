# coding: utf-8
from datetime import datetime
from pyspark import SparkContext, SparkConf
from itertools import count
import itertools as it
import json
import random
from collections import defaultdict
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


def generate_utility_matrix(rdd):
    """ Generate utility matrix and get length of unique business_id and uer_id
    :parameter rdd: rdd of train_review.json file
    :type rdd: pyspark rdd
    :returns matrix, user2index
    """
    # Get unique user_id and business_id
    unique_user_id = rdd.map(lambda line: line['user_id']).distinct().collect()
    user2index = dict(zip(unique_user_id, count(0)))
    matrix = rdd.map(lambda line: (line['business_id'], user2index[line['user_id']])).groupByKey()

    return matrix, user2index


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


if __name__ == '__main__':
    # Starting time
    start = timer()

    # Initial parameters
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    # input_file = 'data/train_review.json'
    # output_file = 'task1_ans'

    # Create RDD
    conf = SparkConf().setAppName('task1').setMaster('local[*]')
    conf.set("spark.driver.memory", "10g")
    sc = SparkContext(conf=conf)
    rdd = sc.textFile(input_file).map(lambda x: json.loads(x)).persist()

    # Get utility matrix and number of unique row which is used for mini-hash function
    utility_matrix, user2index = generate_utility_matrix(rdd)
    # Randomly generate parameters for hash functions
    rows = 1
    bands = 30
    num_hash_func = rows * bands
    H = random_hash_function(num_hash_func)
    # Construct signature matrix
    signature_matrix = utility_matrix.mapValues(get_signature_matrix).cache()
    # Get index number if there exists a rate
    index_matrix = defaultdict(set)
    for business in utility_matrix.mapValues(set).collect():
        index_matrix[business[0]] = business[1]

    # Locality sensitive hashing
    all_rdd = []
    for i in range(0, num_hash_func, rows):
        temp = signature_matrix.map(lambda line: (tuple(line[1][i:i+rows]), line[0])) \
            .groupByKey()\
            .filter(lambda line: len(line[1]) > 1) \
            .flatMap(lambda line: [tuple(sorted(i)) for i in it.combinations(line[1], 2)]).cache()
        all_rdd.append(temp)
    candidate = sc.union(all_rdd).distinct()
    # Get actual pairs which jaccard similarity greater than 0.05
    actual_pairs = candidate\
        .map(lambda line: (line, jaccard_similarity(index_matrix[line[0]], index_matrix[line[1]])))\
        .filter(lambda line: line[1] >= 0.05)\
        .collect()

    # Output as json
    with open(output_file, 'w') as f:
        for row in actual_pairs:
            ans = {
                'b1': row[0][0],
                'b2': row[0][1],
                'sim': row[1]
            }
            json.dump(ans, f)
            f.write('\n')
    f.close()
    # Ending time
    timer(start)

# spark-submit hw3/task1.py hw3/data/train_review.json hw3/task1_ans