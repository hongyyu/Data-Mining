# coding: utf-8
from datetime import datetime
from pyspark import SparkContext, SparkConf
from itertools import count
import itertools as it
import json
import random
import math
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
    :returns matrix, user_length, business_length
    """
    # Get unique user_id and business_id
    unique_user_id = rdd.map(lambda line: line['user_id']).distinct().collect()
    unique_business_id = rdd.map(lambda line: line['business_id']).distinct().collect()

    # Provide each usr_id and business a unique index for storing in the utility matrix
    # Column: user_id; Row: business_id
    user_id_index = dict(zip(unique_user_id, count(0)))
    business_id_index = dict(zip(unique_business_id, count(0)))
    index2business = dict(zip(count(0), unique_business_id))

    # Constructing utility matrix with only 0-1
    matrix = [[0 for _ in range(len(unique_user_id))] for _ in range(len(unique_business_id))]
    pairs = rdd.map(lambda line: [business_id_index[line['business_id']], user_id_index[line['user_id']]]).collect()
    # If there exists pair of (business, user), set the position to 1
    for p in pairs:
        matrix[p[0]][p[1]] = 1

    return matrix, len(unique_user_id), len(unique_business_id), index2business


def random_hash_function(bins: int, num_business: int):
    """Randomly generate parameters for hash function and generate a empty signature matrix
    :parameter bins: length of unique user_id which is also total number of records
    :parameter num_business: length of unique business_id
    :returns parameters, signature_matrix
    """
    hash_size = 1000                                                     # Number of hash functions
    num_a = [random.randrange(bins) for _ in range(hash_size)]   # Parameter a for hash function
    num_b = [random.randrange(bins) for _ in range(hash_size)]   # Parameter b for hash function
    bins = math.ceil(math.sqrt(bins))                                   # Mod by bins

    # Combined parameters for each hash function
    parameters = [(num_a[i], num_b[i], 40) for i in range(hash_size)]
    # Generate empty signature matrix with inf number
    empty_h = [[float('inf') for _ in range(len(parameters))] for _ in range(num_business)]

    return parameters, empty_h


def hash_function(num_row, a, b, m):
    """Hash function for calculating signature matrix
    :return hash number for each row number
    """
    return (a * num_row + b) % m


def construct_signature_matrix(utility_matrix, H, signature_matrix):
    """Constructing signature matrix with hashed value for each row
    :param utility_matrix: matrix with only 0-1 if there exist a rate
    :param H: parameters for each hash functions
    :param signature_matrix: empty signature matrix with size(length_business, num_hash)
    :return: signature_matrix
    """
    for col in range(len(utility_matrix[0])):
        hash_list = [hash_function(col, *H[index]) for index in range(len(H))]
        for row in range(len(utility_matrix)):
            if utility_matrix[row][col] == 1:
                for index in range(len(hash_list)):
                    if hash_list[index] < signature_matrix[row][index]:
                        signature_matrix[row][index] = hash_list[index]
    return signature_matrix


def locality_sensitive_hashing(min_hash, b):
    """Locality sensitive hash to find identical pair
    :param min_hash: signature matrix
    :param b: number of bands to split signature matrix
    :return: candidate pairs for each band
    """
    all_rdd = []                                # Empty list to store all rdd
    num_row = int(len(min_hash[0]) / b)         # Number of row for each band
    # For each band, we find identical pairs
    for i in range(0, len(min_hash[0]), num_row):
        if i + num_row >= len(min_hash[0]):
            sub_matrix = [tuple(row[i:]) for row in min_hash]
        else:
            sub_matrix = [tuple(row[i:i+num_row]) for row in min_hash] # .coalesce(1) \
        sub_rdd = sc.parallelize(sub_matrix)\
            .zipWithIndex()\
            .groupByKey() \
            .flatMap(lambda line: [c for c in it.combinations(line[1], 2)])
        all_rdd.append(sub_rdd)
    # Combine all rdd into one
    candidates = sc.union(all_rdd).cache()

    return candidates


def jaccard_similarity(b1, b2, num = None, is_list = False):
    """Jaccard similarity between two set b1 and b2
    :param b1: set of index if there is a rate for business b1
    :param b2: set of index if there is a rate for business b2
    :return: jaccard similarity
    """
    if is_list:
        similarity = len([0 for i, j in zip(b1, b2) if i == j])/num
    else:
        similarity = len(b1.intersection(b2))/len(b1.union(b2))
    return similarity


if __name__ == '__main__':
    # Starting time
    start = timer()

    # Initial parameters
    input_file = 'data/train_review.json'
    output_file = 'task1_ans'

    # Create RDD
    conf = SparkConf().setAppName('task1').setMaster('local[*]')
    conf.set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)
    rdd = sc.textFile(input_file).map(lambda x: json.loads(x)).persist()

    # Get utility matrix and number of unique row which is used for mini-hash function
    utility_matrix, user_length, business_length, index2business = generate_utility_matrix(rdd)
    # Randomly generate parameters for hash functions
    H, signature_matrix = random_hash_function(user_length, business_length)
    # Construct signature matrix
    signature_matrix = construct_signature_matrix(utility_matrix, H, signature_matrix)
    # Locality sensitive hashing
    num_bands = 20
    candidate_pairs = locality_sensitive_hashing(signature_matrix, num_bands)
    # Get index number if there exists a rate
    index_matrix = sc.parallelize(utility_matrix)\
        .map(lambda line: set([i for i in range(len(line)) if line[i]]))\
        .collect()
    # Verify actual similar pairs
    actual_pairs = candidate_pairs.distinct()\
        .map(lambda line: (line, jaccard_similarity(index_matrix[line[0]], index_matrix[line[1]])))\
        .filter(lambda line: line[1] >= 0.05)\
        .collect()
    # Output as json
    with open(output_file, 'w') as f:
        for row in actual_pairs:
            ans = {
                'b1': index2business[row[0][0]],
                'b2': index2business[row[0][1]],
                'sim': row[1]
            }
            json.dump(ans, f)
            f.write('\n')
    f.close()
    # Ending time
    timer(start)