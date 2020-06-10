# coding: utf-8
import os
from datetime import datetime
from pyspark import SparkContext, SparkConf
import random
import csv
import json
import binascii
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


def random_hash_function(hash_size, denominator):
    """Randomly generate parameters for hash function
    :parameter hash_size:
    :parameter denominator:
    :returns
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

    return [(num_a[i], num_b[i], random.choice(primes_list), denominator) for i in range(hash_size)]


def splitByOddEven(line, k):
    odd_str = ''.join([line[i] for i in range(len(line)) if i % 2 != 0])
    even_str = ''.join([line[i] for i in range(len(line)) if i % 2 == 0])
    return odd_str, even_str


def hashString(line):
    res = []
    for a, b, p, m in hash_parameters:
        hash_num = ((a * line + b) % p) % m
        res.append(hash_num)
    return res


def checkSecondPass(line):
    try:
        num = int(binascii.hexlify(line.encode('utf8')), 16)
    except ValueError:
        num = -1

    if num == -1:
        return 0
    else:
        index_list = hashString(num)
        hashed_list = [bit_array[i] for i in index_list]
    return 1 if all(hashed_list) else 0


if __name__ == '__main__':
    # Starting time
    start = timer()

    # Initial Parameters
    # first_json_path = sys.argv[1]
    # second_json_path = sys.argv[2]
    # output_file = sys.argv[3]
    first_json_path = 'data/business_first.json'
    second_json_path = 'data/business_second.json'
    output_file = 'task1ans'

    # Create RDD and load data
    conf = SparkConf().setAppName('task1').setMaster('local[*]')
    conf.set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)
    first_rdd = sc.textFile(first_json_path)\
        .map(lambda line: json.loads(line).get('city'))\
        .filter(lambda x: x is not None and x != '')
    second_rdd = sc.textFile(second_json_path).map(lambda line: json.loads(line).get('city'))

    # Initial bit array and hash parameters
    len_filter = 10000
    num_hash = 5
    bit_array = [0 for _ in range(len_filter)]
    hash_parameters = random_hash_function(num_hash, len_filter)

    # Create filter for first data set
    distinct_index = first_rdd \
        .map(lambda x: int(binascii.hexlify(x.encode('utf8')), 16))\
        .flatMap(lambda num: hashString(num)).distinct()
    for index in distinct_index.collect():
        bit_array[index] = 1

    # Check if city in second data set already appeared in the first data set
    ans = second_rdd.map(lambda line: checkSecondPass(line)).collect()

    # Output file
    with open(output_file, 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerow(ans)
    f.close()

    # Finishing time
    timer(start)