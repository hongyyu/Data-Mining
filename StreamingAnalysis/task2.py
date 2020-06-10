# coding: utf-8
import os
from datetime import datetime
from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
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


def Flajolet_Martin(line):
    # Get distinct city names in current streaming
    names = line.distinct().collect()
    # If not appear, add to set for actual count
    for n in names:
        if n not in name_set:
            name_set.add(n)

    # Number of trailing zeros for multi hash values
    trailing_zeros_list = []
    for h in hash_list:
        zeros = 0
        for n in name_set:
            num = int(binascii.hexlify(n.encode('utf8')), 16)
            binary_num = bin(((h[0] * num + h[1]) % h[2]) % h[3])[2:]
            zeros = max(zeros, len(binary_num) - len(binary_num.strip('0')))
        trailing_zeros_list.append(zeros)
    # Calculate average
    avg_with_interval = sorted([sum([2 ** r for r in trailing_zeros_list[i: i+10]])/10 for i in range(0, num_hash, 10)])
    mid = len(avg_with_interval)//2
    avg_count = avg_with_interval[mid] if len(avg_with_interval) % 2 != 0 \
        else (avg_with_interval[mid] + avg_with_interval[mid-1]) / 2
    # avg_count = sum([2 ** i for i in trailing_zeros_list]) / len(trailing_zeros_list)

    # Write new line for current stream
    with open(output_file_path, 'a') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow([str(datetime.now())[:19], len(name_set), avg_count])
    f.close()
    print((str(datetime.now())[:19], len(name_set), avg_count))



if __name__ == '__main__':
    # java -cp data/generate_stream.jar StreamSimulation data/business.json 9999 100
    # Start time
    start = timer()

    # Initial parameter
    name_set = set()
    batch_size = 5
    window_size = 30
    sliding_interval = 10
    num_hash = 50
    hash_list = random_hash_function(num_hash, 2 ** 9)
    output_file_path = 'task2ans'
    port = 9999

    # Input values
    # port = int(sys.argv[1])
    # output_file_path = sys.argv[2]

    # Create RDD and load data
    conf = SparkConf().setAppName('task2').setMaster('local[*]')
    conf.set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)
    sc.setLogLevel('ERROR')
    ssc = StreamingContext(sc, batch_size)
    ssc.checkpoint("./checkpoint")

    # Write column names
    col_names = ['Time', 'Ground Truth', 'Estimation']
    with open(output_file_path, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(col_names)
    f.close()

    # Create streaming
    stream = ssc.socketTextStream('localhost', port)\
        .map(lambda line: json.loads(line)['city'])\
        .window(window_size, sliding_interval)\
        .foreachRDD(Flajolet_Martin)

    # Start and terminate streaming
    ssc.start()
    ssc.awaitTermination()
    # ssc.awaitTerminationOrTimeout(50)

    # Finish time
    timer(start)
