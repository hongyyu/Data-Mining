# coding: utf-8
from datetime import datetime
from pyspark import SparkContext, SparkConf
import math
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


# def dot(A, B):
#     """Dot product between business profile and user profile
#     :param A: business profile
#     :param B: user profile
#     :return: dot product
#     """
#     return sum(ai * bi for ai, bi in zip(A, B))


def cosine_similarity(user, business):
    """Cosine similarity between business profile and user profile
    :param user: user profile
    :param business: business profile
    :return: cosine similarity
    """
    return len(user.intersection(business)) / (math.sqrt(len(user)) * math.sqrt(len(business)))


def check_similarity(pair):
    """Compute cosine similarity between user and business
    :param pair: valid pair found on training step
    :return: similarity
    """
    user, business = pair['user_id'], pair['business_id']
    similarity = -1
    try:
        user_text = eval(user_profile[user])
        business_text = eval(business_profile[business])
        similarity = cosine_similarity(user_text, business_text)
    except:
        pass
    return similarity if similarity != -1 else 0


if __name__ == '__main__':
    # Start time
    start = timer()

    # Initial parameters
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    output_file = sys.argv[3]

    # Load profiles
    with open(model_file, 'r') as f:
        business_profile = json.loads(f.readline())
        user_profile = json.loads(f.read())
    f.close()

    # Create RDD for business_profile and user_profile
    conf = SparkConf().setAppName('task2predict').setMaster('local[*]')
    conf.set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)
    test_rdd = sc.textFile(test_file).map(lambda x: json.loads(x))

    # Combine business_profile and user_profile to calculate cosine similarity
    ans = test_rdd.map(lambda line: (line['user_id'], line['business_id'], check_similarity(line)))\
        .filter(lambda line: line[2] >= 0.01).collect()

    # Output answer
    with open(output_file, 'w') as f:
        for line in ans:
            res = {
                'user_id': line[0],
                'business_id': line[1],
                'sim': line[2]
            }
            json.dump(res, f)
            f.write('\n')
    f.close()

    # Finish time
    timer(start)

# spark-submit hw3/task2predict.py hw3/data/test_review.json hw3/task2.model hw3/task2.predict
