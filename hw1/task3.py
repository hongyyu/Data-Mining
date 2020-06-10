# coding: utf-8
import json, sys
from pyspark import SparkContext, SparkConf
from operator import add
from datetime import datetime

# Define parameters for using command line
input_file = sys.argv[1]            # Input review.json
output_file = sys.argv[2]           # Output path, saved as task3_ans.json
partition_type = sys.argv[3]        # Either 'default' or 'customized'
n_partitions = int(sys.argv[4])     # Number of partition to use if 'customized'
n = int(sys.argv[5])                # Number of reviews greater than n will be kept

# Number of partitions either default or customized and result list
num_partition = None
res = []


# define function for computing time
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        sec = (datetime.now() - start_time).total_seconds() % (3600 * 60)
        print('Total time it takes is %s seconds' % round(sec, 2))


# Create a function to compute num of items in each partition
def get_num_item(rdd):
    partition_list = rdd.glom().map(len).collect()
    return partition_list


# Define simple hash function
def hash_function(line):
    return hash(line)


# Create RDD for review
conf = SparkConf().setAppName('task2').setMaster('local[*]')
conf.set("spark.driver.memory", "4g")
sc = SparkContext(conf=conf)

# Default or Customized
if partition_type == 'default':
    default_time = timer()
    # Import review.json as review rdd with default minPartitions
    review_rdd = sc.textFile(input_file) \
        .map(lambda x: json.loads(x))

    # Compute business that have more n reviews in the review file
    n_reviews_rdd = review_rdd \
        .map(lambda x: (x['business_id'], 1)) \
        .reduceByKey(lambda x,y: x + y) \
        .filter(lambda x: x[1] > n)

elif partition_type == 'customized':
    customized_time = timer()
    # Import review.json as review rdd with customized minPartitions
    review_rdd = sc.textFile(input_file) \
                    .map(lambda x: json.loads(x))

    # Compute business that have more n reviews in the review file
    n_reviews_rdd = review_rdd \
                    .map(lambda x: (x['business_id'], 1)) \
                    .partitionBy(n_partitions, hash_function) \
                    .reduceByKey(lambda x,y: x + y) \
                    .filter(lambda x: x[1] > n)

# Get num items in each partition and number of partition
partition_list = get_num_item(n_reviews_rdd)
num_partition = n_reviews_rdd.getNumPartitions()

# Count time
if partition_type == 'default':
    timer(default_time)
elif partition_type == 'customized':
    timer(customized_time)

# Setting res
for business, count in n_reviews_rdd.collect():
    res.append([business, count])

# Store answer to the dictionary
ans_dict = {'n_partitions': num_partition,
            'n_items': partition_list,
            'result': res}

# Output as json file
with open(output_file, 'w') as f:
    json.dump(ans_dict, f)

# Command Line spark-submit
# spark-submit --driver-memory 4G hw1/task3.py hw1_data/review.json task3_default_ans default 20 50
# spark-submit --driver-memory 4G hw1/task3.py hw1_data/review.json task3_customized_ans customized 20 50

# Vocareum
# spark-submit task3.py $ASNLIB/publicdata/review.json task3_default_ans default 20 50
# spark-submit task3.py $ASNLIB/publicdata/review.json task3_customized_ans customized 20 50