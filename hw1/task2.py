# coding: utf-8
import json, sys
from datetime import datetime
from pyspark import SparkContext, SparkConf
from collections import defaultdict

# input parameters
# review_file = '/Users/hongyuli/Desktop/USC/2020_Spring/Foundations_and_Applications_' \
#               'of_Data_Mining/inf553-python3.6/hw1_data/review.json'
# business_file = '/Users/hongyuli/Desktop/USC/2020_Spring/Foundations_and_Applications_' \
#                 'of_Data_Mining/inf553-python3.6/hw1_data/business.json'
review_file = sys.argv[1]       # Input file for review dataset
business_file = sys.argv[2]     # Input file for business dataset
output_file = sys.argv[3]       # Output file directory to save
if_spark = sys.argv[4]          # 'spark' or 'no_spark'
n = int(sys.argv[5])            # top n categories with highest avg stars


# define function for computing time
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        sec = (datetime.now() - start_time).total_seconds() % (3600 * 60)
        print('Total time it takes is %s seconds' % round(sec, 2))


res = []
if if_spark == 'spark':
    # Create RDD for review and business
    conf = SparkConf().setAppName('task2').setMaster('local[*]')
    conf.set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)

    # Import json files as rdd, review_rdd only keep business_id and stars feature
    spark_start_time = timer()    # time start with spark
    review_rdd = sc.textFile(review_file) \
                    .map(lambda x: json.loads(x)) \
                    .map(lambda x: (x['business_id'], x['stars']))
    business_rdd = sc.textFile(business_file) \
                    .map(lambda x: json.loads(x)) \
                    .filter((lambda x: x['categories'] is not None))

    # Extract categories from business_rdd
    categories_rdd = business_rdd \
                    .map(lambda x: (x['business_id'], x['categories'].split(','))) \
                    .flatMap(lambda x: [(x[0], c.strip()) for c in x[1]])

    # Combine review_rdd and categories_rdd with corresponding values, and count average
    category_star_rdd = review_rdd \
                        .join(categories_rdd) \
                        .map(lambda x: (x[1][1], x[1][0])) \
                        .groupByKey() \
                        .mapValues(lambda x: sum(x) / len(x)) \
                        .takeOrdered(n, key=lambda x: (-x[1], x[0]))

    # print final result about using spark to get top n categories with highest avg stars
    print(category_star_rdd)
    # Count time
    timer(spark_start_time)

    # Setting res
    for category, star in category_star_rdd:
        res.append([category, star])
elif if_spark == 'no_spark':
    # Start Time
    no_spark_time = timer()

    # Import review and business json file as dictionary
    review_data = [json.loads(line) for line in open(review_file)]
    business_data = [json.loads(line) for line in open(business_file)]

    # Split categories in business data
    for record in business_data:
        if record['categories']:
            record['categories'] = [c.strip() for c in record['categories'].split(',')]

    # Save all stars in a list with corresponding business_id
    review_star_dict = defaultdict(list)
    for record in review_data:
        review_star_dict[record['business_id']].append(record['stars'])


    # Save (key, value) pair (business_id, categories) in a whole dictionary
    business_category_dict = {}
    for record in business_data:
        if record['categories']:
            business_category_dict[record['business_id']] = record['categories']

    # Store category with corresponding stars based on the same business_id
    sum_dict, count_dict = {}, {}
    for key in review_star_dict.keys():                     # for every key in review data
        if key in business_category_dict.keys():
            for category in business_category_dict[key]:    # for category in business data
                for star in review_star_dict[key]:          # for every star in review data
                    if category not in sum_dict.keys():
                        sum_dict[category] = star
                        count_dict[category] = 1
                    else:
                        sum_dict[category] += star
                        count_dict[category] += 1

    # Calculate avg and sort in descending order
    ans = []
    for key in sum_dict.keys():
        avg = sum_dict[key] / count_dict[key]
        ans.append([key, avg])
    ans.sort(key=lambda x: (-x[1], x[0]))
    res = ans[:n]

    # Count Time
    timer(no_spark_time)

# Store in dictionary
ans_dict = {'result': res}
# Output as json file
with open(output_file, 'w') as f:
    json.dump(ans_dict, f)

# Command Line spark-submit
# spark-submit hw1/task2.py hw1_data/review.json hw1_data/business.json task2_spark_ans.json spark 10
# spark-submit hw1/task2.py hw1_data/review.json hw1_data/business.json task2_no_spark_ans.json no_spark 10

# Vocareum
# spark-submit task2.py $ASNLIB/publicdata/review.json $ASNLIB/publicdata/business.json task2_spark_ans spark 20
# spark-submit task2.py $ASNLIB/publicdata/review.json $ASNLIB/publicdata/business.json task2_no_spark_ans no_spark 20


