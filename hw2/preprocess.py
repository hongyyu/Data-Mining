# coding: utf-8
import sys, json, csv
from datetime import datetime
from pyspark import SparkContext, SparkConf


def timer(start_time=None):
    """Counting processing time
    :parameter start_time: if None start to computing time, if not None compute total processing time
    :type start_time: None, datetime
    :return start datetime or print out total processing time
    """
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        sec = (datetime.now() - start_time).total_seconds() % (3600 * 60)
        print('Total time it takes is %s seconds' % round(sec, 2))


if __name__ == '__main__':
    # Consuming time
    start = timer()

    review_path = 'data/review.json'
    business_path = 'data/business.json'

    # Create RDD
    conf = SparkConf().setAppName('task1').setMaster('local[*]')
    conf.set("spark.driver.memory", "15g")
    sc = SparkContext(conf=conf)

    # Load review and business to rdd
    review_rdd = sc.textFile(review_path).map(lambda x: json.loads(x)).persist()
    business_rdd = sc.textFile(business_path).map(lambda x: json.loads(x)).persist()

    # business_rdd with state only at 'NV'
    business_rdd = business_rdd.filter(lambda line: line['state'] == 'NV') \
        .map(lambda line: (line['business_id'], line['business_id']))

    # Only user_id and business_id from review_rdd with business_id only at 'NV'
    review_user_business = review_rdd \
        .map(lambda line: (line['business_id'], line['user_id']))
    inner_join = review_user_business.join(business_rdd) \
        .map(lambda line: list(line[1]))

    # Save as csv file
    file = inner_join.collect()
    with open('data/sampleData.csv', mode='w') as csv_file:
        header = ['user_id', 'business_id']
        writer = csv.writer(csv_file, delimiter=',')
        # Add header
        writer.writerow(header)
        # Write into file
        for row in file:
            writer.writerow(row)
    csv_file.close()

    # Consuming time
    timer(start)
