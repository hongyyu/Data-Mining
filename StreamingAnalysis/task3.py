# coding: utf-8
import os
from datetime import datetime
from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from collections import Counter
import tweepy
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


# Twitter streaming listener
class MyStreamListener(StreamListener):

    def __init__(self, output_path):
        self.output_path = output_path
        self.tag_list = []
        self.num = 0
    
    def on_data(self, raw_data):
        try:
            data = json.loads(raw_data)
            entities = data['entities']['hashtags']
            if entities:
                self.num += 1
                tag = entities[0]['text']
                if len(self.tag_list) < 100:
                    self.tag_list.append(tag)
                elif tag.isalpha():
                    prob2keep = 100 / self.num
                    random_prob = random.uniform(0, 1)
                    if prob2keep > random_prob:
                        random_index = random.randint(0, 99)
                        self.tag_list.pop(random_index)
                        self.tag_list.append(tag)
                # Sort by frequency and tag
                frequency = Counter(self.tag_list)
                top3num = sorted(set(frequency.values()), key=lambda x: -x)[:3]
                top3tag = [(key, value) for key, value in frequency.items() if value in top3num]
                top3tag = sorted(top3tag, key=lambda x: (-x[1], x[0]))
                # Output file
                with open(self.output_path, 'a') as f:
                    f.write('The number of tweets with tags from the beginning: {}\n'.format(self.num))
                    for tag in top3tag:
                        f.write('{} : {}\n'.format(tag[0], tag[1]))
                    f.write('\n')
                f.close()
            return True

        except BaseException as e:
            print('Error: {}'.format(e))
            return True


if __name__ == '__main__':
    # Start time
    start = timer()

    # Initial parameters
    # port = int(sys.argv[1])
    # output_file_path = sys.argv[2]
    output_file_path = 'task3ans'

    # Create empty file
    open(output_file_path, 'w').close()

    # Create RDD and load data
    conf = SparkConf().setAppName('task3').setMaster('local[*]')
    conf.set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)
    sc.setLogLevel('ERROR')

    # Twitter token and key
    ACCESS_TOKEN = ''
    ACCESS_TOKEN_SECRET = ''
    CONSUMER_KEY = ''
    CONSUMER_SECRET = ''

    # Setting auth
    auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

    # Initialize twitter stream
    myStreamListener = MyStreamListener(output_file_path)
    myStream = Stream(auth=auth, listener=myStreamListener)
    myStream.filter(languages=['en'])
    myStream.sample()

    # Finish time
    timer(start)
