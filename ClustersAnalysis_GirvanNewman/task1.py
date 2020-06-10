# coding: utf-8
import os
from datetime import datetime
from graphframes import *
from pyspark import SparkContext, SparkConf
from pyspark import SQLContext
import pyspark.sql.functions as f
from pyspark.sql import Row
import sys
from itertools import chain
from pyspark.sql.functions import *
from functools import reduce


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
        print('Duration: {}s'.format(sec))


if __name__ == '__main__':
    # Start time
    start = timer()

    # Setting Environment
    os.environ["PYSPARK_SUBMIT_ARGS"] = ("--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 pyspark-shell")

    # Initial parameters
    # filter_threshold = int(sys.argv[1])
    # input_file_path = sys.argv[2]
    # community_output_file_path = sys.argv[3]
    filter_threshold = 7
    input_file_path = 'ub_sample_data.csv'
    community_output_file_path = 'task1_ans'

    # Setting sc and load file as SQL context
    conf = SparkConf().setAppName('task1').setMaster('local[*]')
    conf.set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)
    sc.setLogLevel('Error')
    sqlContext = SQLContext(sc)

    # Load file as spark dataframe
    df = sqlContext.read.format('csv')\
        .options(header='true', inferschema='true')\
        .load(input_file_path)

    # Group by user_id and make value as unique list
    df = df.groupBy('user_id')\
        .agg(f.collect_set('business_id').alias('business_id'))\
        .cache()
    # Give each user_id a index number
    # user2index = df.rdd.map(lambda line: (line.user_id, 1)) \
    #     .zipWithIndex() \
    #     .map(lambda line: (line[0][0], line[1])) \
    #     .collectAsMap()
    # mapping = create_map([f.lit(x) for x in chain(*user2index.items())])
    # # Replace user_id by its corresponding index number
    # df = df.select(mapping[df.user_id].alias('user_id'), 'business_id')

    # Copy df for cross join itself
    df_copy = df.selectExpr('user_id as user_id1', 'business_id as business_id1')

    # Cross join dataframe including unique pairs
    combinations = df.crossJoin(df_copy).filter(df.user_id < df_copy.user_id1)

    # Pairs from dataframe to rdd
    rdd = combinations.rdd
    # Row type for edges and vertices
    edge_type = Row('src', 'dst')
    vertex_type = Row('id')
    # Find edges with number of co-review at least threshold and save as Row
    one_direction = rdd\
        .filter(lambda line: len(set(line['business_id']) & set(line['business_id1'])) >= filter_threshold)\
        .map(lambda line: (line['user_id'], line['user_id1']))\
        .collect()
    # For undirected graph, get edges in both left and right
    to_right = sc.parallelize(one_direction)
    to_left = sc.parallelize(one_direction).map(lambda line: (line[1], line[0]))
    # Construct edges in undirected
    edges = sc.union([to_right, to_left])\
        .map(lambda line: edge_type(line[0], line[1])).cache()
    # Get unique user_id in the edges
    vertices = edges.flatMap(lambda line: list(line)).distinct().map(vertex_type).cache()

    # Transform edges and vertices rdd as spark dataframe
    edges_df = edges.toDF()
    vertices_df = vertices.toDF()

    # Load vertices and edges into graph
    g = GraphFrame(vertices_df, edges_df)
    result = g.labelPropagation(maxIter=5)

    # Get result and sort
    result_rdd = result.select("id", "label").rdd
    ans = result_rdd \
        .map(lambda line: (line.label, line.id)) \
        .groupByKey() \
        .map(lambda line: (len(line[1]), [str(i) for i in line[1]]))\
        .groupByKey()\
        .flatMap(lambda line: sorted([sorted(i) for i in line[1]], key=lambda x: x[0]))\
        .collect()

    # Output as txt file
    with open(community_output_file_path, 'w') as f:
        for line in ans:
            for i in range(len(line)):
                user = line[i]
                if i != len(line) - 1:
                    f.write('\'' + str(user) + '\', ')
                else:
                    f.write('\'' + str(user) + '\'')
            f.write('\n')
    f.close()

    # Finish time
    timer(start)

# spark-submit --packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 task1.py 7 ub_sample_data.csv task1_ans
# spark-submit --packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 task1.py 7 $ASNLIB/publicdata/ub_sample_data.csv task1_ans
