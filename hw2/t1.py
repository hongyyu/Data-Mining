# coding: utf-8
import sys
from collections import defaultdict
from datetime import datetime
import itertools as it
from itertools import count
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
        print('Duration: %ss' % round(sec, 2))


def checkWholeDataset(key, support, baskets):
    """Second Pass of the whole data set to check candidate singleton, pairs, or triples which are frequent
    :parameter key: key got from first pass as candidate
    :type key: list of candidates key
    :parameter support: support of global data set
    :type support: int
    :parameter baskets: the whole data set
    :type baskets: list of list
    """
    frequent_items = defaultdict(int)
    for k in key:
        for line in baskets:
            if set(k).issubset(line):
                frequent_items[tuple(k)] += 1
    for key, value in list(frequent_items.items()):
        if value < support:
            frequent_items.pop(key)
    return frequent_items


def output_transform(file, ans):
    """Transform output file into correct format
    :parameter file: output of ans before transforming
    :type file: str
    :parameter ans: answer of singleton, pairs, triples, and etc
    :type ans: list of tuple
    :return file: correct format of singleton, pairs, triples, and etc
    :rtype file: str
    """
    for i in range(len(ans)):
        if i == len(ans) - 1:
            if len(ans[i]) == 1:
                file += '(\'' + ans[i][0] + '\')\n'
            else:
                file += str(ans[i]) + '\n'
        else:
            if len(ans[i]) == 1:
                file += '(\'' + ans[i][0] + '\'),'
            else:
                file += str(ans[i]) + ','
    return file


def apriori_next_candidate(pre_frequent, k):
    """Generate k+1 size of frequent item set candidate from previous level
    :parameter pre_frequent: previous level of frequent item set
    :type pre_frequent: list of tuple
    :parameter k: size of candidate
    :type k: int
    :return candidate_list: candidate list of size k
    :rtype candidate_list: list of set
    """
    num = len(pre_frequent)     # Length of previous frequent candidate
    candidate_list = []         # Storing candidate of size k

    for i in range(num):
        for j in range(i + 1, num):
            temp1 = list(pre_frequent[i])[:k - 2]
            temp2 = list(pre_frequent[j])[:k - 2]
            temp1.sort()
            temp2.sort()
            if temp1 == temp2:
                candidate_list.append(set(pre_frequent[i]) | set(pre_frequent[j]))
    return candidate_list


def pcy(partition, p, s, k, pre_frequent = None):
    partition = list(partition)         # List of partition
    ps = int(s/p)                       # Support for partition of data set
    cur_frequent = defaultdict(int)     # Initial an empty dict

    if pre_frequent:
        unique_id = set([id for pair in pre_frequent for id in pair])

    if k == 1:
        a = set([i for j in partition for i in j])
        m = dict(zip(count(0), a))
        temp = [0 for _ in range(len(m))]
        for line in partition:
            for k,v in m.items():
                if v in line:
                    temp[k] += 1
        for i in range(len(temp)):
            if temp[i] >= ps:
                cur_frequent[(m[i])] = 1
    elif k == 2:
        mapping = dict(zip(count(0), unique_id))
        reverse_mapping = dict(zip(unique_id, count(0)))
        matrix = [[0 for _ in range(len(mapping))] for _ in range(len(mapping))]
        for line in partition:
            combination = it.combinations(line, 2)
            for c in combination:
                if c[0] not in reverse_mapping.keys() or c[1] not in reverse_mapping.keys():
                    continue
                if reverse_mapping[c[0]] > reverse_mapping[c[1]]:
                    i, j = reverse_mapping[c[0]], reverse_mapping[c[1]]
                else:
                    i, j = reverse_mapping[c[1]], reverse_mapping[c[0]]
                matrix[i][j] += 1
        for i in range(len(mapping)):
            for j in range(i):
                if matrix[i][j] >= ps:
                    cur_frequent[(mapping[i], mapping[j])] = 1
    else:
        # hash_number = 10
        # bit_map = {i:0 for i in range(hash_number)}
        # count_map = defaultdict(list)
        # mapping = dict(zip(count(0), unique_id))
        # reverse_mapping = dict(zip(unique_id, count(0)))
        temp = defaultdict(int)

        can = apriori_next_candidate(pre_frequent, k)

        for line in partition:
            for c in can:
                if c.issubset(line):
                    temp[tuple(c)] += 1
        for key, value in temp.items():
            if value >= ps:
                cur_frequent[key] = 1
    return cur_frequent


if __name__ == '__main__':
    # Start counting time
    start = timer()

    # Initial parameters
    case = int(sys.argv[1])
    support = int(sys.argv[2])
    input_file = sys.argv[3]
    output_file = sys.argv[4]

    # Create RDD
    conf = SparkConf().setAppName('task1').setMaster('local[*]')
    conf.set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)
    rdd = sc.textFile(input_file) \
        .map(lambda line: line.split(','))

    # header to remove the first row
    header = rdd.first()

    # Case 1 count frequent item set of business_id
    if case == 1:
        # Filter header, group values with corresponding user_id, and keep only the unique
        # business_id for each user_id
        rdd = rdd.filter(lambda line: header != line) \
            .groupByKey() \
            .map(lambda line: list(set(line[1]))).persist()
        baskets = rdd.collect()
    # Case 2 count frequent item set of user_id
    elif case == 2:
        rdd = rdd.filter(lambda line: header != line) \
            .map(lambda line: [line[1], line[0]]) \
            .groupByKey() \
            .map(lambda line: list(set(line[1]))).persist()
        baskets = rdd.collect()

    p = rdd.getNumPartitions()  # Number of partitions
    candidates_list = []        # List of candidates list
    frequent_list = []          # List of frequent item set list
    k = 1                       # Initial size of k equal to 1

    while True:
        print('Processing size %d of frequent item set...' % k)
        # First pass to figure out frequent candidates
        if candidates_list:
            candidate = rdd.mapPartitions(lambda line: pcy(line, p, support, k, candidates_list[-1])).distinct()
        else:
            candidate = rdd.mapPartitions(lambda line: pcy(line, p, support, k)).distinct().map(lambda line: [line])
        c = candidate.collect()

        # If candidate is empty, there are no more candidate to compute
        if len(c) == 0:
            break

        # Second pass to compute actual frequent item set
        if frequent_list:
            frequent = candidate.mapPartitions(lambda line: checkWholeDataset(line, support, baskets))
        else:
            frequent = candidate.mapPartitions(lambda line: checkWholeDataset(line, support, baskets)).map(lambda line: list(line))
        f = frequent.collect()

        # Update size k by 1 and add candidate and frequent item set into list
        k += 1
        candidates_list.append(c)
        frequent_list.append(f)

    # Sorting final result
    print('Start to sorting...')
    for i in range(len(candidates_list)):
        candidates_list[i] = sorted([tuple(sorted(j)) for j in candidates_list[i]])
    for i in range(len(frequent_list)):
        frequent_list[i] = sorted([tuple(sorted(j)) for j in frequent_list[i]])

    # Write txt file
    candidate_str = ''
    frequent_str = ''
    for c in candidates_list:
        candidate_str = output_transform(candidate_str, c)
    for f in frequent_list:
        if f:
            frequent_str = output_transform(frequent_str, f)
    with open(output_file, 'w') as file:
        file.write('Candidates:\n')
        file.write(candidate_str)
        file.write('Frequent Itemsets:\n')
        file.write(frequent_str)
    file.close()

    # Consuming time
    timer(start)

# spark-submit task1_first_attempt.py 1 4 $ASNLIB/publicdata/small1.csv Hongyu_Li_task1.txt


