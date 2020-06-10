# coding: utf-8
from datetime import datetime
import os
from pyspark import SparkContext, SparkConf
from itertools import count
import itertools as it
import json
import random
from collections import defaultdict
import csv
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


def load_file(file_path):
    """Loading file in chunk
    :param file_path: path to load file
    :return: current chunk of data
    """
    data = defaultdict(list)
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            data[line[0]] = [float(line[i]) for i in range(1, len(line))]
    f.close()
    return data


def get_sample_data(data, percent):
    """Randomly generate data from given data set by given percent
    :param data: data set you want to get sample from
    :param percent: percent of data you sample contains
    :return: sample data
    """
    num = int(len(data) * percent)
    keys = list(data.keys())
    sample_data = defaultdict(list)
    for _ in range(num):
        random_key = random.choice(keys)
        while random_key in sample_data.keys():
            random_key = random.choice(keys)
        sample_data[random_key] = data.pop(random_key)
    return sample_data, data


def euclidean_distance(p1, p2):
    """Calculate euclidean distance between two points (dimension unknown)
    :param p1: location of first point
    :param p2: location of second point
    :return: euclidean distance
    """
    return sum([(p1[i] - p2[i]) ** 2 for i in range(len(p1))]) ** .5


# def get_init_centroids(data, k):
#     dimension = len(list(data.values())[0])
#     max_range = [float('-inf') for _ in range(dimension)]
#     min_range = [float('inf') for _ in range(dimension)]
#     for value in data.values():
#         for i in range(dimension):
#             max_range[i] = max(max_range[i], value[i])
#             min_range[i] = min(min_range[i], value[i])
#     init_centroids = [[random.uniform(min_range[i], max_range[i]) for i in range(dimension)] for _ in range(k)]
#     centroids_dict = dict(zip(count(0), init_centroids))
#     return centroids_dict, dimension


def get_init_centroids(data, k):
    """Randomly generate number of k centroids from data
    :param data: data with name on key and location on value
    :param k: number of centroid
    :return: centroids, dimension
    """
    dimension = len(list(data.values())[0])
    centroids_dict = defaultdict(list)
    for i in range(k):
        centroids_dict[i] = data.get(random.choice(list(data.keys())))
    return centroids_dict, dimension


def k_mean(data, k, tolerance=10, n_iter=100):
    centroids_dict, dimension = get_init_centroids(data, k)
    track_num = {i: 0 for i in range(k)}
    N = defaultdict(int)
    sumi = defaultdict(list)
    sumsq = defaultdict(list)
    final_cluster = defaultdict(list)

    for i in range(n_iter):
        centroid2key = defaultdict(list)
        centroid2num = defaultdict(int)
        centroid2sum = defaultdict(list)
        centroid2sum_sq = defaultdict(list)
        for key, value in data.items():
            nearest_point = 0
            nearest_distance = float('inf')
            for num, c in centroids_dict.items():
                cur_distance = euclidean_distance(value, c)
                if cur_distance < nearest_distance:
                    nearest_distance = cur_distance
                    nearest_point = num
            centroid2num, centroid2sum, centroid2sum_sq = \
                update_num_sum_sumsq(centroid2num, centroid2sum, centroid2sum_sq, value, nearest_point)
            centroid2key[nearest_point].append(key)
        print(centroid2num)
        # Compute latest centroid for next iteration
        centroids_dict, no_use = get_centroid_sd(centroid2num, centroid2sum, centroid2sum_sq)
        # If number of changes for each iteration smaller than tolerance
        is_tolerance = [abs(value - track_num.get(key)) <= tolerance for key, value in centroid2num.items()]
        if all(is_tolerance) or i == n_iter - 1:
            # Keep track of number of nodes in each cluster and corresponding standard deviation
            N = centroid2num
            sumi = centroid2sum
            sumsq = centroid2sum_sq
            final_cluster = centroid2key
            break
        track_num = centroid2num
    # for i in range(k):
    #     if i not in centroids_dict.keys():
    #         centroids_dict[i] = sample.get(random.choice(list(sample.keys())))
    #         sd[i] = [0 for _ in range(dimension)]
    return N, sumi, sumsq, final_cluster, dimension


def mahalanobis_distance(p, c, sd):
    """Calculate Mahalanobis distance between each centroid and node
    :param p: location of node
    :param c: location of centroid
    :param sd: standard variation corresponding to the centroid
    :return: mahalanobis distance
    """
    return sum([((p[i] - c[i]) / sd[i]) ** 2 if sd[i] != 0 else (p[i] - c[i]) ** 2
                for i in range(len(p))]) ** .5


def update_num_sum_sumsq(num_dict, sum_dict, sd_dict, point, c):
    num_dict[c] += 1
    if sum_dict.get(c) is None:
        sum_dict[c] = point
        sd_dict[c] = [num ** 2 for num in point]
    else:
        sum_dict[c] = [sum([x, y]) for x, y in zip(sum_dict.get(c), point)]
        sd_dict[c] = [sum([x, y ** 2]) for x, y in zip(sd_dict.get(c), point)]
    return num_dict, sum_dict, sd_dict


def get_centroid_sd(num_dict, sum_dict, sd_dict):
    centroids_dict = defaultdict(list)
    standard_deviation = defaultdict(list)

    for key, value in sum_dict.items():
        num_nodes = num_dict.get(key)
        centroids_dict[key] = [s/num_nodes for s in value]
    for key, value in sd_dict.items():
        num_nodes = num_dict.get(key)
        c = centroids_dict.get(key)
        standard_deviation[key] = [((value[i]/num_nodes) - (c[i] ** 2)) ** 0.5 for i in range(len(value))]

    return centroids_dict, standard_deviation


def check_mahalanobis_distance(data, cent, sd, threshold, current_set):
    cur_num = defaultdict(int)
    cur_sum = defaultdict(list)
    cur_sum_sq = defaultdict(list)
    remaining_points = defaultdict(list)

    for key, point in data.items():
        cur_distance = float('inf')
        belong_to_centroid = 0
        for c_name in cent.keys():
            temp_dist = mahalanobis_distance(point, cent.get(c_name), sd.get(c_name))
            if temp_dist < cur_distance:
                cur_distance = temp_dist
                belong_to_centroid = c_name
        if cur_distance < threshold:
            cur_num, cur_sum, cur_sum_sq = update_num_sum_sumsq(cur_num, cur_sum, cur_sum_sq, point, belong_to_centroid)
            current_set[belong_to_centroid].append(key)
        else:
            remaining_points[key] = point
    return cur_num, cur_sum, cur_sum_sq, current_set, remaining_points


def update_sets(previous, current, d):
    # 0 is centroid to number of nodes
    # 1 is centroid to sum of ith dimension
    # 2 is centroid to sum of square of ith dimension
    for i in range(len(previous)):
        previous_data = previous[i]
        new_data = current[i]
        if i == 0:
            for key in list(new_data.keys()):
                previous_data[key] += new_data.get(key)
        else:
            for key in list(new_data.keys()):
                v1 = new_data.get(key)
                v2 = previous_data.get(key)
                temp_sum = [v1[j] + v2[j] for j in range(d)]
                previous_data[key] = temp_sum
        previous[i] = previous_data
    return previous


# def merge_two_set(set1, set2):
#     c1, s1 = get_centroid_sd(set1[0], set1[1], set1[2])
#     c2, s2 = get_centroid_sd(set2[0], set2[1], set2[2])
#     threshold = 3 * (len(list(c1.values())[0]) ** .5)
#     new_cluster = defaultdict(set)
#
#     for i, record1 in enumerate(c1.items()):
#         c2set = set()
#         for j, record2 in enumerate(c2.items()):
#             if i != j:
#                 cur_distance = mahalanobis_distance(record1[1], record2[1], s1.get(record1[0]))
#                 if cur_distance < threshold:
#                     c2set.add(record1[0])
#                     c2set.add(record2[0])
#         new_cluster[record1[0]] = c2set
#
#     all_key = set(new_cluster.keys())
#     mini_cluster = []
#     while all_key:
#         temp_key = all_key.pop()
#         temp = new_cluster.get(temp_key)
#         visited = set()
#         for i in all_key:
#             if len(temp & new_cluster.get(i)) > 0:
#                 temp = temp.union(new_cluster.get(i))
#             visited.add(i)
#         for v in visited:
#             all_key.remove(v)
#         mini_cluster.append(temp)
#
#     return None


def merge(set1, set2, threshold, cluster1, cluster2):
    c1, s1 = get_centroid_sd(set1[0], set1[1], set1[2])
    c2, s2 = get_centroid_sd(set2[0], set2[1], set2[2])

    for k1, v1 in list(c1.items()):
        nc = -1
        min_dist = float('inf')
        for k2, v2 in c2.items():
            cur_dist = mahalanobis_distance(v1, v2, s2.get(k2))
            if cur_dist < threshold and cur_dist < min_dist:
                min_dist = cur_dist
                nc = k2
        if nc != -1:
            set2[0][nc] = set2[0].get(nc) + set1[0].pop(k1)
            set2[1][nc] = [x+y for x, y in zip(set2[1].get(nc), set1[1].pop(k1))]
            set2[2][nc] = [x+y for x, y in zip(set2[2].get(nc), set1[2].pop(k1))]
            cluster2[nc].extend(cluster1.pop(k1))
    return set1, set2, cluster1, cluster2


def Bradley_Fayyad_Reina(data, ds_set, cs_set, rs_set, dimension, k, ds_c, cs_c):

    # If distance is smaller than threshold then nodes belong to clusters
    threshold = 3 * (dimension ** .5)

    # Get discard set's centroid and standard deviation
    ds_cent, ds_sd = get_centroid_sd(ds_set[0], ds_set[1], ds_set[2])
    # Update discard set and retained set
    ds_num, ds_sum, ds_sum_sq, ds_c, rp = check_mahalanobis_distance(data, ds_cent, ds_sd, threshold, ds_c)
    cur_ds = [ds_num, ds_sum, ds_sum_sq]
    cur_ds = update_sets(ds_set, cur_ds, dimension)

    # Get compression set's centroid and standard deviation
    cs_cent, cs_sd = get_centroid_sd(cs_set[0], cs_set[1], cs_set[2])
    # Update compression set and retained set
    cs_num, cs_sum, cs_sum_sq, cs_c, rp = check_mahalanobis_distance(rp, cs_cent, cs_sd, threshold, cs_c)
    cur_cs = [cs_num, cs_sum, cs_sum_sq]
    cur_cs = update_sets(cs_set, cur_cs, dimension)

    # K_mean on retained set and merge with cs set
    rs_set.update(rp)
    # if len(rs_set) > 3 * k:
    #     rs_num, rs_sum, rs_sum_sq, rs_cluster, rs_dim = k_mean(rs_set, 3 * k)

    return cur_ds, cur_cs, ds_c, cs_c, rs_set


if __name__ == '__main__':
    # Starting time
    start = timer()

    # Initial parameters
    input_path = sys.argv[1]
    n_cluster = int(sys.argv[2])
    out_file1 = sys.argv[3]
    out_file2 = sys.argv[4]
    # input_path = 'data/test2/'
    # n_cluster = 10
    # out_file1 = 'cluster_res'
    # out_file2 = 'intermediate_res'

    # BFR parameters
    discard_set = []
    compression_set = []
    retained_set = defaultdict(list)
    dim = 0
    threshold = 0
    intermediate_ans = []

    # Keep track of nodes' cluster
    ds2node = defaultdict(list)
    cs2node = defaultdict(list)

    # Iterate file path
    num_iter = 1
    for file_name in os.listdir(input_path):
        print('Processing file {} with time: {}s'.format(num_iter, datetime.now() - start))
        # Loading chunk
        data_points = load_file(input_path + '/' + file_name)
        # First round, compute sample k_mean and find centroids
        if num_iter == 1:
            # Get random sample from data set
            sample, remain = get_sample_data(data_points, 0.2)
            # For DS set
            ds_N, ds_sumi, ds_sumsq, ds_cluster, ds_dim = k_mean(sample, n_cluster)
            dim = ds_dim
            threshold = 3 * (dim ** .5)
            discard_set = [ds_N, ds_sumi, ds_sumsq]
            # For CS and RS
            cs_N, cs_sumi, cs_sumsq, cs_cluster, cs_dim = k_mean(remain, 3 * n_cluster, tolerance=25)
            for key, value in list(cs_N.items()):
                if value == 1:
                    retained_set[cs_cluster.pop(key)[0]] = cs_sumi.pop(key)
                    cs_N.pop(key)
                    cs_sumsq.pop(key)
            compression_set = [cs_N, cs_sumi, cs_sumsq]
            ds2node = ds_cluster
            cs2node = cs_cluster
            intermediate_ans.append([num_iter, n_cluster, sum(ds_N.values()), len(cs_N),
                                     sum(cs_N.values()), len(retained_set)])
        else:
            discard_set, compression_set, ds2node, cs2node, retained_set = \
                Bradley_Fayyad_Reina(data_points, discard_set, compression_set,
                                     retained_set, dim, n_cluster, ds2node, cs2node)
            intermediate_ans.append([num_iter, n_cluster, sum(discard_set[0].values()), len(compression_set[0]),
                                     sum(compression_set[0].values()), len(retained_set)])
        num_iter += 1

    # Last Round
    rs_N, rs_sumi, rs_sumsq, rs_cluster, rs_dim = k_mean(retained_set, 3 * n_cluster)
    rs_set, compression_set, rs2node, cs2node = \
        merge([rs_N, rs_sumi, rs_sumsq], compression_set, threshold, rs_cluster, cs2node)
    compression_set, discard_set, cs2node, ds2node \
        = merge(compression_set, discard_set, threshold, cs2node, ds2node)

    # Output file1
    ans = defaultdict(int)
    for k, v in ds2node.items():
        temp_dict = {node: k for node in v}
        ans.update(temp_dict)
    for k, v in cs2node.items():
        temp_dict = {node: -1 for node in v}
        ans.update(temp_dict)
    for k, v in rs2node.items():
        temp_dict = {node: -1 for node in v}
        ans.update(temp_dict)
    with open(out_file1, 'w') as f:
        json.dump(ans, f)
    f.close()

    # Output file2
    title = ['round_id', 'nof_cluster_discard', 'nof_point_discard',
             'nof_cluster_compression', 'nof_point_compression', 'nof_point_retained']
    with open(out_file2, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(title)
        for line in intermediate_ans:
            writer.writerow(line)
    f.close()

    # Ending time
    timer(start)