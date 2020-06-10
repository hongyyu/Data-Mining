# coding: utf-8
import os
from datetime import datetime
from pyspark import SparkContext, SparkConf
from collections import deque, defaultdict
from operator import add
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


def calculate_shortest_path(root):
    """Calculate all shortest path from root node to its children
    :param root: root node
    :type root: int
    :return: shortest paths count, nodes and corresponding level, tree from root node
    """
    # BFS from root node
    nodes_deque = deque([root])
    # Dictionary to keep number of shortest paths
    path_count, tree = defaultdict(int), defaultdict(list)
    path_count[root] += 1
    # Keep track of visited node and corresponding level
    visited2level, level, num_parents = defaultdict(int), 1, 1
    visited2level[root] = 0
    # Compute shortest path from root node
    while nodes_deque:
        # Compute number of parent of previous level
        temp_num = 0
        for _ in range(num_parents):
            # Pop node on the left
            parent = nodes_deque.popleft()
            # Children of parent node
            for child in mapping[parent]:
                # If visited, then continue
                # Else add it to queue and record its level
                if child not in visited2level.keys():
                    temp_num += 1
                    nodes_deque.append(child)
                    visited2level[child] = level
                # If the node the child of its parent, shortest path add one
                if visited2level[child] > visited2level[parent]:
                    path_count[child] += path_count[parent]
                    tree[parent].append(child)
    # Update number of previous level's parents and level
        num_parents = temp_num
        level += 1
    return path_count, visited2level, tree


def calculate_betweenness(root, tree, reversed_tree, edges, path_count):
    """Recursively compute betweenness of edges for a specific tree
    :param root: root node
    :param tree: tree being figured out from root node
    :param reversed_tree: tree from children to parent
    :param edges: keep track of edges and its betweenness
    :param path_count: shortest path which calcualted from previous step
    :return: value of betweenness of root node
    """
    # If the node is leaf, return 1
    if tree.get(root) is None:
        return 1
    # For nodes which are not leaf node
    else:
        # Sum of betweenness of all edges of current parent node
        sum_edges = 0
        # For each child of current parent node
        for node in tree[root]:
            # Recursively get previous node value
            # Node value = 1 + sum_edges
            node_val = calculate_betweenness(node, tree, reversed_tree, edges, path_count)
            # Number of shortest paths of current parent node
            root_num = path_count[root]
            # Sum of shortest paths of parents of current child
            denominator = sum([path_count[i] for i in reversed_tree[node]])
            # Compute edge betweenness
            result = node_val * (root_num / denominator)
            edges[tuple(sorted([root, node]))] = result
            sum_edges += result
        # Return node value which is 1 plus sum of edges' betweenness
        return 1 + sum_edges


def Girvan_Newman(line):
    """Using Girvan Newman algorithm to compute betweenness for every root
    :param line: root node
    :return: edges with corresponding betweenness
    """
    # Computer shortest paths from root node to all of its children
    path_count, visited2level, tree = calculate_shortest_path(line)
    # Construct reversed tree for computing sum of shortest paths of parents
    reversed_tree = defaultdict(list)
    for node in tree.items():
        for child in node[1]:
            reversed_tree[child].append(node[0])
    # Compute and construct edges
    edges = defaultdict(int)
    root_val = calculate_betweenness(line, tree, reversed_tree, edges, path_count)
    return edges


def calculate_modularity():
    """Calculate modularity and cluster for current graph
    :return: current modularity and cluster
    """
    # Compute current clusters
    clusters = []
    all_nodes = set(i for i in range(len(id2index)))
    while all_nodes:
        temp_set = set()
        next_node = 0
        for i in all_nodes:
            next_node = i
            break
        a, b, tree = calculate_shortest_path(next_node)
        if len(tree) != 0:
            for k, v in tree.items():
                temp_set.add(k)
                for i in v:
                    temp_set.add(i)
        else:
            temp_set.add(next_node)
        clusters.append(temp_set)
        all_nodes -= temp_set

    # Compute modularity
    modularity = 0
    for c in clusters:
        for i in c:
            for j in c:
                if i != j:
                    modularity += A[i][j] - ((len(mapping[i]) * len(mapping[j])) / (2 * m))
    modularity = (1 / (2 * m)) * modularity
    return modularity, clusters


if __name__ == '__main__':
    # Start time
    start = timer()

    # Initial parameters
    # filter_threshold = int(sys.argv[1])
    # input_file_path = sys.argv[2]
    # betweenness_output_file_path = sys.argv[3]
    # community_output_file_path = sys.argv[4]
    filter_threshold = 7
    input_file_path = 'ub_sample_data.csv'
    betweenness_output_file_path = 'task2_betweenness_ans'
    community_output_file_path = 'task2_community_ans'

    # Create RDD and load data
    conf = SparkConf().setAppName('task1').setMaster('local[*]')
    conf.set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)
    rdd = sc.textFile(input_file_path)

    # Filter first line and split string
    first_line = rdd.first()
    rdd = rdd.filter(lambda line: line != first_line)\
        .map(lambda line: tuple(line.split(',')))\
        .groupByKey()\
        .sortByKey().cache()

    # Get valid pairs and filter by number of co-review which is at least filter_threshold
    combinations = rdd.cartesian(rdd)\
        .filter(lambda line: line[0][0] < line[1][0])\
        .filter(lambda line: len(set(line[0][1]) & set(line[1][1])) >= filter_threshold)\
        .map(lambda line: (line[0][0], line[1][0])).cache()
    # Unique user_id to corresponding index
    id2index = combinations.flatMap(lambda line: [(i, 1) for i in line])\
        .distinct()\
        .sortByKey()\
        .zipWithIndex()\
        .map(lambda line: (line[0][0], line[1]))\
        .collectAsMap()
    # Reversed previous dictionary
    index2id = {value: key for key, value in id2index.items()}
    # Undirected graph including nodes and corresponding children
    mapping = combinations\
        .flatMap(lambda line: [(id2index[line[0]], id2index[line[1]]), (id2index[line[1]], id2index[line[0]])])\
        .groupByKey() \
        .sortByKey().mapValues(list) \
        .collectAsMap()
    # Compute betweenness by Girvan Newman
    unique_id = combinations.flatMap(lambda line: [id2index[i] for i in line])\
        .distinct().cache()
    betweenness = unique_id.flatMap(lambda line: Girvan_Newman(line).items())\
        .reduceByKey(add)\
        .map(lambda line: ((index2id[line[0][0]], index2id[line[0][1]]), line[1]/2))
    # Sort the betweenness answer
    betweenness_ans = sorted([(sorted(i[0]), i[1]) for i in betweenness.collect()], key=lambda x: (-x[1], x[0][0]))

    # For detecting communities
    # Adjacency matrix of the graph
    A = [[0 for _ in range(len(id2index))] for _ in range(len(id2index))]
    # All edges in the graph
    edges = set(combinations.map(lambda line: (id2index[line[0]], id2index[line[1]])).collect())
    # If there is a edge then 1, else 0
    for pair in edges:
        A[pair[0]][pair[1]] = 1
        A[pair[1]][pair[0]] = 1
    # Total number of edges one direction only
    m = len(edges)
    # Keep track of maximum modularity and corresponding cluster
    max_modularity = float('-inf')
    proper_cluster = []
    # Delete one edge and recompute betweenness until finding peak of modularity
    while edges:
        e = unique_id.flatMap(lambda line: Girvan_Newman(line).items()) \
            .reduceByKey(add) \
            .sortBy(lambda line: -line[1]) \
            .mapValues(lambda line: line/2).first()
        edges.remove(e[0])
        mapping[e[0][0]].remove(e[0][1])
        mapping[e[0][1]].remove(e[0][0])
        cur_modularity, cur_cluster = calculate_modularity()
        if cur_modularity > max_modularity:
            max_modularity = cur_modularity
            proper_cluster = cur_cluster
        else:
            break
        print(cur_modularity)

    # Sort and transform final clusters
    community_ans = defaultdict(int)
    for line in proper_cluster:
        community_ans[tuple(sorted([index2id[index] for index in line]))] = len(line)
    community_ans = sorted(community_ans.items(), key=lambda x: x[1])

    # Output betweenness text
    with open(betweenness_output_file_path, 'w') as f:
        for line in betweenness_ans:
            f.write('(\'' + line[0][0] + '\', \'' + line[0][1] + '\'), ' + str(line[1]) + '\n')
    f.close()
    # Output detected communities text
    with open(community_output_file_path, 'w') as f:
        for line in community_ans:
            for i in range(len(line[0])):
                user = line[0][i]
                if i != len(line[0]) - 1:
                    f.write('\'' + user + '\', ')
                else:
                    f.write('\'' + user + '\'')
            f.write('\n')
    f.close()

    # Finish time
    timer(start)
