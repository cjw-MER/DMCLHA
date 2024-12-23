from cProfile import label
from tqdm import tqdm
import numpy as np, networkx as nx, time
import pandas as pd

def conceptnet_graph(filename):
    """
    Reads conceptnet english graph from text file which has the following format:
    head_concept \t relation_type \t tail_concept
    
    The graph and reverse graph are stored in networkx directed graph format.
    Concept and relation mappings from strings to ids are also created.
    
    Returns graph, reverse graph, concept mapping, relation mapping
    """
    array = []#创建一个空的列表 array 用于存储读取的数据行
    concept_set, relation_set, concept_map, relation_map = set(), set(), {}, {}#创建空的集合、空的字典，用于存储图中的概念和关系信息

    f = open(filename, 'r')#打开指定的文件 filename 进行读取操作
    for line in f:
        array.append(line[:-1].split('\t'))#遍历文件的每一行，将每行数据按制表符 \t 进行分割，并将分割结果添加到 array 列表中
        concept_set.add(array[-1][0]); concept_set.add(array[-1][1])
        relation_set.add(array[-1][2])
    f.close()
    
    for item in list(relation_set):
        relation_map[item] = len(relation_map)
    
    for item in list(concept_set):
        concept_map[item] = len(concept_map)

    G = nx.DiGraph()#创建了两个空的有向图对象
    G_reverse = nx.DiGraph()

    time.sleep(0.1)#暂停执行 0.1 秒

    for x in tqdm(array):#构建图
        G.add_edges_from([(x[0], x[1])], label=x[2])#将 (x[0], x[1]) 作为一条边添加到 G 中，边的标签为 x[2]
        G_reverse.add_edges_from([(x[1], x[0])], label=x[2])
        
    return G, G_reverse, concept_map, relation_map
#构造原始图
def construct_graph(dir="labeled_data.csv"):    
    data = pd.read_csv(dir, index_col = False)
    parent = data['author_parent']#从读取的数据中获取父节点（author_parent）、子节点（author_child）和标签（label）的列
    child = data['author_child']
    labels = data['label']
    G = nx.Graph()
    dic = {}
    return
        
#该函数用于从一个二维的NumPy数组中删除重复的行
def unique_rows(a):
    """
    Drops duplicate rows from a numpy 2d array
    """
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

#获取以指定节点为中心、在给定深度范围内的子图
def subgraph_dense(node, graph, graph_reverse, depth=1):
    """
    Returns subgraph around a concept with radius of depth
    """
    subgraph = []
    try:
        subgraph_edges = list(nx.ego_graph(graph, node, radius=depth).edges())
        for item in subgraph_edges:    
            subgraph.append([item[0], graph[item[0]][item[1]]['label'], item[1]])
    except nx.NodeNotFound:
        pass
    
    try:
        subgraph_edges_reverse = list(nx.ego_graph(graph_reverse, node, radius=depth).edges())
        for item in subgraph_edges_reverse:
            if item[0] == node:
                subgraph.append([item[1], graph_reverse[item[0]][item[1]]['label'], item[0]])
    except nx.NodeNotFound:
        pass
    
    return subgraph

#该函数用于获取以指定节点为中心的子图，仅包含其直接邻居节点
def subgraph_sparse(node, graph, graph_reverse):
    """
    Returns only the immediate neighbours around a concept
    """
    subgraph = []
    try:
        atlas = graph[node]
        for item in atlas:    
            subgraph.append([node, atlas[item]['label'], item])
    except KeyError:
        pass
    
    try:
        atlas = graph_reverse[node]
        for item in atlas:    
            subgraph.append([item, atlas[item]['label'], node])
    except KeyError:
        pass
    
    return subgraph

#该函数用于从ConceptNet中创建领域聚合图，该图以种子概念为基础
def domain_aggregated_graph(seeds, G, G_reverse, concept_map, relation_map, dense=True, depth=1):
    """
    Creates the domain aggregated graph from conceptnet with the seed concepts.
    """
    time.sleep(0.2)
    graph = []
    for node in tqdm(seeds):
        if dense:
            sg = subgraph_dense(node, G, G_reverse, depth)
        else:
            sg = subgraph_sparse(node, G, G_reverse)
            
        for triplet in sg:
            graph.append([concept_map[triplet[0]], relation_map[triplet[1]], concept_map[triplet[2]]])
            
    x = unique_rows(np.array(graph))

    unique_nodes = list(set(x[:, 0]).union(set(x[:, 2])))
    unique_nodes_mapping = {}

    for item in unique_nodes:
        unique_nodes_mapping[item] = len(unique_nodes_mapping)

    triplets = x.copy()
    triplets[:, 0] = np.vectorize(unique_nodes_mapping.get)(x[:, 0])
    triplets[:, 2] = np.vectorize(unique_nodes_mapping.get)(x[:, 2])

    return triplets, unique_nodes_mapping
#该函数用于获取特定概念（节点）的子图
def subgraph_for_concept(node, G, G_reverse, concept_map, relation_map, dense=True, depth=1):
    """
    Returns subgraph for a particular concept (node).
    """
    graph = []
           
    for triplet in sg:
        graph.append([concept_map[triplet[0]], relation_map[triplet[1]], concept_map[triplet[2]]])
            
    graph = np.array(graph)
    
    if len(graph) == 0:
        return np.zeros((1, 3))
    
    graph = unique_rows(graph)
    return graph
    