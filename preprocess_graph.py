import torch
from tqdm import tqdm
import numpy as np
import os.path, pickle
from random import sample
import  os, numpy as np
import random
import pandas as pd
from datetime import datetime, timezone
from utils_graph import conceptnet_graph

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True  
#获取图中所有的节点,总共23101个节点
def get_nodes(data):
    data = data.sort_values(by=['datetime'])#按照日期时间的顺序处理数据
    parent = set(data['author_parent'])#提取数据中的'author_parent'列，将其转换为一个集合
    child = set(data['author_child'])
    nodes = parent.union(child)#合并集合
    return nodes

if __name__ == '__main__':
    data = pd.read_csv("labeled_data.csv", index_col = False,encoding='latin-1')
    '''data1 = pd.read_csv("Brexit.csv",index_col = False,encoding='latin-1')
    data2 = pd.read_csv("climate.csv",index_col = False,encoding='latin-1')
    data3 = pd.read_csv("BlackLivesMatter.csv",index_col = False,encoding='latin-1')
    data4 = pd.read_csv("Republican.csv",index_col = False,encoding='latin-1')
    data5 = pd.read_csv("democrats.csv",index_col = False,encoding='latin-1')
    data = pd.concat([data1, data2, data3, data4, data5], ignore_index=True)'''
    all_nodes = get_nodes(data)#获取节点集合
    unique_nodes_mapping = {}#为每个节点编号,从0——23100

    for item in all_nodes:#遍历all_nodes集合中的每个节点'item'
        unique_nodes_mapping[item] = len(unique_nodes_mapping)#将item作为键，将当前字典的长度作为值添加到unique_nodes_mapping字典中

    relation_matrix = {}#原始图,小于0是中立关系，等于0是朋友关系，大于0是敌人关系,{(12108, 19426): 1}
    meet_matrix = {}
    relation_map = {'friend':1,'enemy':2,'neutral':0,'interact':3}#将关系标签映射到数字编码，interact表示互动


    data = data.sort_values(by=['datetime'])
    print(len(unique_nodes_mapping))
    print(data.shape)#获取data数据框的形状，即行数和列数
    
    train = data.iloc[0:int(len(data)*0.8)]#将原始数据集的前80%划分为训练集
    #train = pd.concat([data2, data4, data5, data3], ignore_index=True)
    #train = train.sort_values(by=['datetime'])

    parent = (train['author_parent'])#从train训练集中提取'author_parent'列的值
    child = (train['author_child'])
    relations = train['label']
    
    #构造原始图
    triplets = []#(356, 3, 8806)-->(child,relation,parent)
    for p in enumerate(zip(child,parent,relations)):#遍历三元组
        p=p[1]
        if (unique_nodes_mapping[p[0]],unique_nodes_mapping[p[1]]) in relation_matrix:
            relation_matrix[(unique_nodes_mapping[p[0]],unique_nodes_mapping[p[1]])] += p[2]-1#把中立的标签变为小于0，把朋友的标签变为等于0，把敌人的关系变为大于0
        else : relation_matrix[(unique_nodes_mapping[p[0]],unique_nodes_mapping[p[1]])] =p[2]-1#把关系变为-1、0、1

    #负采样 
    concept_net = {}#负采样的图
    for u in (unique_nodes_mapping.values()):
        concept_net[u] = []#对于每个索引值u，将其作为concept_net字典的键，并将对应值设置为空列表[],{23100: []} 
    for key in relation_matrix.keys():#遍历relation_matrix字典的所有键，即节点对
        if relation_matrix[key] > 0:#原关系为敌人
            if random.random()>0.3:
                triplets.append((key[0],1,key[1]))#把敌人变为朋友关系
                concept_net[key[0]].append([key[0],1,key[1]])
            else:
                triplets.append((key[0],3,key[1]))#双向边：互动
                concept_net[key[0]].append([key[0],3,key[1]])
        elif relation_matrix[key] < 0:#原关系为中立
            if random.random()>0.3:
                triplets.append((key[0],2,key[1]))#把中立变为敌人关系
                concept_net[key[0]].append([key[0],2,key[1]])
            else: 
                triplets.append((key[0],3,key[1]))#双向边：互动
                concept_net[key[0]].append([key[0],3,key[1]])
        else:#原关系为朋友
            if random.random()>0.3:
                triplets.append((key[0],0,key[1]))#把朋友关系变为中立
                concept_net[key[0]].append([key[0],0,key[1]])
            else: 
                triplets.append((key[0],3,key[1]))#双向边：互动
                concept_net[key[0]].append([key[0],3,key[1]])        
    test = data.iloc[int(len(data)*0.8):-1]#将剩余的20%的数据创建为测试数据集
    #test = data3
    #test = test.sort_values(by=['datetime'])
    parent = (test['author_parent'])#从测试集数据中分别获取author_parent、author_child和label列的数据
    child = (test['author_child'])
    relations = test['label']
    
    #处理测试集中的数据，针对于那些在训练集中没有出现过的节点对,给他们修改边的类型为interact:3
    i = 0
    for p in enumerate(zip(child,parent,relations)):
        
        p=p[1]#p = p[1]将p的值更新为元组中的第二个元素
        if not ((unique_nodes_mapping[p[0]],unique_nodes_mapping[p[1]]) in relation_matrix):
            triplets.append((unique_nodes_mapping[p[0]],3,unique_nodes_mapping[p[1]]))
            relation_matrix[(unique_nodes_mapping[p[0]],unique_nodes_mapping[p[1]])] = 10000
            concept_net[unique_nodes_mapping[p[0]]].append((unique_nodes_mapping[p[0]],3,unique_nodes_mapping[p[1]]))
            i+=1

    triplets = np.array(triplets)#将triplets列表转换为NumPy数组

    if not os.path.exists('utils'):
        os.mkdir('utils')
    #使用pickle模块将数据保存到文件中，使用numpy的np.ndarray.dump()函数将NumPy数组保存到文件中
    pickle.dump(all_nodes, open('utils/all_n_nodes_bert.pkl', 'wb'))
    pickle.dump(relation_map, open('utils/relation_n_map_bert.pkl', 'wb'))
    pickle.dump(unique_nodes_mapping, open('utils/unique_nodes_n_mapping_bert.pkl', 'wb'))
    pickle.dump(concept_net, open('utils/concept_n_graphs_bert.pkl', 'wb'))
    np.ndarray.dump(triplets, open('utils/triplets_n_bert.np', 'wb'))