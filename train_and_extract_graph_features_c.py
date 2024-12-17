import random
from tqdm import tqdm
from utils_graph import unique_rows
from glue_utils import get_stance_dataset
import numpy as np, pickle, argparse

import torch
import torch.nn.functional as F
from rgcn import RGCN
from torch_scatter import scatter_add
from torch_geometric.data import Data
import pandas as pd
import os

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
#边采样
def sample_edge_uniform(n_triples, sample_size):#从给定的边中均匀地采样一定数量的边，n_triples表示边的总数，sample_size表示要采样的边的数量
    """Sample edges uniformly from all the edges."""
    all_edges = np.arange(n_triples)#创建一个包含从0到n_triples-1的所有边的数组
    return np.random.choice(all_edges, sample_size, replace=False)#从all_edges数组中均匀地选择sample_size个边
#负采样（正样本:原来有连边，负样本：原来没连边）
def negative_sampling(pos_samples, num_entity, negative_rate):#pos_samples表示正样本集合，num_entity表示实体的数量，negative_rate表示负采样率，即每个正样本生成的负样本的数量
    size_of_batch = len(pos_samples)#获取正样本集合的大小
    num_to_generate = size_of_batch * negative_rate#计算需要生成的负样本的数量
    neg_samples = np.tile(pos_samples, (negative_rate, 1))#tile函数将正样本复制negative_rate次，生成一个重复的负样本集合，每个正样本都会对应 negative_rate 个负样本
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)#创建一个用于标记正例和负例的标签数组labels，初始值为0
    labels[: size_of_batch] = 1#前size_of_batch个元素置为1，表示正例，后面的元素为0，表示负例
    values = np.random.choice(num_entity, size=num_to_generate)#从0到num_entity-1中随机选择num_to_generate个实体编号，存储在values变量中。这将作为负样本的实体编号
    choices = np.random.uniform(size=num_to_generate)#生成num_to_generate个在0到1之间的随机数，存储在choices变量中。这些随机数将用于决定负样本是替换正样本的主语（subj）还是宾语（obj）
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    return np.concatenate((pos_samples, neg_samples)), labels#将正样本和负样本集合进行合并

def edge_normalization(edge_type, edge_index, num_entity, num_relation):
    """
        Edge normalization trick
        - one_hot: (num_edge, num_relation)
        - deg: (num_node, num_relation)
        - index: (num_edge)
        - deg[edge_index[0]]: (num_edge, num_relation)
        - edge_norm: (num_edge)
    """
    one_hot = F.one_hot(edge_type, num_classes = 2 * num_relation).to(torch.float)#将边的类型edge_type转换为独热编码形式
    deg = scatter_add(one_hot, edge_index[0], dim = 0, dim_size = num_entity)
    index = edge_type + torch.arange(len(edge_index[0])) * (2 * num_relation)
    edge_norm = 1 / deg[edge_index[0]].view(-1)[index]#计算每条边的边归一化系数edge_norm

    return edge_norm

#生成负采样的图
#triplets表示三元组数据，sample_size表示采样的边的数量，split_size表示将采样的边分割成训练集和验证集的比例，num_entity表示实体的数量，num_rels表示关系的数量，negative_rate表示负采样的比率
def generate_sampled_graph_and_labels(triplets, sample_size, split_size, num_entity, num_rels, negative_rate):
    """
        Get training graph and labels with negative sampling.
    """
    edges = triplets
    src, rel, dst = edges.transpose()#从三元组数据中提取出边的源节点、关系和目标节点
    uniq_entity, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()

    # Negative sampling
    #对重新编号后的边进行负采样，得到采样后的边samples和相应的标签labels。
    samples, labels = negative_sampling(relabeled_edges, len(uniq_entity), negative_rate)

    # further split graph, only half of the edges will be used as graph
    # structure, while the rest half is used as unseen positive samples
    split_size = int(sample_size * split_size)#将采样的边数量乘以split_size得到分割图的边数
    graph_split_ids = np.random.choice(np.arange(sample_size),
                                       size=split_size, replace=False)
    #根据分割图的索引从源节点、目标节点和关系中提取出相应的子集，分别存储在src、dst和rel中
    src = torch.tensor(src[graph_split_ids], dtype = torch.long).contiguous()
    dst = torch.tensor(dst[graph_split_ids], dtype = torch.long).contiguous()
    rel = torch.tensor(rel[graph_split_ids], dtype = torch.long).contiguous()

    # Create bi-directional graph
    src, dst = torch.cat((src, dst)), torch.cat((dst, src))#创建双向图，将源节点和目标节点分别拼接起来
    rel = torch.cat((rel, rel + num_rels))

    edge_index = torch.stack((src, dst))#构建边索引edge_index，其中包含源节点和目标节点的信息
    edge_type = rel#将边的类型信息存储在edge_type中

    data = Data(edge_index = edge_index)
    data.entity = torch.from_numpy(uniq_entity)
    data.edge_type = edge_type
    data.edge_norm = edge_normalization(edge_type, edge_index, len(uniq_entity), num_rels)
    data.samples = torch.from_numpy(samples)
    data.labels = torch.from_numpy(labels)
    
    return data
#生成经过图自编码器后的图
def generate_graph(triplets, num_rels):#传入三元组及三元组的数量
    """
        Get feature extraction graph without negative sampling.
    """
    edges = triplets#将triplets存储在edges变量中
    src, rel, dst = edges.transpose()   
    uniq_entity, edges = np.unique((src, dst), return_inverse=True)#使用np.unique函数对源节点和目标节点进行去重，得到唯一的实体节点编号，保存在uniq_entity中
    src, dst = np.reshape(edges, (2, -1))#使用np.reshape将src和dst数组重新调整为两行
    relabeled_edges = np.stack((src, rel, dst)).transpose()#将重新调整的src、rel和dst数组堆叠起来，并进行转置，创建relabeled_edges数组。
    
    src = torch.tensor(src, dtype = torch.long).contiguous()#将src数组转换为torch.long数据类型的PyTorch张量
    dst = torch.tensor(dst, dtype = torch.long).contiguous()#contiguous()方法返回一个连续的张量，以确保数据在内存中是连续存储的
    rel = torch.tensor(rel, dtype = torch.long).contiguous()

    # Create bi-directional graph
    src, dst = torch.cat((src, dst)), torch.cat((dst, src))#torch.cat((src, dst))将源节点索引和目标节点索引连接在一起，torch.cat((dst, src))将目标节点索引和源节点索引连接在一起，以确保边是双向的
    rel = torch.cat((rel, rel + num_rels))#将关系类型索引扩展为双向关系，以适应双向边的表示

    edge_index = torch.stack((src, dst))#创建一个包含图的边索引的二维张量
    edge_type = rel#将处理后的关系类型索引数组分配给edge_type变量

    data = Data(edge_index = edge_index)
    data.entity = torch.from_numpy(uniq_entity)
    data.edge_type = edge_type
    data.edge_norm = edge_normalization(edge_type, edge_index, len(uniq_entity), num_rels)

    return data

def sentence_features(model, split, all_seeds, concept_graphs, relation_map, unique_nodes_mapping):
    """
        Graph features for each sentence (document) instance in a domain.
    """
    x, dico = get_stance_dataset(exp_type=split)#从数据集中获取句子特征（表示为x）和字典（表示为dico）
    d = list(dico.values())#将字典的值转换为列表

    sent_features = np.zeros((len(x), 100))#创建一个大小为(len(x), 100)的零矩阵，用于存储句子特征
    
    for j in tqdm(range(len(x)), position=0, leave=False):#对于数据集中的每个句子（文档）实例，进行迭代
        c = [dico.id2token[item] for item in np.where(x[j] != 0)[0]]#根据句子特征x中非零标识符的索引，获取对应的词语列表
        n = list(spacy_seed_concepts_list(c).intersection(set(all_seeds)))#根据词语列表c，使用SpaCy进行词性标注并提取种子概念列表all_seeds中的概念

        try:
            xg = np.concatenate([concept_graphs[item] for item in n])#从concept_graphs中获取对应的概念图，并将它们连接起来，形成一个大的图
            xg = xg[~np.all(xg == 0, axis=1)]#从连接的图中移除全零行
        
            absent1 = set(xg[:, 0]) - unique_nodes_mapping.keys()#获取图中第一列节点的集合与unique_nodes_mapping中键的差集
            absent2 = set(xg[:, 2]) - unique_nodes_mapping.keys()#获取图中第三列节点的集合与unique_nodes_mapping中键的差集
            absent = absent1.union(absent2)#将两个差集合并为一个集合,表示在图中存在但在unique_nodes_mapping中不存在的节点

            for item in absent:
                xg = xg[~np.any(xg == item, axis=1)]
        
            xg[:, 0] = np.vectorize(unique_nodes_mapping.get)(xg[:, 0])#使用unique_nodes_mapping字典将概念图中的第一列节点映射为其对应的值
            xg[:, 2] = np.vectorize(unique_nodes_mapping.get)(xg[:, 2])#使用unique_nodes_mapping字典将概念图中的第三列节点映射为其对应的值
            xg = unique_rows(xg).astype('int64')#移除重复的行，并将概念图的数据类型转换为int64
            if len(xg) > 50000:#如果概念图的行数超过50000，则截取前50000行
                xg = xg[:50000, :]

            sg = generate_graph(xg, len(relation_map)).to(torch.device('cuda'))#根据给定的概念图数据xg和关系映射长度len(relation_map)生成图数据sg，并将其转移到CUDA设备上进行计算
            features = model(sg.entity, sg.edge_index, sg.edge_type, sg.edge_norm)#使用模型model对图数据sg进行计算，提取图的特征
            #print(features),exit()
            sent_features[j] = features.cpu().detach().numpy().mean(axis=0)#将计算得到的特征取平均，并将结果存储在sentence_features数组的第j行           
            torch.cuda.empty_cache()
            
        except ValueError:
            pass
    
    return sent_features


def train(train_triplets, model, batch_size, split_size, negative_sample, reg_ratio, num_entities, num_relations):
#生成训练数据train_data，包括采样的图数据和相应的标签。这些数据用于训练模型。
    train_data = generate_sampled_graph_and_labels(train_triplets, batch_size, split_size, 
                                                   num_entities, num_relations, negative_sample)

    train_data.to(torch.device('cuda'))#将train_data移动到CUDA设备上进行计算

    entity_embedding = model(train_data.entity, train_data.edge_index, train_data.edge_type, train_data.edge_norm)
    score, loss = model.score_loss(entity_embedding, train_data.samples, train_data.labels) 
    loss += reg_ratio * model.reg_loss(entity_embedding)
    return score, loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=10000, help='graph batch size')
    parser.add_argument('--split-size', type=float, default=0.5, help='what fraction of graph edges used in training')#用于训练的图边的分割比例
    parser.add_argument('--ns', type=int, default=1, help='negative sampling ratio')
    parser.add_argument('--epochs', type=int, default=2000, help='number of epochs')#2000
    parser.add_argument('--save', type=int, default=200, help='save after how many epochs')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.25, help='learning rate')
    parser.add_argument('--reg', type=float, default=1e-2, help='regularization coefficient')#正则化系数
    parser.add_argument('--grad-norm', type=float, default=1.0, help='grad norm')#梯度范数
    args = parser.parse_args()
    print(args)
    torch.cuda.set_device(0)#将CUDA设备设置为设备ID为1的设备

    set_seed(42)#设置随机种子为42

    graph_batch_size = args.batch_size
    graph_split_size = args.split_size
    negative_sample = args.ns
    n_epochs = args.epochs
    save_every = args.save
    lr = args.lr
    dropout = args.dropout
    regularization = args.reg
    grad_norm = args.grad_norm


    all_seeds = pickle.load(open('utils/all_n_nodes_bert.pkl', 'rb'))#包含所有种子节点的数据文件
    relation_map = pickle.load(open('utils/relation_n_map_bert.pkl', 'rb'))#包含关系映射的数据文件
    unique_nodes_mapping = pickle.load(open('utils/unique_nodes_n_mapping_bert.pkl', 'rb'))#包含唯一节点映射的数据文件#修改为评论回复的图
    train_triplets = np.load(open('utils/triplets_n_c_bert.np', 'rb'), allow_pickle=True)
    
    n_bases = 4#num_bases: R-GCN中的关系的数量


    model = RGCN(len(unique_nodes_mapping), len(relation_map), num_bases=n_bases, dropout=dropout).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)#优化器采用Adam算法

    for epoch in tqdm(range(1, (n_epochs + 1)), desc='Epochs', position=0):

        permutation = torch.randperm(len(train_triplets))#随机打乱训练数据的索引顺序
        losses = []#初始化一个空列表losses，用于存储每个小批量训练的损失值

        for i in range(0, len(train_triplets), graph_batch_size):
            
            model.train()
            optimizer.zero_grad()
            
            indices = permutation[i:i+graph_batch_size]

            score, loss = train(train_triplets[indices], model, batch_size=len(indices), split_size=graph_split_size, 
                                negative_sample=negative_sample, reg_ratio = regularization, 
                                num_entities=len(unique_nodes_mapping), num_relations=len(relation_map))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)#对梯度进行裁剪，以防止梯度爆炸问题。
            optimizer.step()
            losses.append(loss.item())

        avg_loss = round(sum(losses)/len(losses), 4)

        if not os.path.exists('weights'):
            os.mkdir('weights')

        if epoch%save_every == 0:
            tqdm.write("Epoch {} Train Loss: {}".format(epoch, avg_loss))
            torch.save(model.state_dict(), 'weights/model_n_c_epoch_' + str(epoch) +'.pt')
    model.eval()
    print ('Done.')       
    