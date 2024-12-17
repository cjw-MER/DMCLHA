import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing

def uniform(size, tensor):#均匀分布,tensor是待初始化的张量，size是要初始化的张量的维度
    bound = 1.0 / math.sqrt(size)#计算出均匀分布的边界bound，限制张量的取值范围
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)#采样，将采样得到的值作为tensor的初始化值
    

class RGCN(torch.nn.Module):
    def __init__(self, num_entities, num_relations, num_bases, dropout):#num_entities和num_relations指明了节点和关系的数量，num_bases表示边的类型的数量，dropout用于防止模型过拟合
        super(RGCN, self).__init__()

        self.entity_embedding = nn.Embedding(num_entities, 100)#实体嵌入层
        self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, 100))#关系嵌入层

        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))

        self.conv1 = RGCNConv(100, 100, num_relations * 2, num_bases=num_bases)#关系图卷积层
        self.conv2 = RGCNConv(100, 100, num_relations * 2, num_bases=num_bases)
        self.conv3 = RGCNConv(100, 100, num_relations * 2, num_bases=num_bases)
        self.conv4 = RGCNConv(100, 100, num_relations * 2, num_bases=num_bases)
        self.dropout_ratio = dropout

    def forward(self, entity, edge_index, edge_type, edge_norm):#前向传播，输入图中的实体及其对应的边属性
        x = self.entity_embedding(entity)#
        x1 = self.conv1(x, edge_index, edge_type, edge_norm)#
        x = F.relu(self.conv1(x1, edge_index, edge_type, edge_norm))
        x = F.dropout(x, p = self.dropout_ratio, training = self.training)
        x2 = self.conv2(x, edge_index, edge_type, edge_norm)
        x = F.relu(self.conv1(x2, edge_index, edge_type, edge_norm))
        x = F.dropout(x, p = self.dropout_ratio, training = self.training)
        x3 = self.conv3(x, edge_index, edge_type, edge_norm) 
        
        return x3
        #return (x1+x2)/2#输出每个实体的特征表示

    def ccorr(a, b):#计算互相关函数（cross-correlation）的函数，a和b是输入的两个1D张量（tensor），这个函数会在信号处理等领域中用到，用于衡量两个信号之间在时间上的相似性
        return ifft(torch.conj(fft(a)) * fft(b)).real

    #不同的打分函数,实验1：选取不同的知识图谱嵌入模型:将图上的节点和边变成向量
    def distmult(self, embedding, triplets):#计算知识图谱中三元组的距离分数
        s = embedding[triplets[:,0]]
        r = self.relation_embedding[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.sum(s * r * o, dim=1)
        
        return score
    
    def transE(self, embedding, triplets):
        s = embedding[triplets[:,0]]
        r = self.relation_embedding[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = - torch.sqrt(torch.sum((s + r - o).pow(2), dim=1))
        
        return score

    def transF(self, embedding, triplets):
        s = embedding[triplets[:,0]]
        r = self.relation_embedding[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.diag(torch.mm((s+r),o.T)+torch.mm((o-r),s.T))
        return score
    #使用DistMult来恢复正确的边（从新打分）
    def DisMult(self, embedding, triplets):
        s = embedding[triplets[:,0]]
        r = self.relation_embedding[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.diag(torch.mm(r,(s*o).T))
        return score

    def HolE(self,embedding,triplets):
        s = embedding[triplets[:,0]]
        r = self.relation_embedding[triplets[:,1]]
        o = embedding[triplets[:,2]]
        u = self.ccorr(s,o)
        # print(u.shape)
        return torch.sum(r*u,dim=1)

    def score_loss(self, embedding, triplets, target):#三元组分类损失（即Edge Loss）
        score = self.DisMult(embedding, triplets)
        return score, F.binary_cross_entropy_with_logits(score, target)

    def reg_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.relation_embedding.pow(2))
    

class RGCNConv(MessagePassing):#实现一个基于图上消息传递的卷积操作，将每个节点的自身特征与邻居节点的特征进行卷积和汇聚
    """The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
        \mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int): Number of bases used for basis-decomposition.
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, num_relations, num_bases,
                 root_weight=True, bias=True, **kwargs):
        super(RGCNConv, self).__init__(aggr='mean', **kwargs)# aggr='mean' 表示使用平均池化来汇聚节点特征

        self.in_channels = in_channels#输入的节点特征的维度
        self.out_channels = out_channels#输出的节点特征的维度
        self.num_relations = num_relations#关系的数量
        self.num_bases = num_bases#边的类型的数量

        self.basis = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = nn.Parameter(torch.Tensor(num_relations, num_bases))

        if root_weight:#一个布尔类型的参数，如果为True，则需要学习一个用于转换输入特征的权重矩阵root；
            self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:#一个布尔类型的参数，如果为True，则需要学习一个用于偏置的向量bias
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()#调用reset_parameters()函数对模型的所有可训练参数进行了随机初始化

    def reset_parameters(self):#参数重置
        size = self.num_bases * self.in_channels
        uniform(size, self.basis)
        uniform(size, self.att)
        uniform(size, self.root)
        uniform(size, self.bias)


    def forward(self, x, edge_index, edge_type, edge_norm=None, size=None):
        """"""
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type,
                              edge_norm=edge_norm)


    def message(self, x_j, edge_index_j, edge_type, edge_norm):#x_j 是源节点的特征，edge_index_j 是与源节点相连的边的索引
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))

        # If no node features are given, we implement a simple embedding
        # loopkup based on the target node index and its edge type.
        if x_j is None:
            w = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_j
            out = torch.index_select(w, 0, index)
        else:
            w = w.view(self.num_relations, self.in_channels, self.out_channels)
            w = torch.index_select(w, 0, edge_type)
            out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):#更新节点的特征表示                                                                                                                                                                                                                                                                                                                                            可能是从邻居节点汇总的信息,x 是当前节点的特征表示
        if self.root is not None:
            if x is None:
                out = aggr_out + self.root
            else:
                out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)
        
    