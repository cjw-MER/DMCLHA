import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import copy
#from model_utils import *
from multihead_attention import MultiheadAttention

class VanillaAttention(nn.Module):
    """
    Vanilla attention layer is implemented by linear layer.

    Args:
        input_tensor (torch.Tensor): the input of the attention layer

    Returns:
        hidden_states (torch.Tensor): the outputs of the attention layer
        weights (torch.Tensor): the attention weights

    """

    def __init__(self, hidden_dim, attn_dim):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.ReLU(True), nn.Linear(attn_dim, 1))

    def forward(self, input_tensor):
        # (B, Len, num, H) -> (B, Len, num, 1)
        energy = self.projection(input_tensor)
        weights = torch.softmax(energy.squeeze(-1), dim=-1)
        # (B, Len, num, H) * (B, Len, num, 1) -> (B, len, H)
        hidden_states = (input_tensor * weights.unsqueeze(-1)).sum(dim=-2)
        return hidden_states, weights

class FeedForward(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.

    Args:
        input_tensor (torch.Tensor): the input of the point-wise feed-forward layer

    Returns:
        hidden_states (torch.Tensor): the output of the point-wise feed-forward layer

    """

    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": F.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class GCNLayer1(nn.Module):
    def __init__(self, in_feats, out_feats, use_topic=False, new_graph=True):
        super(GCNLayer1, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.use_topic = use_topic
        self.new_graph = new_graph

    def forward(self, inputs, dia_len, topicLabel):
        if self.new_graph:
            # pdb.set_trace()
            adj = self.message_passing_directed_speaker(inputs, dia_len, topicLabel)
        else:
            adj = self.message_passing_wo_speaker(inputs, dia_len, topicLabel)
        x = torch.matmul(adj, inputs)
        x = self.linear(x)
        return x

    def cossim(self, x, y):
        a = torch.matmul(x, y)
        b = torch.sqrt(torch.matmul(x, x)) * torch.sqrt(torch.matmul(y, y))
        if b == 0:
            return 0
        else:
            return (a / b)

    def atom_calculate_edge_weight(self, x, y):
        f = self.cossim(x, y)
        if f > 1 and f < 1.05:
            f = 1
        elif f < -1 and f > -1.05:
            f = -1
        elif f >= 1.05 or f <= -1.05:
            print('cos = {}'.format(f))
        return f

    def message_passing_wo_speaker(self, x, dia_len, topicLabel):
        adj = torch.zeros((x.shape[0], x.shape[0])) + torch.eye(x.shape[0])
        start = 0

        for i in range(len(dia_len)):  #
            for j in range(dia_len[i] - 1):
                for pin in range(dia_len[i] - 1 - j):
                    xz = start + j
                    yz = xz + pin + 1
                    f = self.cossim(x[xz], x[yz])
                    if f > 1 and f < 1.05:
                        f = 1
                    elif f < -1 and f > -1.05:
                        f = -1
                    elif f >= 1.05 or f <= -1.05:
                        print('cos = {}'.format(f))
                    Aij = 1 - math.acos(f) / math.pi
                    adj[xz][yz] = Aij
                    adj[yz][xz] = Aij
            start += dia_len[i]

        if self.use_topic:
            for (index, topic_l) in enumerate(topicLabel):
                xz = index
                yz = x.shape[0] + topic_l - 7
                f = self.cossim(x[xz], x[yz])
                if f > 1 and f < 1.05:
                    f = 1
                elif f < -1 and f > -1.05:
                    f = -1
                elif f >= 1.05 or f <= -1.05:
                    print('cos = {}'.format(f))
                Aij = 1 - math.acos(f) / math.pi
                adj[xz][yz] = Aij
                adj[yz][xz] = Aij

        d = adj.sum(1)
        D = torch.diag(torch.pow(d, -0.5))
        adj = D.mm(adj).mm(D).cuda()

        return adj

    def message_passing_directed_speaker(self, x, dia_len, qmask):
        total_len = sum(dia_len)
        adj = torch.zeros((total_len, total_len)) + torch.eye(total_len)
        start = 0
        use_utterance_edge = False
        for (i, len_) in enumerate(dia_len):
            speaker0 = []
            speaker1 = []
            for (j, speaker) in enumerate(qmask[start:start + len_]):
                # if speaker[0] == 1:
                if speaker[0][0] == 1:
                    speaker0.append(j)
                else:
                    speaker1.append(j)
            if use_utterance_edge:
                for j in range(len_ - 1):
                    f = self.atom_calculate_edge_weight(x[start + j], x[start + j + 1])
                    Aij = 1 - math.acos(f) / math.pi
                    adj[start + j][start + j + 1] = Aij
                    adj[start + j + 1][start + j] = Aij
            for k in range(len(speaker0) - 1):
                f = self.atom_calculate_edge_weight(x[start + speaker0[k]], x[start + speaker0[k + 1]])
                Aij = 1 - math.acos(f) / math.pi
                adj[start + speaker0[k]][start + speaker0[k + 1]] = Aij
                adj[start + speaker0[k + 1]][start + speaker0[k]] = Aij
            for k in range(len(speaker1) - 1):
                f = self.atom_calculate_edge_weight(x[start + speaker1[k]], x[start + speaker1[k + 1]])
                Aij = 1 - math.acos(f) / math.pi
                adj[start + speaker1[k]][start + speaker1[k + 1]] = Aij
                adj[start + speaker1[k + 1]][start + speaker1[k]] = Aij

            start += dia_len[i]

        return adj


class GCN_2Layers(nn.Module):
    def __init__(self, lstm_hid_size, gcn_hid_dim, num_class, dropout, use_topic=False, use_residue=True, return_feature=False):
        super(GCN_2Layers, self).__init__()

        self.lstm_hid_size = lstm_hid_size
        self.gcn_hid_dim = gcn_hid_dim
        self.num_class = num_class
        self.dropout = dropout
        self.use_topic = use_topic
        self.return_feature = return_feature

        self.gcn1 = GCNLayer1(self.lstm_hid_size, self.gcn_hid_dim, self.use_topic)
        self.use_residue = use_residue
        if self.use_residue:
            self.gcn2 = GCNLayer1(self.gcn_hid_dim, self.gcn_hid_dim, self.use_topic)
            self.linear = nn.Linear(self.lstm_hid_size + self.gcn_hid_dim, self.num_class)
        else:
            self.gcn2 = GCNLayer1(self.gcn_hid_dim, self.num_class, self.use_topic)

    def forward(self, x, dia_len, topicLabel):
        x_graph = self.gcn1(x, dia_len, topicLabel)
        if not self.use_residue:
            x = self.gcn2(x_graph, dia_len, topicLabel)
            if self.return_feature:
                print("Error, you should change the state of use_residue")
        else:
            x_graph = self.gcn2(x_graph, dia_len, topicLabel)
            x = torch.cat([x, x_graph], dim=-1)
            if self.return_feature:
                return x
            x = self.linear(x)
        log_prob = F.log_softmax(x, 1)

        return log_prob


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2 * in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l):
        theta = math.log(lamda / l + 1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support

        output = theta * torch.mm(support, self.weight) + (1 - theta) * r
        if self.residual:
            output = output + input
        return output


class TextCNN(nn.Module):
    def __init__(self, input_dim, emb_size=128, in_channels=1, out_channels=128, kernel_heights=[3, 4, 5], dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], input_dim), stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], input_dim), stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], input_dim), stride=1, padding=0)
        self.dropout = nn.Dropout(dropout)
        self.embd = nn.Sequential(
            nn.Linear(len(kernel_heights) * out_channels, emb_size),
            nn.ReLU(inplace=True),
        )

    def conv_block(self, input, conv_layer):
        # (batch_size, out_channels, dim, 1)
        conv_out = conv_layer(input)
        activation = F.relu(conv_out.squeeze(3))
        # (batch_size, out_channels)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)
        return max_out

    def forward(self, frame_x):
        batch_size, seq_len, feat_dim = frame_x.size()  # dia_len, utt_len, batch_size, feat_dim
        frame_x = frame_x.view(batch_size, 1, seq_len, feat_dim)
        max_out1 = self.conv_block(frame_x, self.conv1)
        max_out2 = self.conv_block(frame_x, self.conv2)
        max_out3 = self.conv_block(frame_x, self.conv3)
        all_out = torch.cat((max_out1, max_out2, max_out3), 1)
        fc_in = self.dropout(all_out)
        embd = self.embd(fc_in)
        return embd


class GCNII(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant, return_feature, use_residue, new_graph=False, reason_flag=False):
        super(GCNII, self).__init__()
        self.return_feature = return_feature
        self.use_residue = use_residue
        self.new_graph = new_graph
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden, variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        if not return_feature:
            self.fcs.append(nn.Linear(nfeat + nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

        self.rnn_layer = 1
        self.rnn = torch.nn.LSTM(nhidden, nhidden, self.rnn_layer)  # 400,200,1
        self.reason_flag = reason_flag

    def cossim(self, x, y):
        a = torch.matmul(x, y)
        b = torch.sqrt(torch.matmul(x, x)) * torch.sqrt(torch.matmul(y, y))
        if b == 0:
            return 0
        else:
            return (a / b)

    def forward(self, x, dia_len, qmask):
        if self.new_graph:
            adj = self.message_passing_directed_speaker(x, dia_len, qmask)
        else:
            adj = self.create_big_adj(x, dia_len)
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        h = (torch.zeros_like(layer_inner).unsqueeze(0).repeat(self.rnn_layer, 1, 1),
             torch.zeros_like(layer_inner).unsqueeze(0).repeat(self.rnn_layer, 1, 1))  # [0]  # v1
        for i, con in enumerate(self.convs):
            if self.reason_flag:
                q = layer_inner
                # output, (hn, cn) = rnn(input, (h0, c0))
                layer_inner, h = self.rnn(q.unsqueeze(0), h)
                layer_inner = layer_inner.squeeze(0)
            layer_inner = self.act_fn(con(layer_inner, adj, _layers[0], self.lamda, self.alpha, i + 1))
            # layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            if self.reason_flag:
                layer_inner += q

        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        if self.use_residue:
            layer_inner = torch.cat([x, layer_inner], dim=-1)
        if not self.return_feature:
            layer_inner = self.fcs[-1](layer_inner)
            layer_inner = F.log_softmax(layer_inner, dim=1)
        return layer_inner

    def create_big_adj(self, x, dia_len):
        adj = torch.zeros((x.shape[0], x.shape[0]))
        start = 0
        for i in range(len(dia_len)):
            sub_adj = torch.zeros((dia_len[i], dia_len[i]))
            temp = x[start:start + dia_len[i]]
            temp_len = torch.sqrt(torch.bmm(temp.unsqueeze(1), temp.unsqueeze(2)).squeeze(-1).squeeze(-1))
            temp_len_matrix = temp_len.unsqueeze(1) * temp_len.unsqueeze(0)
            cos_sim_matrix = torch.matmul(temp, temp.permute(1, 0)) / temp_len_matrix
            sim_matrix = torch.acos(cos_sim_matrix * 0.99999)
            sim_matrix = 1 - sim_matrix / math.pi

            sub_adj[:dia_len[i], :dia_len[i]] = sim_matrix

            m_start = start
            n_start = start
            adj[m_start:m_start + dia_len[i], n_start:n_start + dia_len[i]] = sub_adj

            start += dia_len[i]
        d = adj.sum(1)
        D = torch.diag(torch.pow(d, -0.5))
        adj = D.mm(adj).mm(D).cuda()

        return adj

    def message_passing_wo_speaker(self, x, dia_len, topicLabel):
        adj = torch.zeros((x.shape[0], x.shape[0])) + torch.eye(x.shape[0])
        start = 0
        for i in range(len(dia_len)):  #
            for j in range(dia_len[i] - 1):
                for pin in range(dia_len[i] - 1 - j):
                    xz = start + j
                    yz = xz + pin + 1
                    f = self.cossim(x[xz], x[yz])
                    if f > 1 and f < 1.05:
                        f = 1
                    elif f < -1 and f > -1.05:
                        f = -1
                    elif f >= 1.05 or f <= -1.05:
                        print('cos = {}'.format(f))
                    Aij = 1 - math.acos(f) / math.pi
                    adj[xz][yz] = Aij
                    adj[yz][xz] = Aij
            start += dia_len[i]

        d = adj.sum(1)
        D = torch.diag(torch.pow(d, -0.5))
        adj = D.mm(adj).mm(D).cuda()

        return adj

    def atom_calculate_edge_weight(self, x, y):
        f = self.cossim(x, y)
        if f > 1 and f < 1.05:
            f = 1
        elif f < -1 and f > -1.05:
            f = -1
        elif f >= 1.05 or f <= -1.05:
            print('cos = {}'.format(f))
        return f

    def message_passing_directed_speaker(self, x, dia_len, qmask):
        total_len = sum(dia_len)
        adj = torch.zeros((total_len, total_len)) + torch.eye(total_len)
        start = 0
        use_utterance_edge = False
        for (i, len_) in enumerate(dia_len):
            speaker0 = []
            speaker1 = []
            for (j, speaker) in enumerate(qmask[i][0:len_]):
                if speaker[0] == 1:
                    speaker0.append(j)
                else:
                    speaker1.append(j)
            if use_utterance_edge:
                for j in range(len_ - 1):
                    f = self.atom_calculate_edge_weight(x[start + j], x[start + j + 1])
                    Aij = 1 - math.acos(f) / math.pi
                    adj[start + j][start + j + 1] = Aij
                    adj[start + j + 1][start + j] = Aij
            for k in range(len(speaker0) - 1):
                f = self.atom_calculate_edge_weight(x[start + speaker0[k]], x[start + speaker0[k + 1]])
                Aij = 1 - math.acos(f) / math.pi
                adj[start + speaker0[k]][start + speaker0[k + 1]] = Aij
                adj[start + speaker0[k + 1]][start + speaker0[k]] = Aij
            for k in range(len(speaker1) - 1):
                f = self.atom_calculate_edge_weight(x[start + speaker1[k]], x[start + speaker1[k + 1]])
                Aij = 1 - math.acos(f) / math.pi
                adj[start + speaker1[k]][start + speaker1[k + 1]] = Aij
                adj[start + speaker1[k + 1]][start + speaker1[k]] = Aij

            start += dia_len[i]

        d = adj.sum(1)
        D = torch.diag(torch.pow(d, -0.5))
        adj = D.mm(adj).mm(D).cuda()

        return adj.cuda()

    def message_passing_relation_graph(self, x, dia_len):
        total_len = sum(dia_len)
        adj = torch.zeros((total_len, total_len)) + torch.eye(total_len)
        window_size = 10
        start = 0
        for (i, len_) in enumerate(dia_len):
            edge_set = []
            for k in range(len_):
                left = max(0, k - window_size)
                right = min(len_ - 1, k + window_size)
                edge_set = edge_set + [str(i) + '_' + str(j) for i in range(left, right) for j in range(i + 1, right + 1)]
            edge_set = [[start + int(str_.split('_')[0]), start + int(str_.split('_')[1])] for str_ in list(set(edge_set))]
            for left, right in edge_set:
                f = self.atom_calculate_edge_weight(x[left], x[right])
                Aij = 1 - math.acos(f) / math.pi
                adj[left][right] = Aij
                adj[right][left] = Aij
            start += dia_len[i]

        d = adj.sum(1)
        D = torch.diag(torch.pow(d, -0.5))
        adj = D.mm(adj).mm(D).cuda()

        return adj.cuda()


class GCNII_lyc(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant, return_feature, use_residue, new_graph=False, reason_flag=False):
        super(GCNII_lyc, self).__init__()
        self.return_feature = return_feature
        self.use_residue = use_residue
        self.new_graph = new_graph
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden, variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        if not return_feature:
            self.fcs.append(nn.Linear(nfeat + nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

        self.rnn_layer = 1
        self.rnn = torch.nn.LSTM(nhidden, nhidden, self.rnn_layer)  # 400,200,1
        self.reason_flag = reason_flag

    def cossim(self, x, y):
        a = torch.matmul(x, y)
        b = torch.sqrt(torch.matmul(x, x)) * torch.sqrt(torch.matmul(y, y))
        if b == 0:
            return 0
        else:
            return (a / b)

    def forward(self, x, dia_len, topicLabel, adj=None, test_label=False):
        if adj is None:
            if self.new_graph:
                adj = self.message_passing_relation_graph(x, dia_len)
            else:
                adj = self.message_passing_wo_speaker(x, dia_len, topicLabel)
        else:
            adj = adj
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))  # 4971,100 << 4971,200
        _layers.append(layer_inner)
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)  # 873,100

        h = (torch.zeros_like(layer_inner).unsqueeze(0).repeat(self.rnn_layer, 1, 1),
             torch.zeros_like(layer_inner).unsqueeze(0).repeat(self.rnn_layer, 1, 1))  # [0]  # v1

        for i, con in enumerate(self.convs):
            if self.reason_flag:
                q = layer_inner
                # (4419,100), ((1,4419,100)*2) << (1,4419,100), ((1,4419,100)*2)
                # output, (hn, cn) = rnn(input, (h0, c0))
                layer_inner, h = self.rnn(q.unsqueeze(0), h)
                layer_inner = layer_inner.squeeze(0)

            layer_inner = self.act_fn(con(layer_inner, adj, _layers[0], self.lamda, self.alpha, i + 1))
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            if self.reason_flag:
                layer_inner += q
                # layer_inner = torch.cat([layer_inner, q], dim=-1)
            if test_label:
                print('# deepGCN layer ' + str(i))
                print(layer_inner.size())
                import numpy as np
                import os
                if not os.path.isdir('../outputs/iemocap/'): os.makedirs('../outputs/iemocap/')
                np.save("../outputs/iemocap/1080_v1_test_output_layer_{}".format(i), layer_inner.data.cpu().numpy())

        if self.use_residue:
            layer_inner = torch.cat([x, layer_inner], dim=-1)  # 300 << 200,100

        if not self.return_feature:
            layer_inner = self.fcs[-1](layer_inner)
            layer_inner = F.log_softmax(layer_inner, dim=1)
        return layer_inner

    def message_passing_wo_speaker(self, x, dia_len, topicLabel):
        adj = torch.zeros((x.shape[0], x.shape[0]))
        start = 0
        for i in range(len(dia_len)):
            sub_adj = torch.zeros((dia_len[i], dia_len[i]))
            temp = x[start:start + dia_len[i]]
            vec_length = torch.sqrt(torch.sum(temp.mul(temp), dim=1))
            norm_temp = (temp.permute(1, 0) / vec_length)
            cos_sim_matrix = torch.sum(torch.matmul(norm_temp.unsqueeze(2), norm_temp.unsqueeze(1)), dim=0)
            cos_sim_matrix = cos_sim_matrix * 0.99999
            sim_matrix = torch.acos(cos_sim_matrix)

            d = sim_matrix.sum(1)
            D = torch.diag(torch.pow(d, -0.5))

            sub_adj[:dia_len[i], :dia_len[i]] = D.mm(sim_matrix).mm(D)
            adj[start:start + dia_len[i], start:start + dia_len[i]] = sub_adj
            start += dia_len[i]

        adj = adj.cuda()

        return adj

    def atom_calculate_edge_weight(self, x, y):
        f = self.cossim(x, y)
        if f > 1 and f < 1.05:
            f = 1
        elif f < -1 and f > -1.05:
            f = -1
        elif f >= 1.05 or f <= -1.05:
            print('cos = {}'.format(f))
        return f

    def message_passing_directed_speaker(self, x, dia_len, qmask):
        total_len = sum(dia_len)
        adj = torch.zeros((total_len, total_len)) + torch.eye(total_len)
        start = 0
        use_utterance_edge = False
        for (i, len_) in enumerate(dia_len):
            speaker0 = []
            speaker1 = []
            for (j, speaker) in enumerate(qmask[i][0:len_]):
                if speaker[0] == 1:
                    speaker0.append(j)
                else:
                    speaker1.append(j)
            if use_utterance_edge:
                for j in range(len_ - 1):
                    f = self.atom_calculate_edge_weight(x[start + j], x[start + j + 1])
                    Aij = 1 - math.acos(f) / math.pi
                    adj[start + j][start + j + 1] = Aij
                    adj[start + j + 1][start + j] = Aij
            for k in range(len(speaker0) - 1):
                f = self.atom_calculate_edge_weight(x[start + speaker0[k]], x[start + speaker0[k + 1]])
                Aij = 1 - math.acos(f) / math.pi
                adj[start + speaker0[k]][start + speaker0[k + 1]] = Aij
                adj[start + speaker0[k + 1]][start + speaker0[k]] = Aij
            for k in range(len(speaker1) - 1):
                f = self.atom_calculate_edge_weight(x[start + speaker1[k]], x[start + speaker1[k + 1]])
                Aij = 1 - math.acos(f) / math.pi
                adj[start + speaker1[k]][start + speaker1[k + 1]] = Aij
                adj[start + speaker1[k + 1]][start + speaker1[k]] = Aij

            start += dia_len[i]

        d = adj.sum(1)
        D = torch.diag(torch.pow(d, -0.5))
        adj = D.mm(adj).mm(D).cuda()

        return adj.cuda()

    def message_passing_relation_graph(self, x, dia_len):
        total_len = sum(dia_len)
        adj = torch.zeros((total_len, total_len)) + torch.eye(total_len)
        window_size = 10
        start = 0
        for (i, len_) in enumerate(dia_len):
            edge_set = []
            for k in range(len_):
                left = max(0, k - window_size)
                right = min(len_ - 1, k + window_size)
                edge_set = edge_set + [str(i) + '_' + str(j) for i in range(left, right) for j in range(i + 1, right + 1)]
            edge_set = [[start + int(str_.split('_')[0]), start + int(str_.split('_')[1])] for str_ in list(set(edge_set))]
            for left, right in edge_set:
                f = self.atom_calculate_edge_weight(x[left], x[right])
                Aij = 1 - math.acos(f) / math.pi
                adj[left][right] = Aij
                adj[right][left] = Aij
            start += dia_len[i]

        d = adj.sum(1)
        D = torch.diag(torch.pow(d, -0.5))
        adj = D.mm(adj).mm(D).cuda()

        return adj.cuda()


class DAGERC_fusion(nn.Module):

    def __init__(self, args, num_class):
        super().__init__()
        self.args = args
        # gcn layer

        self.dropout = nn.Dropout(args.dropout)

        self.gnn_layers = args.gnn_layers

        if not args.no_rel_attn:
            self.rel_attn = True
        else:
            self.rel_attn = False

        if self.args.attn_type == 'linear':#计算边的权重公式
            gats = []
            for _ in range(args.gnn_layers):#构造线性层，如果存在关系类型的话则使用关系类型的线性层
                gats += [GatLinear(args.hidden_dim) if args.no_rel_attn else GatLinear_rel(args.hidden_dim)]
            self.gather = nn.ModuleList(gats)
        elif self.args.attn_type == 'dotprod':
            gats = []
            for _ in range(args.gnn_layers):
                gats += [GatDot(args.hidden_dim) if args.no_rel_attn else GatDot_rel(args.hidden_dim)]
            self.gather = nn.ModuleList(gats)
        
        elif self.args.attn_type == 'rgcn':
            gats = []
            for _ in range(args.gnn_layers):
                # gats += [GAT_dialoggcn(args.hidden_dim)]
                gats += [GAT_dialoggcn_v1(args.hidden_dim)]
            self.gather = nn.ModuleList(gats)

        grus_c = []#构造GRU
        for _ in range(args.gnn_layers):
            grus_c += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus_c = nn.ModuleList(grus_c)

        grus_p = []
        for _ in range(args.gnn_layers):
            grus_p += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus_p = nn.ModuleList(grus_p)
        
        fcs = []
        for _ in range(args.gnn_layers):
            fcs += [nn.Linear(args.hidden_dim * 2, args.hidden_dim)]
        self.fcs = nn.ModuleList(fcs)

        self.fc1 = nn.Linear(args.emb_dim, args.hidden_dim)
        self.nodal_att_type = args.nodal_att_type
        
        in_dim = args.hidden_dim * (args.gnn_layers + 1) + args.emb_dim
        self.fc2 = nn.Linear(in_dim, 600)
        # output mlp layers
        layers = [nn.Linear(in_dim, args.hidden_dim), nn.ReLU()]
        for _ in range(args.mlp_layers - 1):
            layers += [nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU()]
        layers += [self.dropout]
        layers += [nn.Linear(args.hidden_dim, num_class)]

        self.out_mlp = nn.Sequential(*layers)

        self.attentive_node_features = attentive_node_features(in_dim)

    def forward(self, features, adj, s_mask, s_mask_onehot, lengths):
        '''
        :param features: (B, N, D)
        :param adj: (B, N, N)
        :param s_mask: (B, N, N)
        :param s_mask_onehot: (B, N, N, 2)
        :return:
        '''
        num_utter = features.size()[1]
        H0 = F.relu(self.fc1(features))#(16,103,300)
        # H0 = self.dropout(H0)
        H = [H0]
        for l in range(self.args.gnn_layers):
            C = self.grus_c[l](H[l][:,0,:]).unsqueeze(1) #过一个GRU层，维度保持不变(16,1,300),获取会话中的第一句话
            M = torch.zeros_like(C).squeeze(1) #(16,300),刚开始的第一个句子并没有初始状态的输入
            # P = M.unsqueeze(1) 
            P = self.grus_p[l](M, H[l][:,0,:]).unsqueeze(1)
            #H1 = F.relu(self.fcs[l](torch.cat((C,P) , dim = 2)))  
            #H1 = F.relu(C+P)
            H1 = C+P
            for i in range(1, num_utter):
                # print(i,num_utter)
                if self.args.attn_type == 'rgcn':
                    _, M = self.gather[l](H[l][:,i,:], H1, H1, adj[:,i,:i], s_mask[:,i,:i])
                    # _, M = self.gather[l](H[l][:,i,:], H1, H1, adj[:,i,:i], s_mask_onehot[:,i,:i,:])
                else:
                    if not self.rel_attn:
                        _, M = self.gather[l](H[l][:,i,:], H1, H1, adj[:,i,:i])
                    else:
                        _, M = self.gather[l](H[l][:,i,:], H1, H1, adj[:,i,:i], s_mask[:, i, :i])
                C = self.grus_c[l](H[l][:,i,:], M).unsqueeze(1)
                P = self.grus_p[l](M, H[l][:,i,:]).unsqueeze(1)   
                # P = M.unsqueeze(1)
                #H_temp = F.relu(self.fcs[l](torch.cat((C,P) , dim = 2)))  
                #H_temp = F.relu(C+P)
                H_temp = C+P
                H1 = torch.cat((H1 , H_temp), dim = 1)#把第一句话给拼接上  
                # print('H1', H1.size())
                # print('----------------------------------------------------')
            H.append(H1)
        H.append(features)
        
        H = torch.cat(H, dim = 2) 

        H = self.attentive_node_features(H,lengths,self.nodal_att_type) 
        logits = self.out_mlp(H)

        #return logits
        return H



class DIFMultiHeadAttention(nn.Module):
    """
    DIF Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(self, n_heads, hidden_size,attribute_hidden_size,feat_num, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,fusion_type,max_len):
        super(DIFMultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.attribute_attention_head_size = [int(_ / n_heads) for _ in attribute_hidden_size]
        self.attribute_all_head_size = [self.num_attention_heads * _ for _ in self.attribute_attention_head_size]
        self.fusion_type = fusion_type
        self.max_len = max_len

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.query_p = nn.Linear(hidden_size, self.all_head_size)
        self.key_p = nn.Linear(hidden_size, self.all_head_size)

        self.feat_num = feat_num
        self.query_layers = nn.ModuleList([copy.deepcopy(nn.Linear(attribute_hidden_size[_], self.attribute_all_head_size[_])) for _ in range(self.feat_num)])
        self.key_layers = nn.ModuleList(
            [copy.deepcopy(nn.Linear(attribute_hidden_size[_], self.attribute_all_head_size[_])) for _ in range(self.feat_num)])

        if self.fusion_type == 'concat':
            self.fusion_layer = nn.Linear(self.max_len, self.max_len)
        elif self.fusion_type == 'gate':
            self.fusion_layer = VanillaAttention(self.max_len,self.max_len)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_for_scores_attribute(self, x,i):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attribute_attention_head_size[i])
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor,attribute_table,position_embedding):
        item_query_layer = self.transpose_for_scores(self.query(input_tensor))
        item_key_layer = self.transpose_for_scores(self.key(input_tensor))
        item_value_layer = self.transpose_for_scores(self.value(input_tensor))

        pos_query_layer = self.transpose_for_scores(self.query_p(position_embedding))
        pos_key_layer = self.transpose_for_scores(self.key_p(position_embedding))

        item_attention_scores = torch.matmul(item_query_layer, item_key_layer.transpose(-1, -2))
        pos_scores = torch.matmul(pos_query_layer, pos_key_layer.transpose(-1, -2))

        attribute_attention_table = []

        for i, (attribute_query, attribute_key) in enumerate(
                zip(self.query_layers, self.key_layers)):
            attribute_tensor = attribute_table.squeeze(-2)
            attribute_query_layer = self.transpose_for_scores_attribute(attribute_query(attribute_tensor),i)
            attribute_key_layer = self.transpose_for_scores_attribute(attribute_key(attribute_tensor),i)
            attribute_attention_scores = torch.matmul(attribute_query_layer, attribute_key_layer.transpose(-1, -2))
            attribute_attention_table.append(attribute_attention_scores.unsqueeze(-2))
        attribute_attention_table = torch.cat(attribute_attention_table,dim=-2)
        table_shape = attribute_attention_table.shape
        feat_atten_num, attention_size = table_shape[-2], table_shape[-1]
        
        if self.fusion_type == 'sum':
            attention_scores = torch.sum(attribute_attention_table, dim=-2)
            attention_scores = attention_scores + item_attention_scores + pos_scores
        elif self.fusion_type == 'concat':
            attention_scores = attribute_attention_table.view(table_shape[:-2] + (feat_atten_num * attention_size,))
            attention_scores = torch.cat([attention_scores, item_attention_scores, pos_scores], dim=-1)
            attention_scores = self.fusion_layer(attention_scores)
        elif self.fusion_type == 'gate':
            attention_scores = torch.cat(
                [attribute_attention_table, item_attention_scores.unsqueeze(-2), pos_scores.unsqueeze(-2)], dim=-2)
            attention_scores,_ = self.fusion_layer(attention_scores)

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        #attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)


        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, item_value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class DIFTransformerLayer(nn.Module):
    """
    One decoupled transformer layer consists of a decoupled multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(
        self, n_heads, hidden_size,attribute_hidden_size,feat_num, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
        layer_norm_eps,fusion_type,max_len
    ):
        super(DIFTransformerLayer, self).__init__()
        self.multi_head_attention = DIFMultiHeadAttention(
            n_heads, hidden_size,attribute_hidden_size, feat_num, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps,fusion_type,max_len,
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states,attribute_embed,position_embedding):
        attention_output = self.multi_head_attention(hidden_states,attribute_embed,position_embedding)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output

class DIFTransformerEncoder(nn.Module):
    r""" One decoupled TransformerEncoder consists of several decoupled TransformerLayers.

        - n_layers(num): num of transformer layers in transformer encoder. Default: 2
        - n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        - hidden_size(num): the input and output hidden size. Default: 64
        - attribute_hidden_size(list): the hidden size of attributes. Default:[64]
        - feat_num(num): the number of attributes. Default: 1
        - inner_size(num): the dimensionality in feed-forward layer. Default: 256
        - hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        - attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        - hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        - layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12
        - fusion_type(str): fusion function used in attention fusion module. Default: 'sum'
                            candidates: 'sum','concat','gate'

    """

    def __init__(
        self,
        n_layers=2,
        n_heads=5,
        hidden_size=700,
        attribute_hidden_size=[700],
        feat_num=1,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act='gelu',
        layer_norm_eps=1e-12,
        fusion_type = 'sum',
        max_len = None
    ):

        super(DIFTransformerEncoder, self).__init__()
        layer = DIFTransformerLayer(
            n_heads, hidden_size,attribute_hidden_size,feat_num, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps,fusion_type,max_len
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states,attribute_hidden_states,position_embedding, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attribute_hidden_states,
                                                                  position_embedding)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class Global_GNN(nn.Module):
    def __init__(self, fuse_A, hidden_size, step=1):
        super(Global_GNN, self).__init__()
        self.fuse_A = fuse_A
        self.step = step
        self.hidden_size = hidden_size
        self.w_h = Parameter(torch.Tensor(self.hidden_size*3, self.hidden_size))
        self.w_hf = Parameter(torch.Tensor(self.hidden_size*2, self.hidden_size))

    def GateCell(self, A, hidden):
        hidden_w = F.linear(hidden, self.w_h)

        hidden_w_0, hidden_w_1, hidden_w_2 = hidden_w.chunk(3, -1)
        hidden_fuse_0, hidden_fuse_1 = F.linear(torch.matmul(A, hidden_w_0), self.w_hf).chunk(2, -1)

        gate = torch.relu(hidden_fuse_0 + hidden_w_1)
        return hidden_w_2 + gate * hidden_fuse_1

    def Fuse_with_correlation(self, A_Global, hidden):
        correlation_A = torch.matmul(hidden, hidden.transpose(1, 0))
        correlation_A_std = torch.norm(correlation_A, p=2, dim=1, keepdim=True)
        correlation_A = correlation_A/correlation_A_std
        return A_Global + correlation_A

    def forward(self, A_Global, hidden):
        seqs = []
        if self.fuse_A:
            A_Global = self.Fuse_with_correlation(A_Global, hidden)
        for i in range(self.step):
            hidden = self.GateCell(A_Global, hidden)
            seqs.append(hidden)
        return hidden, torch.mean(torch.stack(seqs, dim=1), dim=1)


class CrossModalTransformerEncoder(nn.Module):

    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False):
        super().__init__()
        self.dropout = embed_dropout      # Embedding dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        
        self.attn_mask = attn_mask
        self.x_q_fc = nn.Linear(embed_dim, embed_dim)
        self.x_1_k_fc = nn.Linear(embed_dim, embed_dim)
        self.x_1_v_fc = nn.Linear(embed_dim, embed_dim)

        
        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = CrossModalTransformerEncoderLayer(embed_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                relu_dropout=relu_dropout,
                                                res_dropout=res_dropout,
                                                attn_mask=attn_mask)
            self.layers.append(new_layer)

        self.register_buffer('version', torch.Tensor([2]))


    def forward(self, x_in, x_in_k = None, x_in_v = None):

        x_q = self.x_q_fc(x_in)
        x_q = self.embed_scale * x_q
        x_q = F.dropout(x_q, p=self.dropout, training=self.training)


        x_k = self.x_1_k_fc(x_in_k)
        x_v = self.x_1_v_fc(x_in_v)
        
        x_k = self.embed_scale * x_k
        x_k = F.dropout(x_k, p=self.dropout, training=self.training)

        x_v = self.embed_scale * x_v
        x_v = F.dropout(x_v, p=self.dropout, training=self.training)
        
        # encoder layers
        intermediates = [x_in]
        for index,layer in enumerate(self.layers):
            x_q = layer(x_q, x_k, x_v)
            intermediates.append(x_q)

        return x_q


class CrossModalTransformerEncoderLayer(nn.Module):

    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                 attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )
        self.attn_mask = attn_mask

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.fc1 = nn.Linear(self.embed_dim, 4*self.embed_dim)   # The "Add & Norm" part in the paper
        self.fc2 = nn.Linear(4*self.embed_dim, self.embed_dim)
        self.layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x_q, x_k=None, x_v=None):

        residual = x_q
        x_q = self.layer_norm(x_q)
        x_k = self.layer_norm(x_k)
        x_v = self.layer_norm(x_v)

        x_att, _ = self.self_attn(query=x_q, key=x_k, value=x_v)
        x_att = F.dropout(x_att, p=self.res_dropout, training=self.training)
        x_att = residual + x_att
        x_att = self.layer_norm(x_att)

        # FFD
        residual = x_att
        x_att = F.relu(self.fc1(x_att))
        x_att = F.dropout(x_att, p=self.relu_dropout, training=self.training)
        x_att = self.fc2(x_att)
        x_att = F.dropout(x_att, p=self.res_dropout, training=self.training)
        x_att = residual + x_att
        x_att = self.layer_norm(x_att)
        return x_att


class DecoupleFusion(nn.Module):
    def __init__(self, embed, head, layer):
        super().__init__()
        self.embed=embed
        self.head=head
        self.layer=layer
        # 2.1 Modality-specific encoder
        self.encoder_1 = nn.Linear(self.embed, self.embed, bias=False)
        self.encoder_2 = nn.Linear(self.embed, self.embed, bias=False)
        self.encoder_3 = nn.Linear(self.embed, self.embed, bias=False)
        
        self.wet_1_2 = nn.Linear(self.embed, self.embed)
        self.wet_1_3 = nn.Linear(self.embed, self.embed)

        self.wet_2_1 = nn.Linear(self.embed, self.embed)
        self.wet_2_3 = nn.Linear(self.embed, self.embed)

        self.wet_3_1 = nn.Linear(self.embed, self.embed)
        self.wet_3_2 = nn.Linear(self.embed, self.embed)

        self.wet = nn.Linear(self.embed*2, self.embed*2)

        # 2.2 Modality-invariant encoder
        self.encoder_c = nn.Linear(self.embed, self.embed, bias=False)
        
        # Modality-invariant conv2d
        self.conv2d_c = nn.Conv2d(3, 1, (1,2), stride=(1,2))

        # Modality-specific fusion
        self.fusion_1 = CrossModalTransformerEncoder(self.embed, self.head, self.layer)
        self.fusion_2 = CrossModalTransformerEncoder(self.embed, self.head, self.layer)
        self.fusion_3 = CrossModalTransformerEncoder(self.embed, self.head, self.layer)

    def forward(self, x_1, x_2, x_3):

        # specific-fusion
        x_1_s = self.encoder_1(x_1)
        x_2_s = self.encoder_2(x_2)
        x_3_s = self.encoder_3(x_3)
        
        # (2,3)->1
        x_1_2_s = self.fusion_1(x_1_s, x_2_s, x_2_s)
        x_1_3_s = self.fusion_1(x_1_s, x_3_s, x_3_s)
        x_1_2_s_wet = F.sigmoid(self.wet_1_2(x_1_2_s))
        x_1_3_s_wet = F.sigmoid(self.wet_1_3(x_1_3_s))
        x_1_fusion = torch.cat([x_1_2_s*x_1_2_s_wet, x_1_3_s*x_1_3_s_wet], dim=-1)

        # (1,3)->2
        x_2_1_s = self.fusion_2(x_2_s, x_1_s, x_1_s)
        x_2_3_s = self.fusion_2(x_2_s, x_3_s, x_3_s)
        x_2_1_s_wet = F.sigmoid(self.wet_2_1(x_2_1_s))
        x_2_3_s_wet = F.sigmoid(self.wet_2_3(x_2_3_s))
        x_2_fusion = torch.cat([x_2_1_s*x_2_1_s_wet, x_2_3_s*x_2_3_s_wet], dim=-1)

        # (1,2)->3
        x_3_1_s = self.fusion_3(x_3_s, x_1_s, x_1_s)
        x_3_2_s = self.fusion_3(x_3_s, x_2_s, x_2_s)
        x_3_1_s_wet = F.sigmoid(self.wet_3_1(x_3_1_s))
        x_3_2_s_wet = F.sigmoid(self.wet_3_2(x_3_2_s))
        x_3_fusion = torch.cat([x_3_1_s*x_3_1_s_wet, x_3_2_s*x_3_2_s_wet], dim=-1)

        x_1_fusion_wet = F.log_softmax(self.wet(x_1_fusion))
        x_2_fusion_wet = F.log_softmax(self.wet(x_2_fusion))
        x_3_fusion_wet = F.log_softmax(self.wet(x_3_fusion))

        x_specific = x_1_fusion*x_1_fusion_wet + x_2_fusion*x_2_fusion_wet + x_3_fusion*x_3_fusion_wet

        #invariant fuison
        ls=[]
        ls_1=[]
        x_1_c = self.encoder_c(x_1)
        x_2_c = self.encoder_c(x_2)
        x_3_c = self.encoder_c(x_3)
        for i in range(x_1.size(0)):
            ls_1.append(x_1[i])
            ls_1.append(x_2[i])
            ls_1.append(x_3[i])
            ls.append(torch.stack(ls_1))
            ls_1=[]
        x_invariant = torch.stack(ls)
        x_invariant = self.conv2d_c(x_invariant).squeeze(1)

        return torch.cat((x_invariant, x_specific), dim=-1)
        #return x_invariant
        #return x_specific














