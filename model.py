from random import betavariate
from re import S
import re
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.onnx.symbolic_opset9 import dim, unsqueeze
from transformers import BertModel,RobertaModel

from bert import BertPreTrainedModel,RobertaPreTrainedModel
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
from vae import VAE
from CLLOSS import batched_contrastive_loss_1
from transformer import TransformerEncoder
class BertSent_Encode(BertPreTrainedModel):
    def __init__(self, bert_config):
        """
        :param bert_config: configuration for bert model
        """
        super(BertSent_Encode, self).__init__(bert_config)
        self.bert_config = bert_config
        self.bert = BertModel(bert_config)


    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        return outputs[0]
    
class RobertaStance(RobertaPreTrainedModel):
    def __init__(self, roberta_config):
        """
        :param bert_config: configuration for bert model
        """
        super(RobertaStance, self).__init__(roberta_config)
        self.roberta_config = roberta_config
        self.roberta = RobertaModel(roberta_config)
        penultimate_hidden_size = roberta_config.hidden_size*2
        self.g1 = nn.Linear(100, 768)#线性层，将输入的100维特征转换为768维特征
        self.g1_c = nn.Linear(100, 768)#线性层，将输入的100维特征转换为768维特征
        self.g1_drop = nn.Dropout(0.1)#dropout层，用于随机丢弃输入的一部分特征
        self.LayerNorm = nn.LayerNorm(768, eps=1e-8, elementwise_affine=True)
        self.linear = nn.Linear(1536, 768)
        #decode
        self.d1 = nn.Linear(768, 768)#两个线性层，用于特征解码
        self.d2 = nn.Linear(768, 100)
        self.d1_c = nn.Linear(768, 768)#两个线性层，用于特征解码
        self.d2_c = nn.Linear(768, 100)
        self.d_drop = nn.Dropout(0.1)#dropout层，用于随机丢弃解码后的特征
        self.sent_loss = CrossEntropyLoss()
        self.sent_cls = nn.Linear(penultimate_hidden_size, roberta_config.sent_number)
        self.vae = VAE(embedding_dimension=100,z_dimension=100)
        self.transformer = TransformerEncoder(hidden_dim=768,heads=4,layers=2)
    def encode2(self, x2):#x2表示输入的图的特征
        x2 = self.g1(x2)#通过线性层self.g1进行特征转换
        x2 = F.relu(x2)#进行ReLU激活函数操作，增加非线性性
        x2 = self.g1_drop(x2)#随机丢弃一部分特征，以防止过拟合(这是将参数进行失活，哪是丢弃特征噢！！)
        return x2

    def decode(self, z):
        z = self.d1(z)
        z = F.relu(z)
        z = self.d_drop(z)
        z = self.d2(z)
        return z 

    def loss_ae(self,recon_x, x):#recon_x和x分别表示重构的输入和原始输入
        dim = x.size(1)
        MSE = F.mse_loss(recon_x, x.view(-1, dim), reduction='mean')#F.mse_loss计算重构输入recon_x与原始输入x之间的均方差
        return MSE
    
    def loss_function(self,recon_out,graph_feature,mean,std):
        MSE = F.mse_loss(recon_out, graph_feature, reduction='mean')
        var = torch.pow(torch.exp(std),2)
        KLD = -0.5 * torch.sum(1+torch.log(var)-torch.pow(mean,2)-var)
        loss = MSE+KLD
        return loss
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, sent_labels=None,
                position_ids=None, graph_feature =None, graph_feature_c=None):
        outputs = self.roberta(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,#这块代码用于文本特征提取
                            attention_mask=attention_mask)
        hidden = outputs[0]

        w1 = hidden[:,0,:]#从hidden中获取第一个位置的隐藏状态w1

        z_ae1 = self.encode2(graph_feature)#将graph_feature编码为z2
        reconstruct_ae1 = self.decode(z_ae1)
        z_ae2 = self.encode2(graph_feature_c)
        reconstruct_ae2 = self.decode(z_ae2)

        reconstruct_vae1,mean,std = self.vae(graph_feature)
        reconstruct_vae2,mean_c,std_c = self.vae(graph_feature_c)
        z = z_ae1+z_ae2
        z = self.transformer(z.unsqueeze(0))
        z = z.squeeze(0)
        w1 = self.transformer(w1.unsqueeze(0))
        w1 = w1.squeeze(0)
        f = torch.cat((w1,z),dim = 1)#将w1和z2在维度1上拼接为f
       #计算重构损失recon_loss，衡量重构输入reconstruct与原始输入graph_feature之间的差异
        recon_loss_ae1 = self.loss_ae(reconstruct_ae1,graph_feature)
        recon_loss_ae2 = self.loss_ae(reconstruct_ae2,graph_feature_c)
        recon_loss_ae = recon_loss_ae1+ recon_loss_ae2
        recon_loss_vae1 = self.loss_function(reconstruct_vae1,graph_feature,mean,std)
        recon_loss_vae2 = self.loss_function(reconstruct_vae2,graph_feature_c,mean_c,std_c)
        recon_loss_vae = recon_loss_vae1+recon_loss_vae2
        recon_loss = recon_loss_ae+recon_loss_vae 
        #graph feature对比学习
        loss_cl_feature1=batched_contrastive_loss_1(graph_feature, graph_feature_c)
        loss_cl_feature2=batched_contrastive_loss_1(graph_feature_c, graph_feature)
        loss_cl_feature=(loss_cl_feature1+loss_cl_feature2)/2.0
        #intra对比损失
        loss_cl_up1 = batched_contrastive_loss_1(reconstruct_ae1,reconstruct_vae1)
        loss_cl_up2 = batched_contrastive_loss_1(reconstruct_vae1,reconstruct_ae1)
        loss_cl_up = (loss_cl_up1+loss_cl_up2)/2.0
        loss_cl_down1 = batched_contrastive_loss_1(reconstruct_ae2,reconstruct_vae2)
        loss_cl_down2 = batched_contrastive_loss_1(reconstruct_vae2,reconstruct_ae2)
        loss_cl_down = (loss_cl_down1+loss_cl_down2)/2.0
        loss_cl_intra = loss_cl_up+loss_cl_down
        #inter对比损失
        loss_cl_ae1 = batched_contrastive_loss_1(reconstruct_ae1,reconstruct_ae2)
        loss_cl_ae2 = batched_contrastive_loss_1(reconstruct_ae2,reconstruct_ae1)
        loss_cl_ae = (loss_cl_ae1+loss_cl_ae2)/2.0
        loss_cl_vae1 = batched_contrastive_loss_1(reconstruct_vae1,reconstruct_vae2)
        loss_cl_vae2 = batched_contrastive_loss_1(reconstruct_vae2,reconstruct_vae1)
        loss_cl_vae = (loss_cl_vae1+loss_cl_vae2)/2.0
        loss_cl_inter = loss_cl_ae+loss_cl_vae

        loss_cl = loss_cl_feature+loss_cl_intra+loss_cl_inter
       #通过self.sent_cls对f进行分类，得到句子级别的预测结果sent_logits
        sent_logits = self.sent_cls(f)
       #损失函数
        if len(sent_logits.shape) == 1:
            sent_logits = sent_logits.unsqueeze(0)#.unsqueeze(0)增加维度，（0表示，在第一个位置增加维度）
        sent_loss = self.sent_loss(sent_logits, sent_labels)
        
        loss =sent_loss+recon_loss+loss_cl
    
        return sent_logits,loss

class BertStance(BertPreTrainedModel):
    def __init__(self, bert_config):
        """
        :param bert_config: configuration for bert model
        """
        super(BertStance, self).__init__(bert_config)
        self.bert_config = bert_config
        self.bert = BertModel(bert_config)

        penultimate_hidden_size = bert_config.hidden_size*2

        #encode
        self.g1 = nn.Linear(100, 768)
        self.g1_drop = nn.Dropout(0.1)
        #decode
        self.d1 = nn.Linear(768, 768)
        self.d2 = nn.Linear(768, 100)
        self.d_drop = nn.Dropout(0.1)
        self.sent_loss = CrossEntropyLoss()
        self.sent_cls = nn.Linear(penultimate_hidden_size, bert_config.sent_number)

    def encode2(self, x2):
        x2 = self.g1(x2)
        x2 = F.relu(x2)
        x2 = self.g1_drop(x2)
        return x2

    def decode(self, z):
        z = self.d1(z)
        z = F.relu(z)
        z = self.d_drop(z)
        z = self.d2(z)
        return z 

    def loss_ae(self,recon_x, x):
        dim = x.size(1)
        MSE = F.mse_loss(recon_x, x.view(-1, dim), reduction='mean')
        return MSE

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, sent_labels=None,
                position_ids=None, head_mask=None,graph_feature =None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        hidden = outputs[0]
        w1 = hidden[:,0,:]


        z2 = self.encode2(graph_feature)
        reconstruct = self.decode(z2)
        f = torch.cat((w1,z2),dim = 1)

        recon_loss = self.loss_ae(reconstruct,graph_feature)

        sent_logits = self.sent_cls(f)

        if len(sent_logits.shape) == 1:
            sent_logits = sent_logits.unsqueeze(0)
        sent_loss = self.sent_loss(sent_logits, sent_labels)
        loss =sent_loss+recon_loss

        return sent_logits,loss

class LSTMStance(nn.Module):
    def __init__(self, embedding_dim,hidden_dim,sent_size,dropout=0.5):
        """

        :param bert_config: configuration for bert model
        """
        super(LSTMStance, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True, batch_first=True,dropout=dropout)
        self.sent_loss = CrossEntropyLoss()
        self.sent_cls = nn.Linear(hidden_dim*2+100, sent_size)

    def forward(self, embedding=None, sent_labels=None, SentEmb = None, attention_mask=None,graph_features = None,meg='Encode'):
        hidden, n = self.lstm(embedding)
        h = torch.bmm(torch.transpose(hidden,1,2),attention_mask.unsqueeze(2).to(torch.float32)).squeeze()
        if meg == 'Encode':
            return h
        if len(SentEmb.shape) == 1:
                SentEmb = SentEmb.unsqueeze(0)
                h = h.unsqueeze(0)
        SentEmb = torch.cat((SentEmb,h),dim=1)
        SentEmb = torch.cat((SentEmb,graph_features),dim=1)
        sent_logits = self.sent_cls(SentEmb)
       
        sent_loss = self.sent_loss(sent_logits, sent_labels)
        return sent_logits,sent_loss