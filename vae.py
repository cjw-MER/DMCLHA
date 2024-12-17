import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 检查CUDA是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class VAE(nn.Module):
    def __init__(self,embedding_dimension,z_dimension):
        super().__init__()
        # 定义编码器
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dimension, 50),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(50, 25),
            nn.LeakyReLU(0.2, inplace=True),
        ).to(device) 

        # 编码器输出均值和标准差
        self.encoder_fc1=nn.Linear(25,z_dimension)
        self.encoder_fc2=nn.Linear(25,z_dimension)
        
        # 定义解码器
        #self.decoder_fc = nn.Linear(z_dimension, 64)
        self.decoder = nn.Sequential(
            nn.Linear(z_dimension, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, embedding_dimension),
            # nn.ReLU(inplace=True),
            # nn.Linear(256, embedding_dimension),
        ).to(device)

    def noise_reparameterize(self,mean,logvar):
        # 用标准正态分布采样任意高斯分布
        eps = torch.randn(mean.shape).to(mean.device)
        z = mean + eps * torch.exp(logvar)
        return z
    
    def forward(self,graph_features):
        # 通过编码器进行前向传播
        z = self.encoder(graph_features)
        mean = self.encoder_fc1(z)
        logstd = self.encoder_fc2(z)
        # 根据均值和方差进行采样
        z = self.noise_reparameterize(mean, logstd)
        # 采样后的数据送入解码器
        out = self.decoder(z)
        return out,mean,logstd     