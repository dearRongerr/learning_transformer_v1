import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
import numpy as np
import copy

# 调用封装好的类
embedding = nn.Embedding(10,3)
input1 = torch.LongTensor([[1,2,5,4],[4,3,2,9]])
print(embedding(input1))

# 创建Embedding类来实现文本嵌入层
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        # d_model : 词嵌入的维度
        # vocab : 词表的大小
        super(Embeddings,self).__init__()
        # 定义Embedding层
        self.lut = nn.Embedding(vocab,d_model)
        # 将参数传入类中
        self.d_model = d_model
    def forward(self,x):
        # x: 代表输入进模型的文本通过词汇映射后的数字张量
        return self.lut(x) * math.sqrt(self.d_model)
    
d_model = 512
vocab = 1000

x = Variable(torch.LongTensor([[100,2,421,508],[491,998,1,221]]))

emb = Embeddings(d_model,vocab)
embr = emb(x)
print("embr：",embr)
print(embr.shape)




